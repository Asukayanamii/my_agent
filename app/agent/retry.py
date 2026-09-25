"""
上游模型调用的重试：错误分类 + 指数退避 + 可见性。

照 Pi（earendil-works/pi）的做法——它把重试分成两层：SDK/provider 一层，agent 一层。
我们只保留 **agent 一层**，并把 SDK 自带的重试关掉（`models.build_model(max_retries=0)`）：

- **SDK 的重试是隐形的**：不记日志、次数会和我们的叠加（我们的 3 次 × SDK 的 2 次），
  延迟也不可控。Pi 为此显式用 `maxRetries: 0` 调 SDK，再由自己这层统一处理。
- **分类决定值不值得重试**（Pi 的 `isRetryableAssistantError` 原样搬过来）：
  配额/计费类（`insufficient_quota`、billing、"monthly usage limit"…）与鉴权/参数错误
  **立刻失败**——重试只会白等，有些网关还会一路等到额度恢复；限流、5xx、网络与流中断
  才是瞬态，值得退避重试。
- **延迟**：`2s × 2^(n-1)`，单次上限 60 秒；provider 明确要求 `Retry-After` 时听它的，
  但**超过上限就直接失败并说明原因**，不静默等几分钟（Pi 的 `validateServerRetryDelayMs`）。
- **已经吐过字的调用不重试**：前端没法把已经渲染的半截回答撤回去，重试只会把它再拉一遍，
  看起来像答了两遍。要支持得先有"重置本轮文本"的事件，见 LESSONS。
  **例外是"流被掐断"**（`stream_was_interrupted`）：残句本来就该丢，重试补的是本来拿不到的结果。
"""

import asyncio
import logging
import re

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
"""最多重试几次（首次调用不算）。Pi 的 `retry.maxRetries` 默认也是 3。"""

BASE_DELAY_SECONDS = 2.0
"""退避基数：第 n 次重试前等 `BASE_DELAY_SECONDS × 2^(n-1)`（2s、4s、8s）。"""

MAX_DELAY_SECONDS = 60.0
"""单次退避上限（Pi 的 `retry.maxAgentDelayMs` 默认 60000）。"""

NON_RETRYABLE_PATTERN = re.compile(
    "|".join(
        [
            # 配额、计费、订阅额度：不是"暂时忙"，等到天荒地老也不会好
            "insufficient_quota",
            "out of budget",
            "quota exceeded",
            "usage limit",
            "available balance",
            "billing",
        ]
    ),
    re.IGNORECASE,
)

RETRYABLE_PATTERN = re.compile(
    "|".join(
        [
            # 限流与服务端瞬态
            "overloaded",
            "rate.?limit",
            "too many requests",
            "service.?unavailable",
            "server.?error",
            "internal.?error",
            # 网关/中转的瞬时上游故障（OpenRouter 的 "Provider returned error" 之类）
            "provider.?returned.?error",
            "upstream",
            # 网络与传输中断
            "network.?error",
            "connection.?error",
            "connection.?refused",
            "connection.?lost",
            "other side closed",
            "fetch failed",
            "getaddrinfo",
            "ENOTFOUND",
            "EAI_AGAIN",
            "reset before headers",
            "socket hang up",
            "socket connection was closed",
            "timed? out",
            "timeout",
            "terminated",
            # 流提前结束（SDK 与各家转运层都有各自的说法）
            "ended without",
            "stream ended before",
            "did not get a response",
            # 显式让调用方重试
            "you can retry your request",
            "try your request again",
            "please retry your request",
            "resourceexhausted",
        ]
    ),
    re.IGNORECASE,
)

INTERRUPTED_STREAM_PATTERN = re.compile(
    "|".join(
        [
            # langchain-openai 的停顿看门狗：连续 N 秒收不到新分片就抛。
            # 网关会静默停顿超过默认的 120s（TCP 还活着，只是不再往下发分片）。
            "streamchunktimeout",
            "no streaming chunk received",
            # 传输层断在半路（各家说法）
            "stream ended before",
            "ended without",
            "did not get a response",
            "incomplete",
            "premature",
            "connection reset",
            "connection lost",
            "other side closed",
            "socket hang up",
            "peer closed",
            "unexpected eof",
            "response payload is not completed",
        ]
    ),
    re.IGNORECASE,
)

RETRYABLE_STATUS = frozenset({408, 409, 429})

_sleep = asyncio.sleep
"""退避用的 sleep。单独拎出来是为了测试能替换掉它——不然验一次退避要真等 2+4+8 秒。"""


class _TokenSpy(BaseCallbackHandler):
    """只回答一个问题：这次尝试有没有已经吐出内容（吐过就不重试，见模块说明）。"""

    seen = False

    def on_llm_new_token(self, *args: object, **kwargs: object) -> None:
        self.seen = True


def _status_of(exc: BaseException) -> int | None:
    """openai SDK 的 APIStatusError 带 status_code；别家兼容层的字段名不完全一致。"""
    for name in ("status_code", "status", "http_status"):
        value = getattr(exc, name, None)
        if isinstance(value, int):
            return value
    return None


def _retry_after_seconds(exc: BaseException) -> float | None:
    """provider 在响应头里明确要求的等待时间（秒）。"""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        millis = headers.get("retry-after-ms")
        if millis:
            return float(millis) / 1000
        after = headers.get("retry-after")
        if after:
            return float(after)
    except (TypeError, ValueError):
        return None
    return None


def stream_was_interrupted(exc: BaseException) -> bool:
    """
    这次失败是不是"流断在半路"——也就是已经吐出的内容是**残句**。

    "吐过字就不重试"防的是把**一句完整的回答**拉两遍（前端撤不回去）。流被掐断不在这个范围里：
    残句本来就要丢掉，而且抛异常时那个 `AIMessage` 根本没构造出来、工具也没执行——
    重试不会重复任何副作用，只会补上那个本来拿不到的结果。
    """
    return bool(INTERRUPTED_STREAM_PATTERN.search(f"{type(exc).__name__}: {exc}"))


def is_retryable(exc: BaseException) -> bool:
    """
    这次失败值不值得重试。

    先看配额/计费（**不可重试优先**——限流常被网关报成 429，但"额度用完"的 429 重试没意义），
    再看是不是"流被掐断"（见 `stream_was_interrupted`），最后看 HTTP 状态，再按错误文本判断。

    流中断要单独判一次、不能只靠 `RETRYABLE_PATTERN`：langchain-openai 那条
    `StreamChunkTimeoutError` 能被认出**只是因为类名里恰好带 "timeout"**，
    消息本身（"No streaming chunk received for 120.0s"）一个可重试词都不含。
    换个类名、同样的措辞就会被判成不可重试、当场失败——这层不能靠巧合。
    """
    text = f"{type(exc).__name__}: {exc}"
    if NON_RETRYABLE_PATTERN.search(text):
        return False
    if stream_was_interrupted(exc):
        return True

    status = _status_of(exc)
    if status is not None:
        return status in RETRYABLE_STATUS or status >= 500

    return bool(RETRYABLE_PATTERN.search(text))


def delay_seconds(exc: BaseException, attempt: int) -> float:
    """
    这次重试前等多久。

    provider 明确给了 `Retry-After` 就听它的；但要是有上限、而它要求的等待更长，
    直接失败说明原因——静默等几分钟比报错更糟（Pi 的同一处理）。
    """
    asked = _retry_after_seconds(exc)
    if asked is not None:
        if MAX_DELAY_SECONDS > 0 and asked > MAX_DELAY_SECONDS:
            raise RuntimeError(
                f"服务端要求的重试等待 {asked:.0f}s 超过上限 {MAX_DELAY_SECONDS:.0f}s，"
                f"不再等待。原始错误：{exc}"
            )
        return asked

    return min(BASE_DELAY_SECONDS * 2 ** (attempt - 1), MAX_DELAY_SECONDS)


def _config_with_spy(
    config: dict | None, spy: _TokenSpy, tags: list[str] | None
) -> dict:
    """
    在**现有**回调后面追加探针，顺手把标签也并进去。

    不能只传探针：`ensure_config` 对 `callbacks` 是**整体替换**而不是合并，只传探针会把
    外层（`astream_events`）的回调整个顶掉——前端从此收不到 `text_delta`（实测：事件流直接
    空了）。所以要把外层回调取出来、追加探针、再一起传下去。

    `tags` 同理：摘要调用靠 `compact` 标签让事件层把它挡在前端之外，合并而不是覆盖。
    """
    merged = dict(config or {})

    existing = merged.get("callbacks")
    if isinstance(existing, list):
        handlers = list(existing)
    elif existing is None:
        handlers = []
    else:
        # CallbackManager / 单个 handler 之类：取出可达的 handlers
        handlers = list(getattr(existing, "handlers", []) or [existing])
    merged["callbacks"] = [*handlers, spy]

    if tags:
        merged["tags"] = [*(merged.get("tags") or []), *tags]
    return merged


async def ainvoke_with_retry(
    model,
    messages: list,
    *,
    config: dict | None = None,
    thread_id: str = "-",
    label: str = "模型调用",
    tags: list[str] | None = None,
    **kwargs: object,
):
    """
    调模型，瞬态失败按指数退避重试；配额/参数类错误立刻抛。

    `config` 传当前的运行配置（节点里就是 `get_config()`）：探针要追加到它的回调上，
    覆盖传会丢流式事件（见 `_config_with_spy`）。

    被取消（用户点停止）**不重试**，原样抛给上层——那是用户的意思，不是失败。
    """
    for attempt in range(1, MAX_RETRIES + 2):
        spy = _TokenSpy()
        try:
            return await model.ainvoke(
                messages, config=_config_with_spy(config, spy, tags), **kwargs
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if attempt > MAX_RETRIES or not is_retryable(exc):
                raise
            if spy.seen and not stream_was_interrupted(exc):
                logger.warning(
                    "%s 失败但已经吐出内容，不再重试（重试会把半截回答再拉一遍）"
                    " thread=%s：%s",
                    label,
                    thread_id,
                    exc,
                )
                raise
            delay = delay_seconds(exc, attempt)  # 要求的等待过长时在这里抛出去
            logger.warning(
                "%s 失败（第 %d/%d 次尝试），%.0fs 后重试 thread=%s：%s",
                label,
                attempt,
                MAX_RETRIES,
                delay,
                thread_id,
                exc,
            )
            await _sleep(delay)
