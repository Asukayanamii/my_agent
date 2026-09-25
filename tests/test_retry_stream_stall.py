"""
回归测试：模型流被中途掐断后，重试层与停顿看门狗的行为。

背景（网关实测）：

    StreamChunkTimeoutError: No streaming chunk received for 120.0s
    (model=deepseek-*, chunks_received=5164)

网关吐了几千个分片之后不再往下发，但 TCP 连接还活着。**三个缺陷**把它放大成一次彻底失败：

1. `retry.py` 的"吐过字就不重试"把**被掐断的流**也挡住了。该规则防的是把一句**完整**回答
   拉两遍；残句本来就要丢。
2. `retry.py` 的 `is_retryable` 能放行这条错误，**只是因为类名里恰好有 "timeout"** ——
   消息本身一个 `RETRYABLE_PATTERN` 认识的词都没有。换个类名就会当场失败。
3. `models.py` 没设停顿看门狗，SDK 默认的 120s 对这个网关太紧。

不依赖 pytest、不联网，直接跑：

    python tests/test_retry_stream_stall.py

退出码 0 = 全部通过。
"""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(("PASS  " if ok else "FAIL  ") + label + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(label)


class StallError(Exception):
    """生产里的措辞，但换了个**中性**的类名。

    这正是要盯住的那个用例：消息里没有一个 `RETRYABLE_PATTERN` 认识的词，
    所以判对只能靠"流被掐断"这条规则，而不是靠类名里恰好带 "timeout"。
    """

    def __init__(self) -> None:
        super().__init__(
            "No streaming chunk received for 120.0s (model=deepseek-chat, "
            "chunks_received=5164). The connection may be alive at the TCP layer but "
            "is not producing content."
        )


class StreamChunkTimeoutError(Exception):
    """langchain-openai 的真实类名，里面带 "timeout"。"""

    def __init__(self) -> None:
        super().__init__("No streaming chunk received for 120.0s")


class GatewayError(Exception):
    """非流式的失败、但发生在吐过字之后：**仍然**不许重试。"""

    def __init__(self) -> None:
        super().__init__("400 - invalid request: messages[3].role is not allowed")


def fake_model(raise_with, *, succeed_on: int = 2, emit_tokens: bool = True):
    """按次数抛错的假模型；`emit_tokens` 模拟"已经吐出内容"。"""

    class _Model:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, messages, config=None, **kwargs):  # noqa: ANN001
            self.calls += 1
            if self.calls < succeed_on:
                if emit_tokens:
                    for handler in (config or {}).get("callbacks") or []:
                        handler.on_llm_new_token("partial")
                raise raise_with()
            return f"ok-after-{self.calls}-calls"

    return _Model()


def check_classification() -> None:
    from app.agent import retry as retry_module

    print("=== 1. 「流被掐断」的判定 ===")
    check("停顿看门狗抛的错被认出来", retry_module.stream_was_interrupted(StallError()))
    check("  按类名也认得出", retry_module.stream_was_interrupted(StreamChunkTimeoutError()))
    check("传输中断也算", retry_module.stream_was_interrupted(Exception("Connection reset by peer")))
    check("普通接口错误不算", not retry_module.stream_was_interrupted(GatewayError()))

    print()
    print("--- 可重试性（真正承重的那一半） ---")
    check(
        "掐断的流可重试：只凭消息（中性类名、消息里没有可重试词）",
        retry_module.is_retryable(StallError()),
    )
    check("  按真实类名也可重试", retry_module.is_retryable(StreamChunkTimeoutError()))
    check("普通接口错误仍然不可重试", not retry_module.is_retryable(GatewayError()))


def check_retry_behaviour() -> None:
    from app.agent import retry as retry_module

    print()
    print("=== 2. 掐断的流会被重试（原来这里直接失败） ===")
    original_sleep = retry_module._sleep
    slept: list[float] = []

    async def no_wait(seconds: float) -> None:
        slept.append(seconds)

    retry_module._sleep = no_wait
    try:
        model = fake_model(StallError, succeed_on=2)
        result = asyncio.run(retry_module.ainvoke_with_retry(model, [], label="模型调用", thread_id="t"))
        check("重试后成功拿到结果", result == "ok-after-2-calls", f"result={result!r} calls={model.calls}")
        check("只花掉一次重试", model.calls == 2, f"calls={model.calls}")
        check("两次尝试之间有退避", slept == [2.0], f"slept={slept}")

        print()
        print("=== 3. 老规矩对「非流式」失败仍然生效 ===")
        slept.clear()
        model = fake_model(GatewayError, succeed_on=5)
        raised = False
        try:
            asyncio.run(retry_module.ainvoke_with_retry(model, [], label="模型调用", thread_id="t"))
        except GatewayError:
            raised = True
        check("吐过字之后遇到普通错误，仍然不重试", raised)
        check("  只调用了一次", model.calls == 1, f"calls={model.calls}")

        print()
        print("=== 4. 一直不恢复的流最终要放弃（不能无限重试） ===")
        slept.clear()
        model = fake_model(StallError, succeed_on=99)
        raised = False
        try:
            asyncio.run(retry_module.ainvoke_with_retry(model, [], label="模型调用", thread_id="t"))
        except StallError:
            raised = True
        check("永久掐断的流最终抛出去", raised)
        check("  用满重试预算", model.calls == retry_module.MAX_RETRIES + 1, f"calls={model.calls}")
    finally:
        retry_module._sleep = original_sleep


def check_watchdog_config() -> None:
    print()
    print("=== 5. 停顿看门狗不再是 SDK 默认的 120s ===")

    # 必须在 reload 之前设好：config 是在 import 时读环境的。
    os.environ["LLM_API_KEY"] = "k"
    os.environ["LLM_BASE_URL"] = "http://127.0.0.1:1/v1"

    def load(value: str | None):
        from app import config as config_module
        from app.agent import models as models_module

        if value is None:
            os.environ.pop("LLM_STREAM_CHUNK_TIMEOUT", None)
        else:
            os.environ["LLM_STREAM_CHUNK_TIMEOUT"] = value
        importlib.reload(config_module)
        importlib.reload(models_module)
        return config_module, models_module

    original_key = os.environ.get("LLM_API_KEY")
    original_base = os.environ.get("LLM_BASE_URL")
    try:
        config_module, models_module = load(None)
        check(
            "默认值比 SDK 的 120s 宽",
            config_module.LLM_STREAM_CHUNK_TIMEOUT == 300,
            str(config_module.LLM_STREAM_CHUNK_TIMEOUT),
        )

        model = models_module.build_model()
        check(
            "这个值确实传到了模型客户端",
            getattr(model, "stream_chunk_timeout", None) == 300.0,
            f"{type(model).__name__}.stream_chunk_timeout="
            f"{getattr(model, 'stream_chunk_timeout', None)}",
        )

        config_module, models_module = load("0")
        check(
            "填 0 关掉看门狗",
            config_module.LLM_STREAM_CHUNK_TIMEOUT is None,
            str(config_module.LLM_STREAM_CHUNK_TIMEOUT),
        )
        check(
            "  关掉时传给客户端的是 None",
            getattr(models_module.build_model(), "stream_chunk_timeout", "missing") is None,
        )

        config_module, _ = load("garbage")
        check(
            "填错退回默认值并记账",
            config_module.LLM_STREAM_CHUNK_TIMEOUT == 300
            and any("LLM_STREAM_CHUNK_TIMEOUT" in w or "garbage" in w for w in config_module.CONFIG_WARNINGS),
            str(config_module.LLM_STREAM_CHUNK_TIMEOUT),
        )
    finally:
        os.environ.pop("LLM_STREAM_CHUNK_TIMEOUT", None)
        if original_key is None:
            os.environ.pop("LLM_API_KEY", None)
        else:
            os.environ["LLM_API_KEY"] = original_key
        if original_base is None:
            os.environ.pop("LLM_BASE_URL", None)
        else:
            os.environ["LLM_BASE_URL"] = original_base
        load(None)


def main() -> int:
    check_classification()
    check_retry_behaviour()
    check_watchdog_config()

    print()
    print("result: " + ("ALL OK" if not failures else f"{len(failures)} FAILED -> {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
