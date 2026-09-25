"""
模型构建：正常模型、摘要模型、以及没配 key 时的占位模型。

摘要模型必须单独建实例（要带输出上限，见 `SummaryModel`），所以这层值得独立出来——
`graph` 与 `LangGraphRunner` 都从这里拿模型，不再各自拼参数。
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI

from app.agent.compaction.policy import summary_max_tokens
from app.config import (
    COMPACT_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_CONTEXT_WINDOW,
    LLM_MODEL,
    LLM_STREAM_CHUNK_TIMEOUT,
)
from app.exceptions import ModelUnavailable

NO_MODEL_REASON = (
    "未配置 LLM_API_KEY，无法对话。"
    "在项目根目录的 .env 里填上 key（可参考 .env.example），重启应用后即可使用。"
)


class UnavailableModel(BaseChatModel):
    """
    没配 `LLM_API_KEY` 时的占位模型：一调用就抛，不假装能回答。

    为什么只换模型、不换整个 runner：历史、待确认项、删除这些**读路径**都存在检查点里，
    跟模型没关系。换个空实现的 runner 会把它们一起弄丢——打开旧会话一片空白。
    """

    @property
    def _llm_type(self) -> str:
        return "unavailable"

    def bind_tools(self, tools: object, **kwargs: object) -> BaseChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object | None = None,
        **kwargs: object,
    ) -> ChatResult:
        raise ModelUnavailable(NO_MODEL_REASON)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object | None = None,
        **kwargs: object,
    ) -> ChatResult:
        raise ModelUnavailable(NO_MODEL_REASON)


class SummaryModel(ChatOpenAI):
    """
    摘要模型：把输出上限用 `max_tokens` 发出去。

    摘要的长度必须由外部卡住（接口参数），不能只写进提示词里——模型可以不理，
    而且那句"别太长"本身也占注意力。

    这版 langchain 会在 `ChatOpenAI._get_request_payload` 里把 `max_tokens` 改名成
    `max_completion_tokens`（跟着 OpenAI 的新命名走）。本项目面向的是**任意 OpenAI 兼容
    端点**，DeepSeek 这类只认 `max_tokens`：改名等于上限没设上，老端点还可能直接 400。
    所以这里把名字改回兼容的那一个（实测见 LESSONS）。
    """

    def _get_request_payload(
        self, input_: object, *, stop: list[str] | None = None, **kwargs: object
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        return payload


def build_model(name: str | None = None, max_tokens: int | None = None) -> BaseChatModel:
    """没配 key 就用占位模型：读路径照常，只有真去调模型时才报错。

    `max_tokens` 只有摘要模型用：卡住摘要长度，别让一次压缩写回比原文还长的东西。
    """
    if not LLM_API_KEY:
        return UnavailableModel()
    model_class = SummaryModel if max_tokens else ChatOpenAI
    return model_class(
        model=name or LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0,
        max_tokens=max_tokens,
        streaming=True,  # 关掉流式就没有 text_delta，前端只能等整段回复
        # SDK 自带的重试关掉：它不记日志、延迟不可控，还会和我们那层叠加次数
        # （3 × 2 次）。重试统一由 app/agent/retry.py 负责，理由见那里。
        max_retries=0,
        # 流式停顿看门狗：SDK 默认 120s 太紧，网关一次静默停顿就会让这一路调用直接失败
        # （见 app/config 里的 LLM_STREAM_CHUNK_TIMEOUT）。
        stream_chunk_timeout=LLM_STREAM_CHUNK_TIMEOUT,
    )


def build_summary_model() -> BaseChatModel:
    """
    摘要模型：`COMPACT_MODEL` 留空就是主模型，只是换个实例、带上输出上限。
    """
    return build_model(COMPACT_MODEL or None, summary_max_tokens(LLM_CONTEXT_WINDOW))
