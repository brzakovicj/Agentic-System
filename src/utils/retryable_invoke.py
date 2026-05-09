import logging
from langchain.chat_models import BaseChatModel
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

RETRYABLE_SIGNALS = frozenset({
    "502", "503", "504",
    "timeout", "Timeout",
    "rate_limit", "RateLimitError",
    "overloaded", "ServiceUnavailable",
    "Connection reset",
})

def _is_retryable(exc: Exception) -> bool:
    exc_str = str(exc)
    exc_type = type(exc).__name__
    return any(sig in exc_str or sig in exc_type for sig in RETRYABLE_SIGNALS)

def _build_retry_decorator():
    return retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10),  # jitter sprečava thundering herd
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )

async def ainvoke_llm(llm: BaseChatModel, messages: list) -> any:
    """Izolovani LLM call — retry logika odvojena od business logike."""
    decorated = _build_retry_decorator()(llm.ainvoke)
    return await decorated(messages)