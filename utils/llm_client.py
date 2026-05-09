"""
LLM client utility — wraps Groq API via langchain-groq for LangGraph integration.
Uses llama-3.3-70b-versatile model with retry logic for rate limits.
"""

import asyncio
import os
import time

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

_llm: ChatGroq | None = None


def get_llm() -> ChatGroq:
    """Get a configured ChatGroq LLM instance (singleton) with built-in retries."""
    global _llm
    if _llm is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=api_key,
            max_retries=3,
        )
    return _llm


def invoke_with_retry(runnable, prompt, max_retries=3, wait_seconds=2):
    """Invoke a LangChain runnable with explicit retry on rate limits."""
    for attempt in range(max_retries):
        try:
            return runnable.invoke(prompt)
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = "rate_limit" in err or "429" in err or "too many" in err
            if is_rate_limit and attempt < max_retries - 1:
                delay = wait_seconds * (attempt + 1)
                print(f"[Retry] Rate limit hit, waiting {delay}s (attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
                continue
            raise


async def ainvoke_with_retry(runnable, prompt, max_retries=3, wait_seconds=2):
    """Async invoke with retry on rate limits."""
    for attempt in range(max_retries):
        try:
            return await runnable.ainvoke(prompt)
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = "rate_limit" in err or "429" in err or "too many" in err
            if is_rate_limit and attempt < max_retries - 1:
                delay = wait_seconds * (attempt + 1)
                print(f"[Retry] Rate limit hit, waiting {delay}s (attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(delay)
                continue
            raise


def run_async_safe(coro):
    """Run async coroutine from sync code using run_coroutine_threadsafe.
    Creates a dedicated event loop thread to avoid conflicts with Streamlit's loop.
    """
    import threading

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
