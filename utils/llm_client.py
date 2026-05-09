"""
LLM client utility — wraps Groq API via langchain-groq for LangGraph integration.
Uses llama-3.3-70b-versatile model with retry logic for rate limits.
"""

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
    """Invoke a LangChain runnable with explicit retry on rate limits and parse errors."""
    for attempt in range(max_retries):
        try:
            return runnable.invoke(prompt)
        except Exception as e:
            err = str(e).lower()
            is_retryable = (
                "rate_limit" in err or "429" in err or "too many" in err
                or "output_parser" in err or "validation" in err
                or "json" in err or "parse" in err
                or "server" in err or "500" in err or "503" in err
            )
            if is_retryable and attempt < max_retries - 1:
                delay = wait_seconds * (2 ** attempt)  # exponential backoff
                print(f"[Retry] Error: {type(e).__name__}, waiting {delay}s (attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
                continue
            raise
