
import os

def get_llm():
    """
    Returns a LangChain-compatible LLM with a permissive .invoke(prompt) interface.
    Prefers OpenAI via langchain_openai if OPENAI_API_KEY is set; otherwise falls back to a local textgen.
    """
    try:
        from langchain_openai import ChatOpenAI
        key = os.getenv("OPENAI_API_KEY")
        if key:
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            return ChatOpenAI(model=model, temperature=float(os.getenv("TEMPERATURE", "0.2")))
    except Exception:
        pass

    # Fallback: simple textgen using HuggingFace pipeline if available
    try:
        from transformers import pipeline
        pipe = pipeline("text-generation", model=os.getenv("HF_TEXTGEN", "gpt2"))
        class HFWrap:
            def invoke(self, prompt: str):
                out = pipe(prompt, max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "512")))
                return out[0]["generated_text"]
            __call__ = invoke
        return HFWrap()
    except Exception:
        # Last resort: echo
        class EchoLLM:
            def invoke(self, prompt: str):
                return "LLM not configured. Please set OPENAI_API_KEY or install transformers for local textgen."
            __call__ = invoke
        return EchoLLM()
