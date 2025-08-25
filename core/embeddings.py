
import os

def get_embeddings():
    """
    Return a LangChain-compatible embeddings object.
    Prefers OpenAIEmbeddings if OPENAI_API_KEY set; else falls back to  MiniLM.
    """
    try:
        from langchain_openai import OpenAIEmbeddings
        if os.getenv("OPENAI_API_KEY"):
            return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"))
    except Exception:
        pass
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=os.getenv("HF_EMBEDDINGS", "sentence-transformers/all-MiniLM-L6-v2"))
