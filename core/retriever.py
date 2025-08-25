
import os
from typing import Any

def _try_user_retriever():
    try:
        # Common local paths users employ
        from helpers.vector_store import retriever as _user_retriever  # type: ignore
        return _user_retriever
    except Exception:
        pass
    try:
        from vector_store import retriever as _user_retriever  # type: ignore
        return _user_retriever
    except Exception:
        pass
    return None

def _build_fallback_retriever() -> Any:
    """
    Very small fallback that builds an in-memory retriever from ./data (txt/md/pdf supported via simple txt read).
    This is only used if the project doesn't expose a retriever already.
    """
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from pathlib import Path

    data_dir = Path(os.getenv("SONGBOOK_DATA_DIR", "data"))
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for p in data_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in [".txt", ".md"]:
            try:
                docs.extend(splitter.split_documents(TextLoader(str(p), encoding="utf-8").load()))
            except Exception:
                pass

    embeddings = HuggingFaceEmbeddings(model_name=os.getenv("HF_EMBEDDINGS", "sentence-transformers/all-MiniLM-L6-v2"))
    vect = Chroma.from_documents(docs, embedding=embeddings)

    # Return a simple retriever interface
    return vect.as_retriever(search_kwargs={"k": int(os.getenv("TOP_K", "5"))})

def get_retriever() -> Any:
    user = _try_user_retriever()
    if user is not None:
        return user
    return _build_fallback_retriever()
