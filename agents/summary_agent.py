
from typing import List, Optional, Any

from .utils import chunk_text  # local util (inlined below if missing)
try:
    # Try to import user's existing retriever if present
    from helpers.vector_store import retriever as _user_retriever
except Exception:
    _user_retriever = None

from ..core.retriever import get_retriever
from ..core.llm import get_llm
from ..core.embeddings import get_embeddings

DEFAULT_TOP_K = 5

SUMMARY_SYSTEM_PROMPT = """You are a careful summarization assistant for a Songbook QA system.
- Produce a faithful, concise summary that answers the user's request.
- Use only the provided context; do not invent facts.
- Prefer structured bullet points when helpful.
- If specific song titles/authors are relevant, mention them explicitly.
"""

SUMMARY_USER_PROMPT = """User request:
{question}

Context (top {k} chunks):
{context}

Write a focused summary that directly addresses the user's request. Keep it accurate and self-contained.
"""

def build_summary_prompt(question: str, context: str, k: int) -> str:
    return SUMMARY_SYSTEM_PROMPT + "\n\n" + SUMMARY_USER_PROMPT.format(question=question, context=context, k=k)

def _format_docs(docs: List[Any]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        content = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source") or meta.get("path") or meta.get("file") or ""
        parts.append(f"[{i}] {src}\n{content}")
    return "\n\n".join(parts)

def summarize_rag(question: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """
    Returns a dict: { 'answer': str, 'contexts': List[str], 'raw_docs': List[Document-like] }
    """
    retriever = _user_retriever or get_retriever()
    docs = retriever.get_relevant_documents(question) if hasattr(retriever, "get_relevant_documents") else retriever.retrieve(question)
    docs = docs[:top_k] if len(docs) > top_k else docs

    context_str = _format_docs(docs)
    llm = get_llm()
    prompt = build_summary_prompt(question, context_str, len(docs))
    answer = llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)  # support LC v0.2 and vanilla call

    # normalize answer string
    if isinstance(answer, dict) and "content" in answer:
        answer_text = answer["content"]
    elif hasattr(answer, "content"):
        answer_text = answer.content
    elif isinstance(answer, str):
        answer_text = answer
    else:
        answer_text = str(answer)

    contexts = []
    for d in docs:
        contexts.append(getattr(d, "page_content", None) or getattr(d, "content", None) or str(d))

    return {
        "answer": answer_text.strip(),
        "contexts": contexts,
        "raw_docs": docs,
    }
