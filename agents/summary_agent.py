
from typing import List, Any

from helpers.hybride_retreval import hybrid_search
from core.llm import get_llm

DEFAULT_TOP_K = 5

SUMMARY_SYSTEM_PROMPT = """אתה מסייע סיכום קפדן עבור מערכת שאלות־תשובות של ספר שירים.
- הפק סיכום נאמן ומתומצת העונה לבקשת המשתמש.
- השתמש רק בהקשר שסופק; אל תמציא עובדות.
- העדף נקודות מסודרות כאשר זה מועיל.
- אם שמות שירים או מחברים ספציפיים רלוונטיים, ציין אותם במפורש.
- ענה בעברית בלבד.
"""

SUMMARY_USER_PROMPT = """בקשת המשתמש:
{question}

הקשר (הקטעים העליונים {k}):
{context}

כתוב סיכום ממוקד העונה ישירות לבקשת המשתמש. שמור על דיוק והסתמך רק על המידע שניתן.
"""

def build_summary_prompt(question: str, context: str, k: int) -> str:
    return SUMMARY_SYSTEM_PROMPT + "\n\n" + SUMMARY_USER_PROMPT.format(question=question, context=context, k=k)

def _format_docs(docs: List[Any]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        content = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        meta = getattr(d, "metadata", {}) or {}
        title = meta.get("song_name") or meta.get("source") or meta.get("path") or meta.get("file") or ""
        month = meta.get("hebrew_month_name") or meta.get("month_name") or ""
        year = meta.get("year") or ""
        if month or year:
            src = f"{title} ({month} {year})".strip()
        else:
            src = title
        parts.append(f"[{i}] {src}\n{content}")
    return "\n\n".join(parts)

def summarize_rag(question: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """
    Returns a dict: { 'answer': str, 'contexts': List[str], 'raw_docs': List[Document-like] }
    """
    docs = hybrid_search(question, k=top_k)

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
