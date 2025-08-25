# rag_qa.py
from __future__ import annotations
import os
import re
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.hybride_retreval import hybrid_search, load_jsonl_to_docs
from helpers.config import CHUNKS_PATH, CHROMA_DIR, OPENAI_API_KEY

# ---------- LLM ----------
def build_llm():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

# ---------- Prompt ----------
SYSTEM_PROMPT = """אתה עוזר המשיב בעברית.
ענה על סמך הקונטקסט בלבד, אל תמציא עובדות. אם חסר מידע עובדתי לשאלה—אמור זאת.
אם השאלה היא בקשה **יצירתית/טרנספורמטיבית** (למשל: כתיבה, שכתוב, המשך, ניסוח מחדש, תקציר/תמצות),
מותר לך לייצר תוכן חדש בהשראת הסגנון, הטון והדוגמאות מתוך הקונטקסט שסופק.
אם יש חודש/שנה בשאלה—תן עדיפות לקטעים עם מטא-דאטה תואם.
אל תציג בסוף התשובה את המקורות (חודש-שנה, Headline).
הצג את התשובות לפי הסדר הכרונולוגי של השירים, שהוצגו בשאלה ובהקשר הרלוונטי בלבד. מיולי 2021 עד מרץ 2022.
"""

USER_PROMPT = """שאלה/בקשה:
{question}

הקשר רלוונטי:
{context}

הנחיות:
- אם מדובר בשאלה עובדתית—ענה בקצרה ובדיוק.
- אם מדובר במשימת כתיבה/שכתוב/המשך—צור טיוטה קצרה בסגנון העולה מהקונטקסט.
- אם חסר מידע מהותי—ציין זאת, אך נסה לתת כיוון/הבהרה על סמך הקיים.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", USER_PROMPT),
])

# ---------- Helpers ----------
_Q_GEN = re.compile(r"(כתוב|תכתוב|המשך|תמשיך|שכתב|שכתבי|ניסוח|נַסֵּח|תקצר|תמצת|סכם|סכמי|summary|rewrite|paraphrase|continue)", re.IGNORECASE)

def _tokenize(q: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z\u0590-\u05FF]+", q.lower()) if len(t) > 1]


def _find_hit_span(text: str, tokens: list[str]) -> Optional[tuple[int, int]]:
    low = text.lower()
    for tok in tokens:
        i = low.find(tok)
        if i != -1:
            return (max(0, i - 350), min(len(text), i + 350))
    return None

def _make_snippet(query: str, content: str, fallback_chars: int = 800) -> str:
    toks = _tokenize(query)
    span = _find_hit_span(content, toks)
    snippet = content[span[0]:span[1]] if span else content[:fallback_chars]
    return " ".join(snippet.strip().split())

def _select_docs_for_context(docs: List[Document], query: str, max_docs: int = 6) -> List[Document]:
    q_tokens = _tokenize(query)

    def _score_doc(doc: Document) -> int:
        md = doc.metadata
        title = (md.get("song_name") or "").lower()
        text  = doc.page_content.lower()
        score = 0
        if any(t in title for t in q_tokens):  # פגיעה בכותרת = חשוב
            score += 3
        if any(t in text for t in q_tokens):   # פגיעה בגוף הטקסט
            score += 1
        # אם תרצה בונוסים קלים:
        # if md.get("has_chorus"): score += 1
        return score

    # כל המסמכים הם שירים (type="song")
    songs = [d for d in docs if d.metadata.get("type") == "song"]
    songs.sort(key=_score_doc, reverse=True)
    chosen = songs[:max_docs]

    # fallback אם לא נמצאו התאמות:
    if not chosen and docs:
        chosen = docs[:max_docs]
    return chosen


def _format_context(query: str, docs: List[Document], max_chars: int = 3500) -> str:
    parts, used = [], 0
    for i, d in enumerate(docs, 1):
        mm = d.metadata
        anchor = f"{mm.get('hebrew_month_name', mm.get('month_name',''))}-{mm.get('year','')}"
        title  = mm.get("song_name", "(song)")
        snippet = _make_snippet(query, d.page_content, 800)
        block = f"[{i}] ({anchor}) {title} :: {snippet}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


# טען פעם אחת
docs = load_jsonl_to_docs(CHUNKS_PATH)
# Note: build_hybrid is called when needed by hybrid_search

# ---------- Main ask ----------
def ask(question: str, k: int = 10, month_year: Optional[str] = None) -> str:
    # ניחוש חודש-שנה (אופציונלי)
    month_year = month_year or _infer_month_year_from_question(question)

    hits = hybrid_search(question, k=k, month_year=month_year)
    if not hits:
        return "אין לי מספיק מידע לענות על זה מתוך החומר שיש לי."

    # בחירה וריראנק לקונטקסט
    chosen = _select_docs_for_context(hits, question, max_docs=6)
    context = _format_context(question, chosen)

    llm = build_llm()
    msgs = prompt.format_messages(question=question, context=context)
    resp = llm.invoke(msgs)
    answer = resp.content.strip()

    # מקורות (markdown)
    src_lines = []
    for i, d in enumerate(chosen, 1):
        mm = d.metadata
        title = mm.get("song_name", "(song)")
        anchor = f"{mm.get('hebrew_month_name', mm.get('month_name',''))}-{mm.get('year','')}"
        src_lines.append(f"- [{i}] {anchor} — {title}")
    sources_md = "\n".join(src_lines)

    # return answer
    return f"{answer}" #\n\n מקורות: \n\n {sources_md}"

def _infer_month_year_from_question(q: str) -> Optional[str]:
    he = {"ינואר":1,"פברואר":2,"מרץ":3,"אפריל":4,"מאי":5,"יוני":6,"יולי":7,"אוגוסט":8,"ספטמבר":9,"אוקטובר":10,"נובמבר":11,"דצמבר":12}
    en = {"january":1,"february":2,"march":3,"april":4,"may":5,"june":6,"july":7,"august":8,"september":9,"october":10,"november":11,"december":12}
    ql = q.lower()
    m = re.search(r"(ינואר|פברואר|מרץ|אפריל|מאי|יוני|יולי|אוגוסט|ספטמבר|אוקטובר|נובמבר|דצמבר)\s+(\d{4})", ql)
    if m:
        mm = he[m.group(1)]
        yyyy = int(m.group(2))
        return f"{yyyy:04d}-{mm:02d}"
    m = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})", ql)
    if m:
        mm = en[m.group(1)]
        yyyy = int(m.group(2))
        return f"{yyyy:04d}-{mm:02d}"
    m = re.search(r"(\d{4})[-/](\d{1,2})", ql)
    if m:
        yyyy, mm = int(m.group(1)), int(m.group(2))
        return f"{yyyy:04d}-{mm:02d}"
    return None
