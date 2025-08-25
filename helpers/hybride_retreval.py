# ---------------------------
# Hybrid Retriever (OpenAI embeddings) — SONG chunks
# ---------------------------
"""
Builds a hybrid retriever (BM25 + OpenAI embeddings via Chroma) over the JSONL produced
by the song-per-chunk songbook_chunker.py.

- One-time global init of BM25 + Chroma (reused across calls)
- Optional filters: month_year, hebrew_month_name, song_name, has_chorus, has_monologue
- Query-aware rerank to prioritize title/body hits for Hebrew/English tokens
- Embedding includes song_name + body (so title-only queries hit well)

Run test:
  python -m helpers.hybride_retreval --rebuild --q "על מה כתבתי על שלג?"
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# ---- project config (constants) ----
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.config import CHROMA_DIR, CHUNKS_PATH, OPENAI_API_KEY  # assumes these point to valid paths

# ---------------------------
# Globals (built once per process)
# ---------------------------
_DOCS_GLOBAL: List[Document] | None = None
_BM25_GLOBAL: BM25Retriever | None = None
_VDB_GLOBAL: Chroma | None = None

_COLLECTION_NAME = "songbook"

_TITLE_TOKENS: set[str] = set()  # ימולא ב-init

def _title_tokens_from_docs(docs: List[Document]) -> set[str]:
    toks = set()
    for d in docs:
        t = (d.metadata.get("song_name") or "").lower()
        toks.update(_tokens(t))
    return toks

def _title_precandidates(query: str, limit: int = 12) -> List[Document]:
    if not _DOCS_GLOBAL:
        return []
    q_toks = _tokens(query)
    phrases = _quoted_phrases(query)

    # 1) חיתוך לפי טוקנים שמופיעים בכותרות בכלל
    q_title_toks = [t for t in q_toks if t in _TITLE_TOKENS]

    scored: list[tuple[int, Document]] = []

    for d in _DOCS_GLOBAL:
        title = (d.metadata.get("song_name") or "").lower()

        # פגיעה בביטוי מצוטט מקבלת ניקוד גבוה
        phrase_hits = sum(1 for p in phrases if p and p in title)

        # פגיעה בטוקנים של כותרות
        token_hits = sum(1 for t in q_title_toks if t in title)

        hits = (10 * phrase_hits) + token_hits  # ביטוי מצוטט ≫ טוקן בודד
        if hits > 0:
            scored.append((hits, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:limit]]

# Minimal stopwords (he+en) for token extraction
_STOP = set("""
the a an and or for to of in on with without into from by is are was were be been being as at this that these those
i you he she it we they me him her us them my your his its our their mine yours hers ours themselves yourself himself
אם אז כי אך אבל כפי לכן כאשר כדי בגלל למרות האם זה זו זאת פה שם שלו שלה שלהם שלנו שלי שלך שלכם שלכן אלה אלו גם כאשר כמו מאוד רק כל בין עד תוך תחת מעל לפני אחרי אולי כבר יותר פחות עדיין שוב אין יש היה היתה היו להיות לעשות לומר לתת לקחת לדעת לחשוב לראות לבוא ללכת
""".split())

# ---------------------------
# Utilities
# ---------------------------
def ensure_api_key():
    key = os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in your environment or in helpers.config.OPENAI_API_KEY."
        )
    return key

def load_jsonl_to_docs(path: str) -> List[Document]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found at: {p}")
    raw = p.read_text(encoding="utf-8")
    docs: List[Document] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        page_content = rec.get("page_content", "")
        meta = rec.get("metadata", {})
        docs.append(Document(page_content=page_content, metadata=meta))
    if not docs:
        raise ValueError(f"No documents loaded from {p} (empty file?).")
    return docs

def _sanitize_metadata(meta: dict) -> dict:
    """Coerce metadata to Chroma-safe primitives."""
    safe: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            safe[k] = v
        elif isinstance(v, dict):
            if k == "date_range" and "start" in v and "end" in v:
                safe["date_start"] = v.get("start")
                safe["date_end"] = v.get("end")
            else:
                safe[k] = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, list):
            if all(isinstance(x, (str, int, float, bool)) or x is None for x in v):
                safe[k] = ", ".join("" if x is None else str(x) for x in v)
            else:
                safe[k] = json.dumps(v, ensure_ascii=False)
        else:
            safe[k] = str(v)
    return safe

def _build_bm25(docs: List[Document]) -> BM25Retriever:
    print(f"🔧 Building BM25 retriever with {len(docs)} documents (songs)...")
    bm25_docs: List[Document] = []
    for d in docs:
        title = (d.metadata.get("song_name") or "").strip()
        text  = d.page_content
        merged = f"{title}\n{text}" if title else text
        bm25_docs.append(Document(page_content=merged, metadata=d.metadata))
    bm25 = BM25Retriever.from_documents(bm25_docs)
    bm25.k = 12  # מעט גבוה יותר לריקול טוב
    print(f"✅ BM25 retriever built with k={bm25.k}")
    return bm25
def _build_chroma(docs: List[Document], persist_directory: Optional[str]) -> Chroma:
    print("🔧 Preparing OpenAI embeddings + sanitizing metadata for Chroma...")
    emb = OpenAIEmbeddings(model="text-embedding-3-large", api_key=ensure_api_key())

    # ⚠️ הטמעת כותרת בתוך התוכן המוטמע — חשוב לשאילתות על שם השיר
    safe_docs: List[Document] = []
    for d in docs:
        md = _sanitize_metadata(d.metadata)
        title = md.get("song_name", "")
        embed_text = f"{title}\n{d.page_content}" if title else d.page_content
        safe_docs.append(Document(page_content=embed_text, metadata=md))

    if persist_directory:
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

    print(f"🔧 Building Chroma vector store in {persist_directory or '(memory)'} ...")
    vdb = Chroma.from_documents(
        documents=safe_docs,
        embedding=emb,
        persist_directory=persist_directory,
        collection_name=_COLLECTION_NAME,
    )
    if persist_directory:
        vdb.persist()
        print("✅ Chroma vector store persisted")
    return vdb

def init_hybrid_if_needed(jsonl_path: Optional[str] = None, persist_directory: Optional[str] = None, force_rebuild: bool = False):
    """
    Build global BM25 + Chroma once. Subsequent calls reuse the globals.
    """
    global _DOCS_GLOBAL, _BM25_GLOBAL, _VDB_GLOBAL, _TITLE_TOKENS

    if not force_rebuild and _BM25_GLOBAL is not None and _VDB_GLOBAL is not None and _DOCS_GLOBAL is not None:
        return  # already inited

    jsonl_path = jsonl_path or CHUNKS_PATH
    persist_directory = persist_directory or CHROMA_DIR

    print(f"📦 Loading docs from {jsonl_path}")
    docs = load_jsonl_to_docs(jsonl_path)

    _DOCS_GLOBAL = docs
    _BM25_GLOBAL = _build_bm25(docs)
    _VDB_GLOBAL = _build_chroma(docs, persist_directory)
    _TITLE_TOKENS = _title_tokens_from_docs(docs)

# ---------------------------
# Search helpers
# ---------------------------
def _build_filter_dict(
    month_year: Optional[str] = None,
    hebrew_month_name: Optional[str] = None,
    song_name: Optional[str] = None,
    has_chorus: Optional[bool] = None,
    has_monologue: Optional[bool] = None,
) -> Dict[str, Any]:
    f: Dict[str, Any] = {}
    if month_year:
        f["month_year"] = month_year
    if hebrew_month_name:
        f["hebrew_month_name"] = hebrew_month_name
    if song_name:
        f["song_name"] = song_name
    if has_chorus is not None:
        f["has_chorus"] = has_chorus
    if has_monologue is not None:
        f["has_monologue"] = has_monologue
    return f

def _apply_post_filter_bm25(docs: List[Document], filter_dict: Dict[str, Any]) -> List[Document]:
    def ok(d: Document) -> bool:
        md = d.metadata
        for k, v in filter_dict.items():
            if md.get(k) != v:
                return False
        return True
    return [d for d in docs if ok(d)] if filter_dict else docs

def _similarity_search_with_fallback(vectordb: Chroma, query: str, k: int, filter_dict: Dict[str, Any]):
    if not filter_dict:
        return vectordb.similarity_search(query, k=k)
    try:
        return vectordb.similarity_search(query, k=k, filter=filter_dict)
    except Exception:
        eq_filter = {k: {"$eq": v} for k, v in filter_dict.items()}
        return vectordb.similarity_search(query, k=k, filter=eq_filter)

# --- Query-aware rerank ---
_HE_RE = r"[\w\u0590-\u05FF׳״\"'’-]+"

# החלף את ה-RE לטוקנים כדי *לא* לכלול גרשיים/מירכאות/פיסוק
_TOKEN_RE = re.compile(r"[A-Za-z\u0590-\u05FF]+")  # אותיות בלבד (עברית/אנגלית)

_HE_PREFIXES = tuple("והבכלמו")  # תחיליות נפוצות בעברית

def _canon_hebrew(tok: str) -> str:
    t = tok
    for _ in range(2):  # הורד עד שתי תחיליות
        if t and t.startswith(_HE_PREFIXES):
            t = t[1:]
        else:
            break
    return t

def _tokens(q: str) -> List[str]:
    # מרים רק אותיות (בלי ' " ״ ׳ וכד'), מסנן סטופ-וורדס, ומוסיף גרסאות מנורמלות
    raw = [t for t in _TOKEN_RE.findall(q.lower()) if len(t) > 1]
    raw = [t for t in raw if t not in _STOP]
    normed = {_canon_hebrew(t) for t in raw}
    toks = list({*raw, *normed})
    return [t for t in toks if t and len(t) > 1]

# זיהוי ביטויים במירכאות (כולל גרש/גרשיים עבריים)
_QUOTE_RX = re.compile(r"[\"'“”‚‘’׳״](.+?)[\"'“”‚‘’׳״]")

def _quoted_phrases(q: str) -> List[str]:
    return [m.strip().lower() for m in _QUOTE_RX.findall(q)]


def _title_hit(doc: Document, toks: List[str]) -> bool:
    title = (doc.metadata.get("song_name") or "").lower()
    return any(t in title for t in toks)

def _score_doc_by_query(doc: Document, toks: List[str], phrases: Optional[List[str]] = None) -> int:
    md = doc.metadata
    title = (md.get("song_name") or "").lower()
    text = doc.page_content.lower()

    score = 0
    if phrases:
        if any(p in title for p in phrases):
            score += 200  # אם הכותרת מכילה ביטוי במירכאות
    if any(t in title for t in toks):
        score += 100
    if any(t in text for t in toks):
        score += 5
    return score


def _dedup(docs: List[Document]) -> List[Document]:
    seen = set()
    out: List[Document] = []
    for d in docs:
        key = d.metadata.get("song_id") or (d.metadata.get("month_year"), d.metadata.get("song_name"), d.page_content[:64])
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out

# ---------------------------
# Public API
# ---------------------------
def build_hybrid(docs: List[Document], persist_directory: Optional[str] = None):
    """
    Build and return BM25 and Chroma instances for the given documents.
    This function is provided for backward compatibility with existing code.
    """
    init_hybrid_if_needed(jsonl_path=None, persist_directory=persist_directory, force_rebuild=True)
    return _BM25_GLOBAL, _VDB_GLOBAL

def get_hybrid_instances():
    """
    Get the current global BM25 and Chroma instances.
    Returns (bm25, vectordb) tuple.
    """
    init_hybrid_if_needed()
    return _BM25_GLOBAL, _VDB_GLOBAL

def hybrid_search(
    query: str,
    k: int = 6,
    month_year: Optional[str] = None,
    hebrew_month_name: Optional[str] = None,
    song_name: Optional[str] = None,
    has_chorus: Optional[bool] = None,
    has_monologue: Optional[bool] = None,
) -> List[Document]:
    """
    Return fused results from BM25 and vector search with optional metadata filters.
    Uses global BM25/Chroma that are built once.
    """
    init_hybrid_if_needed()
    assert _BM25_GLOBAL is not None and _VDB_GLOBAL is not None

    fdict = _build_filter_dict(...)
    q_toks = _tokens(query)
    raw_k = max(k, 30)

    # --- Pre-candidates לפי כותרת ---
    title_cands = _title_precandidates(query, limit=10)

    # --- BM25 ---
    try:
        bm25_docs = _BM25_GLOBAL.invoke(query)
        if fdict:
            bm25_docs = _apply_post_filter_bm25(bm25_docs, fdict)
    except Exception:
        bm25_docs = []

    # --- Vector ---
    try:
        vec_docs = _similarity_search_with_fallback(_VDB_GLOBAL, query, k=raw_k, filter_dict=fdict)
    except Exception:
        vec_docs = []

    # --- Fuse & dedup (כולל כותרות) ---
    candidates = _dedup(title_cands + bm25_docs + vec_docs)

    # --- Title-first policy + rerank (יש לך כבר) ---
    title_hits = [d for d in candidates if _title_hit(d, q_toks)]
    others     = [d for d in candidates if d not in title_hits]
    phrases = _quoted_phrases(query)

    # ... אחרי האיחוד והדה-דופ:
    title_hits_scored = sorted(title_hits, key=lambda d: _score_doc_by_query(d, q_toks, phrases), reverse=True)
    others_scored     = sorted(others,     key=lambda d: _score_doc_by_query(d, q_toks, phrases), reverse=True)
    reranked = title_hits_scored + others_scored
    if not reranked and candidates:
        reranked = candidates
    return reranked[:k]

# ---------------------------
# CLI (for quick testing)
# ---------------------------
#if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default=None, help="Path to chunks.jsonl (optional; default from config)")
    parser.add_argument("--persist", default=None, help="Directory to persist/load Chroma (optional; default from config)")
    parser.add_argument("--q", default="על מה כתבתי על שלג?", help="Test query")
    parser.add_argument("--month", default=None, help="Filter by month_year, e.g., 2022-01")
    parser.add_argument("--hmonth", default=None, help="Filter by hebrew_month_name, e.g., יולי")
    parser.add_argument("--song", default=None, help="Filter by exact song_name")
    parser.add_argument("--chorus", default=None, help="Filter has_chorus true/false")
    parser.add_argument("--mono", default=None, help="Filter has_monologue true/false")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild globals")
    args = parser.parse_args()

    if args.rebuild or _BM25_GLOBAL is None or _VDB_GLOBAL is None:
        # לבנייה מחדש חד-פעמית (מומלץ כשעדכנת את הצ'אנקרים)
        if args.jsonl:  # נרצה שהבנאי ידע על הנתיב החדש
            CHUNKS = args.jsonl
        if args.persist:
            PERSIST = args.persist
        init_hybrid_if_needed(jsonl_path=args.jsonl, persist_directory=args.persist, force_rebuild=True)

    def _to_bool(x: Optional[str]) -> Optional[bool]:
        if x is None: return None
        return x.lower() in ("1", "true", "yes", "y", "כן", "true ")

    hits = hybrid_search(
        args.q,
        k=6,
        month_year=args.month,
        hebrew_month_name=args.hmonth,
        song_name=args.song,
        has_chorus=_to_bool(args.chorus),
        has_monologue=_to_bool(args.mono),
    )

    print(f"\nQuery: {args.q}  |  month_year={args.month}  |  hebrew_month={args.hmonth}  |  song={args.song}")
    for i, d in enumerate(hits, 1):
        md = d.metadata
        anchor = f"{md.get('hebrew_month_name', md.get('month_name',''))}-{md.get('year','')}"
        print(f"[{i}] {anchor} — {md.get('song_name','(song)')}")
        preview = d.page_content[:160].replace("\n", " ")
        print(f"    {preview}\n")
