"""
RAG Songbook Chunker — Song-per-Chunk
--------------------------------------
Builds song-level chunks (one per [Headline]) from a single text file.
Each chunk spans from its [Headline] up to the next [Headline] or the next month header.

Outputs JSONL of LangChain-like records: {"page_content", "metadata", "chunk_id", "type"}.

Usage:
  python songbook_chunker.py --input /path/to/writing_mixed.txt --out /path/to/chunks.jsonl
"""
from __future__ import annotations
import re
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Iterable
import bisect

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.config import FILE_PATH, CHUNKS_PATH

# LangChain document container (0.2+)
try:
    from langchain_core.documents import Document  # noqa
except Exception:
    from langchain.schema import Document  # type: ignore

# ---------------------------
# Config
# ---------------------------
LINES_PER_PAGE = 60  # adjust per your editor/pdf export

# ---------------------------
# Regex patterns
# ---------------------------
MONTH_HEADER = re.compile(r"^\[(\d{1,2})\s*\(([^)]+)\)\s*(\d{4})\]\s*$", re.MULTILINE)

# Headline at line start, title may be on same line after the tag.
# We'll later strip any trailing inline tags (e.g., [Note]) from the captured title.
HEADLINE_LINE = re.compile(r'^\[Headline\]\s*(.*)$', re.MULTILINE)

# Count any tag anywhere
TAG_ANYWHERE = re.compile(r'\[(Headline|Monologue|Verse|Chorus|Note|Idea|Bridge|C-Part|Scrap|Intro)\]')

# Minimal stopwords for quick keywords
STOPWORDS = set(
    """
    the a an and or for to of in on with without into from by is are was were be been being as at this that these those
    I you he she it we they me him her us them my your his its our their mine yours hers ours theirs myself yourself himself
    herself itself ourselves yourselves themselves
    אם אז כי אך אבל כפי לכן כאשר כדי בגלל למרות האם האם זה זו זאת פה שם שלו שלה שלהם שלנו שלי שלך שלכם שלכן אלה אלו גם גם כן כאשר כמו מאוד רק כל בין עד תוך תחת מעל לפני אחרי אולי כבר יותר פחות עדיין שוב שובו אין יש היה היתה היו להיות לעשות לומר לתת לקחת לדעת לחשוב לראות לבוא ללכת
    """.split()
)

# ---------------------------
# Data classes
# ---------------------------
@dataclass
class ChunkRecord:
    chunk_id: str
    type: str  # "song"
    text: str
    metadata: Dict[str, Any]

    def to_jsonl(self) -> str:
        payload = {
            "page_content": self.text,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id,
            "type": self.type,
        }
        return json.dumps(payload, ensure_ascii=False)

# ---------------------------
# Helpers
# ---------------------------
def _month_spans(text: str) -> List[Tuple[int, int, str, int, int]]:
    """Return (year, month_num, month_name, start_idx, end_idx) spans for each month."""
    matches = list(MONTH_HEADER.finditer(text))
    spans = []
    for i, m in enumerate(matches):
        month_num = int(m.group(1))
        month_name = m.group(2).strip()
        year = int(m.group(3))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        spans.append((year, month_num, month_name, start, end))
    return spans

def _date_range(year: int, month: int) -> Dict[str, str]:
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    return {
        "start": f"{year:04d}-{month:02d}-01",
        "end": f"{year:04d}-{month:02d}-{last_day:02d}",
    }

def _normalize_lines(block: str) -> str:
    lines = [ln.rstrip() for ln in block.splitlines()]
    out, empty = [], 0
    for ln in lines:
        if ln.strip() == "":
            empty += 1
            if empty <= 1:
                out.append("")
        else:
            empty = 0
            out.append(ln)
    return "\n".join(out).strip()

def _count_blocks(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for m in TAG_ANYWHERE.finditer(text):
        tag = m.group(1)
        counts[tag] = counts.get(tag, 0) + 1
    return counts

def _simple_keywords(text: str, top_k: int = 12) -> List[str]:
    from collections import Counter
    tokens = re.findall(r"[\w\u0590-\u05FF׳״\"'’-]+", text.lower())
    terms = [t for t in tokens if t.isalpha() and t not in STOPWORDS and len(t) > 1]
    freq = Counter(terms)
    return [w for w, _ in freq.most_common(top_k)]

def _first_nonempty_lines(text: str, n: int = 3) -> List[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[:n]

def _line_start_positions(full_text: str) -> List[int]:
    """Return array of absolute char indices where each line starts."""
    starts = [0]
    for i, ch in enumerate(full_text):
        if ch == "\n":
            starts.append(i + 1)
    return starts

def _char_to_line_page(char_idx: int, line_starts: List[int]) -> Tuple[int, int]:
    """Map absolute char index to (line_no starting at 1, page_no starting at 1)."""
    # find rightmost line start <= char_idx
    line_no = bisect.bisect_right(line_starts, char_idx)  # 1-based already
    page_no = ((line_no - 1) // LINES_PER_PAGE) + 1
    return line_no, page_no

def _make_chunk_summary(song_name: str, body_text: str, keywords: List[str], section_count: Dict[str, int]) -> str:
    head = song_name
    k = ", ".join(keywords[:5])
    sc = ", ".join(f"{k2}:{v}" for k2, v in section_count.items() if k2 != "Headline") or "—"
    first = " | ".join(_first_nonempty_lines(body_text, 2))
    return f"{head} — keywords: {k} | sections: {sc} | {first}"

# ---------------------------
# Core parsing: SONGS ONLY
# ---------------------------
def parse_songbook(text: str, source_file: str = FILE_PATH) -> List[ChunkRecord]:
    records: List[ChunkRecord] = []

    # For page/line anchors we need positions in the ORIGINAL full text (not normalized)
    line_starts = _line_start_positions(text)

    for (year, mnum, mname, s, e) in _month_spans(text):
        month_raw = text[s:e]           # raw month slice (keep indices accurate)
        month_norm = _normalize_lines(month_raw)  # not used for indices; sometimes helpful

        # find all headlines in RAW month text
        headline_matches = list(HEADLINE_LINE.finditer(month_raw))
        if not headline_matches:
            continue

        # helper: next headline boundary
        def nxt(start_after: int) -> int:
            for m in headline_matches:
                if m.start() > start_after:
                    return m.start()
            return len(month_raw)

        # enumerate songs in this month
        for idx_in_month, h in enumerate(headline_matches, start=1):
            # title may appear on same line; strip trailing inline tags if any
            same_line = (h.group(1) or "").strip()
            if same_line:
                # cut at first inline tag if appears (e.g., "[Note]")
                same_line = re.split(r"\s*\[", same_line, 1)[0].strip()
            # body boundaries in RAW
            local_start = h.end()
            local_end = nxt(h.end())
            body_raw = month_raw[local_start:local_end].strip()
            # normalize for final page_content
            body = _normalize_lines(body_raw)

            # derive title if missing
            if same_line:
                title = same_line
            else:
                title = _first_nonempty_lines(body, 1)[0] if _first_nonempty_lines(body, 1) else "(untitled)"

            # absolute anchors
            abs_start = s + local_start
            abs_end   = s + local_end
            line_start, page_start = _char_to_line_page(abs_start, line_starts)
            line_end,   page_end   = _char_to_line_page(abs_end,   line_starts)

            # counts/keywords
            section_count = _count_blocks(body_raw)  # count on raw (tags included)
            keywords = _simple_keywords(body, top_k=16)

            # meta
            month_year = f"{year:04d}-{mnum:02d}"
            song_id = f"song-{month_year}-" + re.sub(r"[^\w\-]+", "-", title)[:60].strip("-")

            # booleans for quick filtering
            has_chorus = section_count.get("Chorus", 0) > 0
            has_monologue = section_count.get("Monologue", 0) > 0

            meta = {
                "type": "song",
                "song_id": song_id,
                "song_name": title,
                "song_date": f"{year:04d}-{mnum:02d}-01",     # month anchor
                "month_year": month_year,
                "month_name": mname,
                "year": year,
                "month_num": mnum,

                "section_count": section_count,               # dict of tag counts
                "keywords": keywords,                         # list[str]
                "chunk_summary": _make_chunk_summary(title, body, keywords, section_count),

                "word_count": len(re.findall(r"\w+", body, flags=re.UNICODE)),
                "char_count": len(body),
                "has_chorus": has_chorus,
                "has_monologue": has_monologue,
                "song_index_in_month": idx_in_month,

                # anchors & pagination
                "anchors": {"file": source_file},
                "char_start": abs_start,
                "char_end": abs_end,
                "line_start": line_start,
                "line_end": line_end,
                "page_start": page_start,
                "page_end": page_end,
                "page_number": page_start,                    # convenience field

                # month date range (could be flattened later by sanitizer)
                "date_range": _date_range(year, mnum),
            }

            records.append(ChunkRecord(
                chunk_id=song_id,
                type="song",
                text=body,
                metadata=meta
            ))

    return records

# ---------------------------
# JSONL writing
# ---------------------------
def write_jsonl(records: Iterable[ChunkRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.to_jsonl() + "\n")

# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=False, help="Path to input text file")
    parser.add_argument("--out", type=str, required=False, help="Path to output JSONL")
    args = parser.parse_args()

    if args.input is None:
        args.input = FILE_PATH
    if args.out is None:
        args.out = CHUNKS_PATH

    raw = Path(args.input).read_text(encoding="utf-8")
    records = parse_songbook(raw, source_file=Path(args.input).name)
    write_jsonl(records, Path(args.out))

    # Report
    songs = [r for r in records if r.type == "song"]
    print(f"Wrote {len(records)} song-chunks → {args.out}")
    print(f"  Songs: {len(songs)}")

if __name__ == "__main__":
    main()
