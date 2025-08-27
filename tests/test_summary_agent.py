import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.summary_agent import summarize_rag


def test_summarize_rag_filters_to_song(monkeypatch):
    calls = {}

    class FakeDoc:
        def __init__(self, text, song):
            self.page_content = text
            self.metadata = {"song_name": song}

    def fake_hybrid_search(query, k=5, song_name=None, **kwargs):
        calls['song_name'] = song_name
        doc_a = FakeDoc("lyrics A", "Imagine")
        doc_b = FakeDoc("lyrics B", "Hey Jude")
        if song_name == "Imagine":
            return [doc_a]
        return [doc_a, doc_b]

    class DummyLLM:
        def __call__(self, prompt):
            return "תשובה על Imagine"

    monkeypatch.setattr("agents.summary_agent.hybrid_search", fake_hybrid_search)
    monkeypatch.setattr("agents.summary_agent.get_llm", lambda: DummyLLM())

    result = summarize_rag('מה המסר בשיר "Imagine"?')

    assert calls['song_name'] == "Imagine"
    assert all(doc.metadata["song_name"] == "Imagine" for doc in result["raw_docs"])
    assert "Imagine" in result["answer"]
