import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ragas_evaluations.ragas_eval_summarize import load_summary_ground_truth


def test_load_summary_ground_truth():
    text = load_summary_ground_truth("איך לשכוח")
    assert "Question 1" in text
