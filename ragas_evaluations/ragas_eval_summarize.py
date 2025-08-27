"""Utility to evaluate summaries using RAGAS metrics."""

from typing import Dict, List, Optional
from difflib import get_close_matches
import os

from helpers.config import OPENAI_API_KEY

# Path to summary ground-truth files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_GT_DIR = os.path.join(BASE_DIR, "ground_truth", "summary_gt", "2021", "july")


def load_summary_ground_truth(song_name: str) -> str:
    """Load ground-truth summary text for a given song.

    Parameters
    ----------
    song_name: str
        The name of the song to search for.

    Returns
    -------
    str
        Contents of the matching ground-truth file.
    """

    files = os.listdir(SUMMARY_GT_DIR)
    # Try direct substring match first for robustness with Hebrew names
    direct = [f for f in files if song_name in f]
    if direct:
        filename = direct[0]
    else:
        match = get_close_matches(song_name, files, n=1, cutoff=0.5)
        if not match:
            raise FileNotFoundError(f"No ground truth summary found for '{song_name}'")
        filename = match[0]
    path = os.path.join(SUMMARY_GT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def evaluate_summary(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """Evaluate a generated summary against its source context.

    Parameters
    ----------
    question: str
        The original user prompt.
    answer: str
        The summary generated for the prompt.
    contexts: List[str]
        Context passages used to create the summary.
    ground_truth: Optional[str]
        Reference summary text. When provided, answer correctness is also scored.

    Returns
    -------
    Optional[Dict[str, float]]
        Mapping of metric names to their scores, or ``None`` if the evaluation
        dependencies are unavailable.
    """

    try:
        from datasets import Dataset
        from ragas.evaluation import evaluate
        from ragas.metrics import faithfulness, answer_correctness
        from langchain_openai import ChatOpenAI
    except Exception as exc:
        # Optional evaluation: gracefully skip when ragas or its dependencies
        # are not installed (e.g. on Python < 3.10 without ``eval_type_backport``)
        print(
            "RAGAS evaluation skipped: dependencies are missing or incompatible with this Python version",
            exc,
        )
        return None

    llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY)

    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }
    metrics = [faithfulness]
    if ground_truth is not None:
        data["ground_truth"] = [ground_truth]
        metrics.append(answer_correctness)

    dataset = Dataset.from_dict(data)

    results = evaluate(dataset, metrics=metrics, llm=llm)

    scores = {"faithfulness": float(results["faithfulness"][0])}
    if ground_truth is not None and "answer_correctness" in results:
        scores["answer_correctness"] = float(results["answer_correctness"][0])
    return scores

