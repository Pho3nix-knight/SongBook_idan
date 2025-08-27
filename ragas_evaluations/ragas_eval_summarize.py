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
) -> Dict[str, float]:
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
        Reference summary text. When provided, answer relevancy is also scored.

    Returns
    -------
    Dict[str, float]
        Mapping of metric names to their scores.
    """

    try:
        from datasets import Dataset
        from ragas.evaluation import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from langchain_openai import ChatOpenAI
    except Exception as exc:
        raise ImportError(
            "ragas evaluation dependencies are missing or incompatible with this Python version"
        ) from exc

    llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY)

    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }
    metrics = [faithfulness]
    if ground_truth is not None:
        data["ground_truth"] = [ground_truth]
        metrics.append(answer_relevancy)

    dataset = Dataset.from_dict(data)

    results = evaluate(dataset, metrics=metrics, llm=llm)

    scores = {"faithfulness": float(results["faithfulness"][0])}
    if ground_truth is not None and "answer_relevancy" in results:
        scores["answer_relevancy"] = float(results["answer_relevancy"][0])
    return scores

