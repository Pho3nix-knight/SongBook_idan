"""Utility to evaluate summaries using RAGAS metrics."""

from __future__ import annotations

from typing import List, Dict

from datasets import Dataset
from ragas.evaluation import evaluate
from ragas.metrics import faithfulness
from langchain_openai import ChatOpenAI

from helpers.config import OPENAI_API_KEY


def evaluate_summary(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate a generated summary against its source context.

    Parameters
    ----------
    question: str
        The original user prompt.
    answer: str
        The summary generated for the prompt.
    contexts: List[str]
        Context passages used to create the summary.

    Returns
    -------
    Dict[str, float]
        Mapping of metric names to their scores.
    """

    llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY)

    dataset = Dataset.from_dict(
        {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
    )

    results = evaluate(dataset, metrics=[faithfulness], llm=llm)

    return {"faithfulness": float(results["faithfulness"][0])}

