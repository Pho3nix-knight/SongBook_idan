
from typing import List, Dict, Optional

def evaluate_with_ragas(question: str, answer: str, contexts: List[str], ground_truth: Optional[str] = None) -> Dict:
    """
    Runs a small RAGAS evaluation on a single (q, a, contexts[, ground_truth]) sample.
    Returns a dict of metrics.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.integrations.langchain import LangchainLLM, LangchainEmbeddings
    except Exception as e:
        return {"error": f"ragas not installed or import failed: {e}"}

    # Build dataset format expected by ragas
    dataset = [{
        "question": question,
        "answer": answer,
        "contexts": contexts,
        # ground_truth is optional but improves answer_relevancy
        **({"ground_truth": ground_truth} if ground_truth else {}),
    }]

    # Create adapters
    from ..core.llm import get_llm
    from ..core.embeddings import get_embeddings
    llm = LangchainLLM(get_llm())
    emb = LangchainEmbeddings(get_embeddings())

    res = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall], llm=llm, embeddings=emb)
    # Convert to simple python dict
    try:
        df = res.to_pandas()
        return {
            "faithfulness": float(df["faithfulness"][0]),
            "answer_relevancy": float(df["answer_relevancy"][0]),
            "context_precision": float(df["context_precision"][0]),
            "context_recall": float(df["context_recall"][0]),
        }
    except Exception:
        # Some versions return .results already as dict
        return getattr(res, "results", {"error": "Unexpected ragas result format"})
