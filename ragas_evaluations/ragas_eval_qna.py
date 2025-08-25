import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.qa_rag import ask
# (הסרת ייבוא לא בשימוש של BM25Retriever)
# from helpers.hybride_retreval import BM25Retriever

# ⬇️ הוספתי: נשתמש ישירות ברטריבר כדי להביא קונטקסטים
from helpers.hybride_retreval import hybrid_search, load_jsonl_to_docs
from helpers.config import CHUNKS_PATH, CHROMA_DIR

from ragas.metrics import faithfulness, context_precision, context_recall
from ragas.evaluation import evaluate
from datasets import Dataset
from helpers.config import RAGAS_QNA_EVALUATION_PATH
from difflib import get_close_matches


# Load ground truth data from the updated JSON file
def load_ground_truth_data():
    """Load questions and answers from the ground truth monologue file."""
    ground_truth_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "helpers",
        "ground_truth_monologue.json"
    )
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = [item["question"] for item in data]
    ground_truths = [item["answer"] for item in data]
    return questions, ground_truths


# Load the data
questions, ground_truths = load_ground_truth_data()


def find_ground_truth(prompt: str):
    """
    Find the closest question and corresponding ground truth.
    """
    match = get_close_matches(prompt, questions, n=1, cutoff=0.5)
    if not match:
        raise ValueError("No close question found for the given prompt.")
    index = questions.index(match[0])
    return match[0], ground_truths[index]


# ⬇️ הוספתי: פונקציה שמביאה קונטקסטים (סניפטים) מהרטריבר ההיברידי שלך
def get_context_snippets(question: str, k: int = 6, snippet_len: int = 800):
    """
    Retrieve top-k docs via the hybrid retriever and return a list[str] of snippets.
    RAGAS expects 'contexts' to be a list of context strings per sample.
    """
    try:
        # טוען מסמכים ובונה רטריבר (ב-runner אמיתי אפשר להחזיק גלובלי)
        print(f"📚 Loading documents from {CHUNKS_PATH}...")
        docs = load_jsonl_to_docs(CHUNKS_PATH)
        print(f"✅ Loaded {len(docs)} documents")
        
        print("🔧 Using hybrid retriever...")
        
        print(f"🔍 Searching for context with query: {question[:50]}...")
        hits = hybrid_search(question, k=k, month_year=None)
        print(f"✅ Found {len(hits)} relevant documents")
        
        contexts = []
        for d in hits:
            text = d.page_content.strip().replace("\n", " ")
            contexts.append(text[:snippet_len])
        
        print(f"📝 Generated {len(contexts)} context snippets")
        return contexts
        
    except Exception as e:
        print(f"❌ Error in get_context_snippets: {e}")
        print("⚠️ Returning empty context list")
        return []


# ⬇️ אופציונלי אבל עוזר למדדים: להסיר "מקורות" מהתשובה לפני הערכה
# def strip_sources(answer: str) -> str:
#     marker = "\n\n**מקורות:**"
#     i = answer.find(marker)
#     return answer[:i].strip() if i != -1 else answer.strip()


def make_qna_ragas_evaluation(prompt: str):
    print(f"🔍 Evaluating QnA for: {prompt}")
    question, ground_truth = find_ground_truth(prompt)

    # Get answer and context
    answer = ask(prompt)

    # ----------------------------
    # Get context - ✅ DONE: fetch from hybrid retriever
    # ----------------------------
    context = get_context_snippets(prompt, k=6, snippet_len=800)
    # ----------------------------

    # (אופציונלי) נקה את קטע ה"מרקרים/מקורות" מהתשובה כדי שלא יפגע במדדים
    # answer = strip_sources(answer_raw)

    # Print for debug
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    print(f"Ground Truth: {ground_truth}\n")
    print(f"Contexts ({len(context)}):\n- " + "\n- ".join(c[:120] + "..." for c in context))

    dataset = Dataset.from_dict({
        "question": [prompt],
        "contexts": [context],        # <- list[str] per row, כמו שראגאס מצפה
        "response": [answer],
        "ground_truth": [ground_truth]
    })

    results = evaluate(
        dataset,
        metrics=[faithfulness, context_precision, context_recall]
    )

    results_dict = {
        "faithfulness": float(results["faithfulness"][0]),
        "context_precision": float(results["context_precision"][0]),
        "context_recall": float(results["context_recall"][0])
    }

    print("=== RAGAS EVALUATION RESULTS FOR Q&A ===\n")
    for key, val in results_dict.items():
        print(f"{key.replace('_', ' ').title()}: {val:.4f}\n")

    with open(RAGAS_QNA_EVALUATION_PATH, "w", encoding="utf-8") as f:
        f.write("=== RAGAS EVALUATION FOR SINGLE Q&A ===\n")
        f.write(f"Question: {prompt}\n")
        f.write(f"Answer: {answer}\n")
        f.write(f"Ground Truth: {ground_truth}\n\n")
        f.write("=== METRICS ===\n")
        for key, val in results_dict.items():
            f.write(f"{key.replace('_', ' ').title()}: {val:.4f}\n")

    print("✅ Evaluation complete. Check the output file.")
    return answer


# Example usage with a Hebrew question from the ground truth
make_qna_ragas_evaluation("בשיר 'איך לשכוח שאתה מאוהב', איך הדובר מתאר את התהליך של ניסיון לשכוח להיות מאוהב?")
