
# This is an old script that we have to change!
# python ragas_eval_summarize.py


from ragas.metrics import answer_correctness
from ragas.evaluation import evaluate
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.config import SUMMARY_RESULT_PATH, SUMMARY_GROUND_TRUTH_PATH, RAGAS_SUMMARIZE_EVALUATION_PATH, FILE_PATH, \
    OPENAI_API_KEY
from langchain_openai import ChatOpenAI
from datasets import Dataset


def get_summary():
    return open(SUMMARY_RESULT_PATH, 'r', encoding='utf-8').read()


def truncate_text(text, max_chars=1500):
    """Truncate text to prevent token overflow"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def smart_truncate_summary(text, max_chars=3000):
    """Smart truncation that tries to preserve complete sections"""
    if len(text) <= max_chars:
        return text
    
    # Try to find a good breaking point near the limit
    # Look for section breaks or paragraph breaks
    truncated = text[:max_chars]
    
    # Try to break at a section header or paragraph
    last_section = truncated.rfind('\n\n')
    if last_section > max_chars * 0.7:  # If we can find a good break point in the last 30%
        return text[:last_section] + "..."
    
    # Try to break at a single newline
    last_newline = truncated.rfind('\n')
    if last_newline > max_chars * 0.8:
        return text[:last_newline] + "..."
    
    return truncated + "..."


def make_summarize_evaluation():
    print("Making summarize evaluation...")
    # Use a reasonable max_tokens value that works with the model
    llm = ChatOpenAI(
        temperature=0, 
        api_key=OPENAI_API_KEY, 
        max_tokens=4000,  # Increased to allow for longer evaluation
        model="gpt-3.5-turbo-16k"  # Use 16k model for larger context
    )

    question = "Summarize the document in the best way possible. Focus on creating a clear and structured summary"
    
    # Read and truncate all inputs to prevent token overflow
    context = open(FILE_PATH, 'r', encoding='utf-8').read()
    summary = get_summary()
    ground_truth = open(SUMMARY_GROUND_TRUTH_PATH, 'r', encoding='utf-8').read()

    # Use much larger limits to preserve more content
    context_truncated = truncate_text(context, 4000)  # Increased from 2000
    summary_truncated = smart_truncate_summary(summary, 4000)  # Increased from 1000
    ground_truth_truncated = smart_truncate_summary(ground_truth, 4000)  # Increased from 1000

    print(f"Context length: {len(context_truncated)} chars")
    print(f"Summary length: {len(summary_truncated)} chars")
    print(f"Ground truth length: {len(ground_truth_truncated)} chars")

    eval_dataset = Dataset.from_dict({
        "question": [question],
        "context": [context_truncated],
        "answer": [summary_truncated],
        "ground_truth": [ground_truth_truncated]
    })

    try:
        print("Starting evaluation...")
        results = evaluate(
            eval_dataset,
            metrics=[answer_correctness],
            llm=llm
        )

        answer_correctness_score = results["answer_correctness"][0]
        print("\n=== RAGAS Answer Correctness Score ===")
        print(f"Answer Correctness: {answer_correctness_score:.4f}")

        with open(RAGAS_SUMMARIZE_EVALUATION_PATH, "w", encoding="utf-8") as f:
            f.write("=== RAGAS ANSWER CORRECTNESS EVALUATION FOR SUMMARY ===\n")
            f.write(f"Answer Correctness Score: {answer_correctness_score:.4f}\n")
            f.write(f"\nInput lengths:\n")
            f.write(f"- Context: {len(context_truncated)} chars\n")
            f.write(f"- Summary: {len(summary_truncated)} chars\n")
            f.write(f"- Ground Truth: {len(ground_truth_truncated)} chars\n")
            f.write("\n--- SUMMARY (truncated) ---\n")
            f.write(summary_truncated)
            f.write("\n\n--- GROUND TRUTH (truncated) ---\n")
            f.write(ground_truth_truncated)
            f.write("\n\n--- CONTEXT (truncated) ---\n")
            f.write(context_truncated)

    except Exception as e:
        print(f"❌ Error evaluating answer correctness: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Write error details to file for debugging
        with open(RAGAS_SUMMARIZE_EVALUATION_PATH, "w", encoding="utf-8") as f:
            f.write("=== RAGAS ANSWER CORRECTNESS EVALUATION FOR SUMMARY ===\n")
            f.write(f"ERROR: {str(e)}\n")
            f.write(f"Error type: {type(e).__name__}\n")
            f.write(f"\nInput lengths:\n")
            f.write(f"- Context: {len(context_truncated)} chars\n")
            f.write(f"- Summary: {len(summary_truncated)} chars\n")
            f.write(f"- Ground Truth: {len(ground_truth_truncated)} chars\n")
            f.write("\n--- SUMMARY (truncated) ---\n")
            f.write(summary_truncated)
            f.write("\n\n--- GROUND TRUTH (truncated) ---\n")
            f.write(ground_truth_truncated)
            f.write("\n\n--- CONTEXT (truncated) ---\n")
            f.write(context_truncated)

# make_summarize_evaluation()
