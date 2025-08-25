# from chains.qa_chain import run_qa_chain
# from ragas_evaluations.ragas_eval_summarize import get_summary
# from ragas_evaluations.ragas_eval_qna import make_qna_ragas_evaluation
# from ragas_evaluations.ragas_eval_summarize import make_summarize_evaluation
import os
import gradio as gr

from helpers.qa_rag import ask

# from agents.summary_agent import summarize_rag
# from eval.ragas_eval import evaluate_with_ragas

def ask_qa_rag(prompt: str):
    answer = ask(prompt)
    return answer


_last_response = {"question": None, "answer": None, "contexts": None}

def router_agent(operation: str, prompt: str):
    if operation == "qna":
        ans = ask_qa_rag(prompt)
        _last_response.update({"question": prompt, "answer": ans, "contexts": []})
        return ans
    elif operation == "summarize":
        # out = summarize_rag(prompt, top_k=int(os.getenv("TOP_K", "5")))
        # _last_response.update({"question": prompt, "answer": out["answer"], "contexts": out["contexts"]})
        # return out["answer"]
        return "בקרוב בקולנוע"
    else:
        return "Unknown operation selected."
    
# def run_ragas(_):
#     if not _last_response["answer"] or _last_response["question"] is None:
#         return "No previous answer to evaluate. Ask something first."
#     metrics = evaluate_with_ragas(
#         question=_last_response["question"],
#         answer=_last_response["answer"],
#         contexts=_last_response["contexts"] or [],
#         ground_truth=None,  # optionally pass a reference summary here
#     )
#     if "error" in metrics:
#         return f"RAGAS error: {metrics['error']}"
#     # pretty print
#     return (
#         f"**RAGAS Metrics**\n\n"
#         f"- Faithfulness: {metrics.get('faithfulness', 'n/a'):.3f}\n"
#         f"- Answer Relevancy: {metrics.get('answer_relevancy', 'n/a'):.3f}\n"
#         f"- Context Precision: {metrics.get('context_precision', 'n/a'):.3f}\n"
#         f"- Context Recall: {metrics.get('context_recall', 'n/a'):.3f}\n"
#     )


def main():
    with gr.Blocks(
        title="Songbook RAG",
        theme=gr.themes.Soft(),
        css="""
        #submit_btn { background: #2563eb; color: white; border: none; }
        #submit_btn:hover { filter: brightness(1.05); }
        #clear_btn  { background: transparent; color: #ef4444; border: 1px solid #ef4444; }
        #clear_btn:hover { background: rgba(239,68,68,0.08); }
        """
        ) as demo:
        gr.Markdown("# 🤖 Songbook RAG")
        gr.Markdown("Ask about the songbook • Choose QnA or Summarize • (Optional) evaluate with RAGAS")
        # gr.Markdown("# 🤖 Songbook RAG")
        # gr.Markdown("Enter a question about the songbook.")

        with gr.Row():
            operation = gr.Dropdown(label="Operation", choices=["qna", "summarize"], value="qna", scale=0)
            prompt = gr.Textbox(label="Prompt", placeholder="Ask a question or request a summary", lines=3)

        
        with gr.Row():
            submit = gr.Button("Submit", elem_id="submit_btn")
            clear = gr.Button("Clear", elem_id="clear_btn")
            
        output = gr.Markdown()
        
        # submit.click(ask_qa_rag, inputs=prompt, outputs=output)
        submit.click(router_agent, inputs=[operation, prompt], outputs=output)
        clear.click(fn=lambda: ("", ""), inputs=None, outputs=[prompt, output])

    demo.launch()


if __name__ == "__main__":
    main()
