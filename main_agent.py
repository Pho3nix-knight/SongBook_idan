import os
import gradio as gr

from helpers.qa_rag import ask
from agents.summary_agent import summarize_rag
from ragas_evaluations.ragas_eval_summarize import evaluate_summary

def ask_qa_rag(prompt: str):
    answer = ask(prompt)
    return answer


_last_response = {"question": None, "answer": None, "contexts": None, "metrics": None}

def router_agent(operation: str, prompt: str):
    if operation == "qna":
        ans = ask_qa_rag(prompt)
        _last_response.update({"question": prompt, "answer": ans, "contexts": [], "metrics": None})
        return ans
    elif operation == "summarize":
        out = summarize_rag(prompt, top_k=int(os.getenv("TOP_K", "5")))
        metrics = evaluate_summary(prompt, out["answer"], out["contexts"])
        _last_response.update({"question": prompt, "answer": out["answer"], "contexts": out["contexts"], "metrics": metrics})
        return out["answer"] + f"\n\n**RAGAS**\nFaithfulness: {metrics['faithfulness']:.3f}"
    else:
        return "Unknown operation selected."
    


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
        gr.Markdown("Ask about the songbook • Choose QnA or Summarize")

        with gr.Row():
            operation = gr.Dropdown(label="Operation", choices=["qna", "summarize"], value="qna", scale=0)
            prompt = gr.Textbox(label="Prompt", placeholder="Ask a question or request a summary", lines=3)

        
        with gr.Row():
            submit = gr.Button("Submit", elem_id="submit_btn")
            clear = gr.Button("Clear", elem_id="clear_btn")
            
        output = gr.Markdown()

        submit.click(router_agent, inputs=[operation, prompt], outputs=output)
        clear.click(fn=lambda: ("", ""), inputs=None, outputs=[prompt, output])

    demo.launch()


if __name__ == "__main__":
    main()
