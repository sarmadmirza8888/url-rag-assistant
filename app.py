"""Gradio interface for the URL-based RAG assistant.  Run:  python app.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import gradio as gr  # noqa: E402

from url_rag.pipeline import answer_from_url  # noqa: E402


def run_rag(url, question):
    if not url or not url.strip():
        return "Please enter a webpage URL."
    if not question or not question.strip():
        return "Please enter a question."
    try:
        return answer_from_url(raw_url=url.strip(), question=question.strip(), top_k=3)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


with gr.Blocks(title="URL-Based RAG Assistant") as demo:
    gr.Markdown("# URL-Based RAG Assistant")
    gr.Markdown(
        "Enter a webpage URL, ask a question, "
        "and receive an answer from your local Mistral model."
    )
    url_input = gr.Textbox(
        label="Webpage URL", placeholder="https://docs.python.org/3/tutorial/"
    )
    question_input = gr.Textbox(
        label="Your question",
        placeholder="What is the Python tutorial designed to teach?",
        lines=2,
    )
    submit_button = gr.Button("Ask Question", variant="primary")
    answer_output = gr.Textbox(label="Mistral's Answer", lines=8)
    submit_button.click(
        fn=run_rag, inputs=[url_input, question_input], outputs=answer_output
    )

if __name__ == "__main__":
    demo.launch(share=os.getenv("GRADIO_SHARE", "0") == "1")
