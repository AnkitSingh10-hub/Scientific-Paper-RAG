"""
Gradio dashboard for visualizing eval.py results instead of running it via CLI.

SETUP
-----
1. Drop this file into the same folder as eval.py, test.py, retrieval.py,
   pipeline.py, llm.py, vector_store.py, embeddings.py (your project root).
2. Install the extra deps (everything else you already have):
       pip install gradio plotly pandas --break-system-packages
3. Run it:
       python gradio_eval.py
   Gradio will print a local URL (and a public one if share=True).

WHAT IT DOES
------------
Single Test tab:
    - Pick one question from tests.jsonl
    - Runs evaluate_retrieval() + evaluate_answer() (same functions eval.py uses)
    - Shows MRR / nDCG / keyword coverage as a bar chart
    - Shows LLM-judge accuracy/completeness/relevance as a bar chart
    - Shows generated answer vs reference answer + judge feedback
    - Shows the retrieved context chunks

Full Evaluation tab:
    - Runs evaluate_all_retrieval() + evaluate_all_answers() over every test
    - Shows average metrics as bar charts
    - Shows a per-category breakdown chart
    - Shows full per-test result tables (sortable in the Gradio UI)
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import gradio as gr

from test import load_tests
from eval import (
    evaluate_retrieval,
    evaluate_answer,
    evaluate_all_retrieval,
    evaluate_all_answers,
)

TESTS = load_tests()


def _test_choices():
    return [f"{i}: {t.question[:70]}" for i, t in enumerate(TESTS)]


def run_single_test(test_choice, progress=gr.Progress()):
    idx = int(test_choice.split(":")[0])
    test = TESTS[idx]

    progress(0.1, desc="Running retrieval...")
    retrieval_result = evaluate_retrieval(test)

    progress(0.5, desc="Generating answer + LLM judging...")
    answer_result, generated_answer, retrieved_docs = evaluate_answer(test)

    progress(1.0, desc="Done")

    retrieval_fig = go.Figure(
        go.Bar(
            x=["MRR", "nDCG", "Keyword Coverage %"],
            y=[
                retrieval_result.mrr,
                retrieval_result.ndcg,
                retrieval_result.keyword_coverage,
            ],
            marker_color=["#6366f1", "#8b5cf6", "#ec4899"],
            text=[
                f"{retrieval_result.mrr:.3f}",
                f"{retrieval_result.ndcg:.3f}",
                f"{retrieval_result.keyword_coverage:.1f}%",
            ],
            textposition="outside",
        )
    )
    retrieval_fig.update_layout(
        title="Retrieval Metrics",
        yaxis_title="Score",
        template="plotly_white",
        height=350,
    )

    answer_fig = go.Figure(
        go.Bar(
            x=["Accuracy", "Completeness", "Relevance"],
            y=[
                answer_result.accuracy,
                answer_result.completeness,
                answer_result.relevance,
            ],
            marker_color=["#10b981", "#f59e0b", "#3b82f6"],
            text=[
                f"{answer_result.accuracy:.1f}",
                f"{answer_result.completeness:.1f}",
                f"{answer_result.relevance:.1f}",
            ],
            textposition="outside",
        )
    )
    answer_fig.update_layout(
        title="Answer Quality (LLM-judged, 1-5)",
        yaxis=dict(range=[0, 5]),
        template="plotly_white",
        height=350,
    )

    retrieved_context = "\n\n---\n\n".join(retrieved_docs)

    summary_md = f"""### Question
{test.question}

**Category:** {test.category}  |  **Keywords:** {", ".join(test.keywords)}  |  **Keywords found:** {retrieval_result.keywords_found}/{retrieval_result.total_keywords}
"""

    return (
        summary_md,
        retrieval_fig,
        answer_fig,
        generated_answer,
        test.reference_answer,
        answer_result.feedback,
        retrieved_context,
    )


def run_full_eval(progress=gr.Progress()):
    n = len(TESTS)

    retrieval_rows = []
    for i, (test, result, prog) in enumerate(evaluate_all_retrieval(TESTS)):
        retrieval_rows.append(
            {
                "idx": i,
                "question": test.question[:60],
                "category": test.category,
                "mrr": result.mrr,
                "ndcg": result.ndcg,
                "coverage_%": result.keyword_coverage,
            }
        )
        progress(prog * 0.5, desc=f"Retrieval eval {i + 1}/{n}")

    retrieval_df = pd.DataFrame(retrieval_rows)

    answer_rows = []
    for i, (test, result, prog) in enumerate(evaluate_all_answers(TESTS)):
        answer_rows.append(
            {
                "idx": i,
                "question": test.question[:60],
                "category": test.category,
                "accuracy": result.accuracy,
                "completeness": result.completeness,
                "relevance": result.relevance,
            }
        )
        progress(0.5 + prog * 0.5, desc=f"Answer eval {i + 1}/{n}")

    answer_df = pd.DataFrame(answer_rows)

    summary_fig = go.Figure(
        go.Bar(
            x=["MRR", "nDCG", "Coverage (÷100)"],
            y=[
                retrieval_df.mrr.mean(),
                retrieval_df.ndcg.mean(),
                retrieval_df["coverage_%"].mean() / 100,
            ],
            marker_color="#6366f1",
        )
    )
    summary_fig.update_layout(
        title="Average Retrieval Metrics",
        template="plotly_white",
        height=350,
        yaxis=dict(range=[0, 1]),
    )

    answer_summary_fig = go.Figure(
        go.Bar(
            x=["Accuracy", "Completeness", "Relevance"],
            y=[
                answer_df.accuracy.mean(),
                answer_df.completeness.mean(),
                answer_df.relevance.mean(),
            ],
            marker_color="#10b981",
        )
    )
    answer_summary_fig.update_layout(
        title="Average Answer Quality",
        template="plotly_white",
        height=350,
        yaxis=dict(range=[0, 5]),
    )

    cat_fig = px.bar(
        retrieval_df.groupby("category")[["mrr", "ndcg"]].mean().reset_index(),
        x="category",
        y=["mrr", "ndcg"],
        barmode="group",
        template="plotly_white",
        title="Retrieval Metrics by Category",
    )
    cat_fig.update_layout(height=350)

    return (
        summary_fig,
        answer_summary_fig,
        cat_fig,
        retrieval_df,
        answer_df,
    )


with gr.Blocks(title="RAG Evaluation Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📊 RAG Evaluation Dashboard")
    gr.Markdown(
        "Visualize retrieval and answer-quality metrics from `eval.py` without reading CLI output."
    )

    with gr.Tab("Single Test"):
        with gr.Row():
            test_dropdown = gr.Dropdown(
                choices=_test_choices(),
                label="Select test question",
                value=_test_choices()[0] if TESTS else None,
            )
            run_btn = gr.Button("Run Evaluation", variant="primary")

        summary_out = gr.Markdown()

        with gr.Row():
            retrieval_plot = gr.Plot(label="Retrieval Metrics")
            answer_plot = gr.Plot(label="Answer Quality")

        with gr.Accordion("Generated vs Reference Answer", open=True):
            with gr.Row():
                generated_out = gr.Textbox(label="Generated Answer", lines=6)
                reference_out = gr.Textbox(label="Reference Answer", lines=6)
            feedback_out = gr.Textbox(label="Judge Feedback", lines=4)

        with gr.Accordion("Retrieved Context", open=False):
            context_out = gr.Textbox(lines=15)

        run_btn.click(
            run_single_test,
            inputs=[test_dropdown],
            outputs=[
                summary_out,
                retrieval_plot,
                answer_plot,
                generated_out,
                reference_out,
                feedback_out,
                context_out,
            ],
        )

    with gr.Tab("Full Evaluation"):
        gr.Markdown(
            "Runs every question in `tests.jsonl` through retrieval + generation + "
            "LLM judging. This calls your LLM API once per question for judging plus "
            "once for generation, so it can take a while and will incur API cost."
        )
        run_all_btn = gr.Button("Run Full Evaluation Suite", variant="primary")

        with gr.Row():
            summary_plot = gr.Plot(label="Avg Retrieval Metrics")
            answer_summary_plot = gr.Plot(label="Avg Answer Metrics")

        category_plot = gr.Plot(label="By Category")

        with gr.Row():
            retrieval_table = gr.Dataframe(label="Per-Test Retrieval Results")
            answer_table = gr.Dataframe(label="Per-Test Answer Results")

        run_all_btn.click(
            run_full_eval,
            inputs=[],
            outputs=[
                summary_plot,
                answer_summary_plot,
                category_plot,
                retrieval_table,
                answer_table,
            ],
        )

if __name__ == "__main__":
    demo.launch()
