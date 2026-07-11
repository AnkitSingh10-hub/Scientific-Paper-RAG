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

import os
import datetime

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
from retrieval import embedder as retrieval_embedder
from llm import DEFAULT_MODEL as LLM_MODEL
from vector_store import collection as chroma_collection

TESTS = load_tests()

# Results are saved relative to wherever you launch the script from
# (matches the same convention vector_store.py uses for "db/chroma_db").
RESULTS_DIR = "results"
RUNS_LOG_PATH = os.path.join(RESULTS_DIR, "runs_log.csv")


def _test_choices():
    return [f"{i}: {t.question[:70]}" for i, t in enumerate(TESTS)]


def _run_metadata():
    """Pulled live from your actual code, not typed in manually, so it can't drift."""
    return {
        "embedding_model": retrieval_embedder.model_name,
        "llm_model": LLM_MODEL,
        "chroma_collection": chroma_collection.name,
    }


def _config_banner():
    meta = _run_metadata()
    return (
        f"**Embedding model:** `{meta['embedding_model']}`  |  "
        f"**LLM model:** `{meta['llm_model']}`  |  "
        f"**Chroma collection:** `{meta['chroma_collection']}`"
    )


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
        retrieval_df,  # duplicated into gr.State so the Save button can access it later
        answer_df,
    )


def save_full_eval(retrieval_df, answer_df, note):
    """Write this run's full results + metadata to results/, and append a
    summary row to results/runs_log.csv for cross-run comparison."""
    if retrieval_df is None or answer_df is None or retrieval_df.empty:
        return "⚠️ Run a full evaluation first, then save.", load_runs_log()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    meta = _run_metadata()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_embed_name = meta["embedding_model"].replace("/", "-")
    run_id = f"{timestamp}_{safe_embed_name}"

    merged = retrieval_df.merge(
        answer_df, on=["idx", "question", "category"], suffixes=("", "_ans")
    )
    for key, val in meta.items():
        merged[key] = val
    merged["note"] = note
    merged["timestamp"] = timestamp

    detail_path = os.path.join(RESULTS_DIR, f"{run_id}.csv")
    merged.to_csv(detail_path, index=False)

    summary_row = {
        "timestamp": timestamp,
        **meta,
        "note": note,
        "avg_mrr": round(retrieval_df["mrr"].mean(), 4),
        "avg_ndcg": round(retrieval_df["ndcg"].mean(), 4),
        "avg_coverage_%": round(retrieval_df["coverage_%"].mean(), 2),
        "avg_accuracy": round(answer_df["accuracy"].mean(), 3),
        "avg_completeness": round(answer_df["completeness"].mean(), 3),
        "avg_relevance": round(answer_df["relevance"].mean(), 3),
        "n_tests": len(retrieval_df),
        "detail_file": f"{run_id}.csv",
    }
    log_row_df = pd.DataFrame([summary_row])

    # Writing runs_log.csv can hit a transient PermissionError on Windows if
    # it's open in Excel or being synced by OneDrive. Retry briefly, then
    # fail gracefully - the detail CSV above already saved successfully, so
    # we don't want a locked log file to blow up the whole callback.
    import time

    log_write_error = None
    for attempt in range(5):
        try:
            if os.path.exists(RUNS_LOG_PATH):
                log_row_df.to_csv(RUNS_LOG_PATH, mode="a", header=False, index=False)
            else:
                log_row_df.to_csv(RUNS_LOG_PATH, index=False)
            log_write_error = None
            break
        except PermissionError as e:
            log_write_error = e
            time.sleep(0.5 * (attempt + 1))

    if log_write_error is not None:
        status = (
            f"✅ Saved detailed results to `{detail_path}`.\n\n"
            f"⚠️ Could NOT update `{RUNS_LOG_PATH}` - it looks like it's locked "
            f"(commonly because it's open in Excel, or being synced by OneDrive "
            f"if this project lives under a synced folder). Close whatever has "
            f"it open and click Save again; your detailed run wasn't lost, "
            f"it's already in `{detail_path}`."
        )
        return status, load_runs_log()

    status = f"✅ Saved detailed results to `{detail_path}` and logged summary to `{RUNS_LOG_PATH}`."
    return status, load_runs_log()


def load_runs_log():
    if os.path.exists(RUNS_LOG_PATH):
        return pd.read_csv(RUNS_LOG_PATH)
    return pd.DataFrame(
        columns=[
            "timestamp",
            "embedding_model",
            "llm_model",
            "chroma_collection",
            "note",
            "avg_mrr",
            "avg_ndcg",
            "avg_coverage_%",
            "avg_accuracy",
            "avg_completeness",
            "avg_relevance",
            "n_tests",
            "detail_file",
        ]
    )


def build_comparison_chart():
    log_df = load_runs_log()
    if log_df.empty:
        return log_df, go.Figure(), go.Figure()

    log_df = log_df.copy()
    log_df["run_label"] = log_df["timestamp"] + " | " + log_df["embedding_model"]

    retrieval_fig = go.Figure()
    for metric, color in [("avg_mrr", "#6366f1"), ("avg_ndcg", "#8b5cf6")]:
        retrieval_fig.add_trace(
            go.Bar(
                name=metric, x=log_df["run_label"], y=log_df[metric], marker_color=color
            )
        )
    retrieval_fig.update_layout(
        title="Retrieval Quality Across Runs",
        barmode="group",
        template="plotly_white",
        height=380,
        xaxis_tickangle=-30,
    )

    answer_fig = go.Figure()
    for metric, color in [
        ("avg_accuracy", "#10b981"),
        ("avg_completeness", "#f59e0b"),
        ("avg_relevance", "#3b82f6"),
    ]:
        answer_fig.add_trace(
            go.Bar(
                name=metric, x=log_df["run_label"], y=log_df[metric], marker_color=color
            )
        )
    answer_fig.update_layout(
        title="Answer Quality Across Runs",
        barmode="group",
        template="plotly_white",
        height=380,
        xaxis_tickangle=-30,
    )

    return log_df, retrieval_fig, answer_fig


with gr.Blocks(title="RAG Evaluation Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📊 RAG Evaluation Dashboard")
    gr.Markdown(
        "Visualize retrieval and answer-quality metrics from `eval.py` without reading CLI output."
    )
    gr.Markdown(_config_banner())

    # Holds the last full-eval dataframes so the Save button can use them
    # without re-running the whole evaluation.
    retrieval_state = gr.State(None)
    answer_state = gr.State(None)

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
                retrieval_state,
                answer_state,
            ],
        )

        gr.Markdown("---")
        gr.Markdown(
            "### Save this run\n"
            "Saves a detailed CSV plus a summary row (tagged with the embedding model, "
            "LLM model, and Chroma collection shown above) to `results/`, so you can "
            "compare configs later in the **Run History** tab."
        )
        with gr.Row():
            note_input = gr.Textbox(
                label="Note (e.g. chunker/config used for this run)",
                placeholder="e.g. TokenChunker(chunk_size=1500, overlap=100)",
                scale=3,
            )
            save_btn = gr.Button("💾 Save Results", variant="secondary", scale=1)
        save_status = gr.Markdown()

    with gr.Tab("Run History"):
        gr.Markdown(
            "All runs you've saved with the button above, so you can compare "
            "embedding models / LLMs / chunking configs side by side."
        )
        refresh_btn = gr.Button("🔄 Refresh")
        history_table = gr.Dataframe(label="Saved Runs", value=load_runs_log())
        with gr.Row():
            history_retrieval_plot = gr.Plot(label="Retrieval Comparison")
            history_answer_plot = gr.Plot(label="Answer Quality Comparison")

        def _refresh_history():
            log_df, r_fig, a_fig = build_comparison_chart()
            return log_df, r_fig, a_fig

        refresh_btn.click(
            _refresh_history,
            inputs=[],
            outputs=[history_table, history_retrieval_plot, history_answer_plot],
        )
        demo.load(
            _refresh_history,
            inputs=[],
            outputs=[history_table, history_retrieval_plot, history_answer_plot],
        )

    save_btn.click(
        save_full_eval,
        inputs=[retrieval_state, answer_state, note_input],
        outputs=[save_status, history_table],
    )

if __name__ == "__main__":
    demo.launch()
