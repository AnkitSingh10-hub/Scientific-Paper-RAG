import sys
import math
import re
from collections import defaultdict
from pydantic import BaseModel, Field

from test import TestQuestion, load_tests
from retrieval import retrieve
from pipeline import ask_with_context
from llm import generate_json


class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance."""

    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(
        description="Normalized Discounted Cumulative Gain (binary relevance)"
    )
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

    feedback: str = Field(
        description="Concise feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context"
    )
    accuracy: float = Field(
        description="How factually correct is the answer compared to the reference answer? 1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). An acceptable answer would score 3."
    )
    completeness: float = Field(
        description="How complete is the answer in addressing all aspects of the question? 1 (very poor - missing key information) to 5 (ideal - all the information from the reference answer is provided completely). Only answer 5 if ALL information from the reference answer is included."
    )
    relevance: float = Field(
        description="How relevant is the answer to the specific question asked? 1 (very poor - off-topic) to 5 (ideal - directly addresses question and gives no additional information). Only answer 5 if the answer is completely relevant to the question and gives no additional information."
    )


def _parse_json_response(raw: str) -> str:
    """Strip markdown code fences some models still wrap JSON in."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _normalize_for_matching(text: str) -> str:
    """Normalize text for keyword containment checks.

    Strips punctuation (commas, hyphens, parens, colons, etc.) and collapses
    whitespace so a keyword still matches when the source text has minor
    formatting differences that don't change its meaning - e.g. the tuple
    keyword "S, A, P, R" matching "(S, A, P, R)" in the source, or
    "Retrieval-Augmented" matching text that got a stray hyphen/space at a
    line-wrap. Content inside parentheses is KEPT (not deleted) since some
    parentheticals carry the actual fact being asked about (e.g. "(157 from
    databases, 8 from supplemental sources)"). This does NOT do fuzzy/
    semantic matching - the keyword's words must still appear, in order, as
    a contiguous (post-normalization) span.
    """
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip().lower()


def calculate_mrr(keyword: str, retrieved_docs: list[str]) -> float:
    """Calculate reciprocal rank for a single keyword (case-insensitive,
    tolerant of parentheticals/punctuation/whitespace differences)."""
    keyword_norm = _normalize_for_matching(keyword)
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_norm in _normalize_for_matching(doc):
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    """Calculate Discounted Cumulative Gain."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)  # i+2 because rank starts at 1
    return dcg


def calculate_ndcg(keyword: str, retrieved_docs: list[str], k: int = 10) -> float:
    """Calculate nDCG for a single keyword (binary relevance, case-insensitive,
    tolerant of parentheticals/punctuation/whitespace differences)."""
    keyword_norm = _normalize_for_matching(keyword)

    relevances = [
        1 if keyword_norm in _normalize_for_matching(doc) else 0
        for doc in retrieved_docs[:k]
    ]

    dcg = calculate_dcg(relevances, k)

    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(test: TestQuestion, k: int = 10) -> RetrievalEval:
    """
    Evaluate retrieval performance for a test question.

    Args:
        test: TestQuestion object containing question and keywords
        k: Number of top chunks to retrieve (default 6)

    Returns:
        RetrievalEval object with MRR, nDCG, and keyword coverage metrics

    NOTE: For "unanswerable" category tests (no keywords by design), this
    always returns mrr=0.0, ndcg=0.0, keyword_coverage=0.0. That's expected,
    not a retrieval failure - see summarize_retrieval() and main() for how
    these are excluded from the headline retrieval averages.
    """
    results = retrieve(test.question, k=k)
    retrieved_docs = results["documents"][0]

    mrr_scores = [calculate_mrr(keyword, retrieved_docs) for keyword in test.keywords]
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    ndcg_scores = [
        calculate_ndcg(keyword, retrieved_docs, k) for keyword in test.keywords
    ]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    keywords_found = sum(1 for score in mrr_scores if score > 0)
    total_keywords = len(test.keywords)
    keyword_coverage = (
        (keywords_found / total_keywords * 100) if total_keywords > 0 else 0.0
    )

    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )


def evaluate_answer(test: TestQuestion) -> tuple[AnswerEval, str, list[str]]:
    """
    Evaluate answer quality using LLM-as-a-judge.

    Args:
        test: TestQuestion object containing question and reference answer

    Returns:
        Tuple of (AnswerEval object, generated_answer string, retrieved_docs list)
    """
    generated_answer, retrieved_docs = ask_with_context(test.question)

    judge_system_prompt = (
        "You are an expert evaluator assessing the quality of answers. Evaluate "
        "the generated answer by comparing it to the reference answer. Only give "
        "5/5 scores for perfect answers. Respond with a single raw JSON object "
        "with exactly these keys: feedback (string), accuracy (number 1-5), "
        "completeness (number 1-5), relevance (number 1-5). No markdown, no "
        "commentary, just the JSON object."
    )

    judge_user_prompt = f"""Question:
{test.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
2. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1."""

    raw_response = generate_json(judge_system_prompt, judge_user_prompt)
    answer_eval = AnswerEval.model_validate_json(_parse_json_response(raw_response))

    return answer_eval, generated_answer, retrieved_docs


def evaluate_all_retrieval(tests: list[TestQuestion] = None):
    """Evaluate all retrieval tests, yielding (test, result, progress) as it goes."""
    tests = tests if tests is not None else load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_retrieval(test)
        progress = (index + 1) / total_tests
        yield test, result, progress


def evaluate_all_answers(tests: list[TestQuestion] = None):
    """Evaluate all answers, yielding (test, result, progress) as it goes."""
    tests = tests if tests is not None else load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_answer(test)[0]
        progress = (index + 1) / total_tests
        yield test, result, progress


def _avg(results: list, attr: str) -> float:
    values = [getattr(r, attr) for r in results]
    return sum(values) / len(values) if values else 0.0


def summarize_retrieval(results: list[tuple[TestQuestion, RetrievalEval]]) -> dict:
    """Aggregate retrieval results, split by category.

    "unanswerable" tests have no keywords by design and always score 0 on
    retrieval metrics (see evaluate_retrieval docstring) - so they're
    reported per-category but excluded from the "overall" scoreable average
    to avoid silently deflating your headline MRR/nDCG/coverage numbers.
    """
    by_category = defaultdict(list)
    for test, result in results:
        by_category[test.category].append(result)

    per_category = {
        category: {
            "n": len(res_list),
            "mrr": _avg(res_list, "mrr"),
            "ndcg": _avg(res_list, "ndcg"),
            "coverage": _avg(res_list, "keyword_coverage"),
        }
        for category, res_list in by_category.items()
    }

    scoreable = [result for test, result in results if test.category != "unanswerable"]
    overall = {
        "n": len(scoreable),
        "mrr": _avg(scoreable, "mrr"),
        "ndcg": _avg(scoreable, "ndcg"),
        "coverage": _avg(scoreable, "keyword_coverage"),
    }

    return {"overall": overall, "per_category": per_category}


def summarize_answers(results: list[tuple[TestQuestion, AnswerEval]]) -> dict:
    """Aggregate answer-quality results, split by category.

    Unlike retrieval, LLM-as-judge answer eval is meaningful for
    "unanswerable" tests too (a correct refusal should score well against
    the reference "I don't have enough information..." answer), so the
    overall average includes every test. The per-category breakdown lets
    you check refusal accuracy specifically instead of it blending in.
    """
    by_category = defaultdict(list)
    for test, result in results:
        by_category[test.category].append(result)

    per_category = {
        category: {
            "n": len(res_list),
            "accuracy": _avg(res_list, "accuracy"),
            "completeness": _avg(res_list, "completeness"),
            "relevance": _avg(res_list, "relevance"),
        }
        for category, res_list in by_category.items()
    }

    all_results = [result for _test, result in results]
    overall = {
        "n": len(all_results),
        "accuracy": _avg(all_results, "accuracy"),
        "completeness": _avg(all_results, "completeness"),
        "relevance": _avg(all_results, "relevance"),
    }

    return {"overall": overall, "per_category": per_category}


def run_cli_evaluation(test_number: int):
    """Run both evaluations for a single test row and print a report."""
    tests = load_tests()

    if test_number < 0 or test_number >= len(tests):
        print(f"Error: test_row_number must be between 0 and {len(tests) - 1}")
        sys.exit(1)

    test = tests[test_number]

    print(f"\n{'=' * 80}")
    print(f"Test #{test_number}")
    print(f"{'=' * 80}")
    print(f"Question: {test.question}")
    print(f"Keywords: {test.keywords}")
    print(f"Category: {test.category}")
    print(f"Reference Answer: {test.reference_answer}")
    if test.category == "unanswerable":
        print(
            "\nNOTE: this is an 'unanswerable' test (no keywords by design)."
            " Expect retrieval MRR/nDCG/coverage to be ~0 below - that's"
            " expected. What matters here is whether the generated answer"
            " correctly declines instead of hallucinating."
        )

    print(f"\n{'=' * 80}")
    print("Retrieval Evaluation")
    print(f"{'=' * 80}")

    retrieval_result = evaluate_retrieval(test)

    print(f"MRR: {retrieval_result.mrr:.4f}")
    print(f"nDCG: {retrieval_result.ndcg:.4f}")
    print(
        f"Keywords Found: {retrieval_result.keywords_found}/{retrieval_result.total_keywords}"
    )
    print(f"Keyword Coverage: {retrieval_result.keyword_coverage:.1f}%")

    print(f"\n{'=' * 80}")
    print("Answer Evaluation")
    print(f"{'=' * 80}")

    answer_result, generated_answer, retrieved_docs = evaluate_answer(test)

    print(f"\nGenerated Answer:\n{generated_answer}")
    print(f"\nFeedback:\n{answer_result.feedback}")
    print("\nScores:")
    print(f"  Accuracy: {answer_result.accuracy:.2f}/5")
    print(f"  Completeness: {answer_result.completeness:.2f}/5")
    print(f"  Relevance: {answer_result.relevance:.2f}/5")
    print(f"\n{'=' * 80}\n")


def main():
    """CLI to evaluate a specific test by row number, or all tests if none given."""
    if len(sys.argv) == 2:
        try:
            test_number = int(sys.argv[1])
        except ValueError:
            print("Error: test_row_number must be an integer")
            sys.exit(1)
        run_cli_evaluation(test_number)
        return

    # No row number given -> run everything and print summary averages
    tests = load_tests()
    n = len(tests)

    print(f"Running retrieval evaluation over {n} tests...")
    retrieval_results = [
        (test, result) for test, result, _progress in evaluate_all_retrieval(tests)
    ]
    retrieval_summary = summarize_retrieval(retrieval_results)

    overall = retrieval_summary["overall"]
    print(f"\n  Overall (excludes 'unanswerable' - see note below):")
    print(f"    Avg MRR: {overall['mrr']:.4f}  (n={overall['n']})")
    print(f"    Avg nDCG: {overall['ndcg']:.4f}")
    print(f"    Avg Keyword Coverage: {overall['coverage']:.1f}%")

    print(f"\n  By category:")
    for category, stats in sorted(retrieval_summary["per_category"].items()):
        print(
            f"    {category:15s} n={stats['n']:>3}  "
            f"MRR={stats['mrr']:.4f}  nDCG={stats['ndcg']:.4f}  "
            f"Coverage={stats['coverage']:.1f}%"
        )
    if "unanswerable" in retrieval_summary["per_category"]:
        print(
            "\n  Note: 'unanswerable' tests have no keywords by design, so "
            "they always score 0 on retrieval metrics. That's expected -"
            " they test the refusal path (see answer eval below), not"
            " retrieval, and are excluded from the 'Overall' average above."
        )

    print(f"\nRunning answer evaluation over {n} tests...")
    answer_results = [
        (test, result) for test, result, _progress in evaluate_all_answers(tests)
    ]
    answer_summary = summarize_answers(answer_results)

    overall = answer_summary["overall"]
    print(f"\n  Overall (all {overall['n']} tests):")
    print(f"    Avg Accuracy: {overall['accuracy']:.2f}/5")
    print(f"    Avg Completeness: {overall['completeness']:.2f}/5")
    print(f"    Avg Relevance: {overall['relevance']:.2f}/5")

    print(f"\n  By category:")
    for category, stats in sorted(answer_summary["per_category"].items()):
        print(
            f"    {category:15s} n={stats['n']:>3}  "
            f"Accuracy={stats['accuracy']:.2f}/5  "
            f"Completeness={stats['completeness']:.2f}/5  "
            f"Relevance={stats['relevance']:.2f}/5"
        )
    if "unanswerable" in answer_summary["per_category"]:
        unanswerable_acc = answer_summary["per_category"]["unanswerable"]["accuracy"]
        print(
            f"\n  Note: 'unanswerable' Accuracy ({unanswerable_acc:.2f}/5) is"
            " your refusal-correctness signal - low scores here mean the"
            " model is hallucinating answers instead of saying it doesn't"
            " have enough information."
        )


if __name__ == "__main__":
    main()
