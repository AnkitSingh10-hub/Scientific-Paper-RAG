import sys
import math
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


def calculate_mrr(keyword: str, retrieved_docs: list[str]) -> float:
    """Calculate reciprocal rank for a single keyword (case-insensitive)."""
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.lower():
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    """Calculate Discounted Cumulative Gain."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)  # i+2 because rank starts at 1
    return dcg


def calculate_ndcg(keyword: str, retrieved_docs: list[str], k: int = 10) -> float:
    """Calculate nDCG for a single keyword (binary relevance, case-insensitive)."""
    keyword_lower = keyword.lower()

    relevances = [
        1 if keyword_lower in doc.lower() else 0 for doc in retrieved_docs[:k]
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
        k: Number of top chunks to retrieve (default 10)

    Returns:
        RetrievalEval object with MRR, nDCG, and keyword coverage metrics
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

    print(f"Running retrieval evaluation over {len(tests)} tests...")
    mrr_total = ndcg_total = coverage_total = 0.0
    for test, result, progress in evaluate_all_retrieval(tests):
        mrr_total += result.mrr
        ndcg_total += result.ndcg
        coverage_total += result.keyword_coverage
    n = len(tests)
    print(f"  Avg MRR: {mrr_total / n:.4f}")
    print(f"  Avg nDCG: {ndcg_total / n:.4f}")
    print(f"  Avg Keyword Coverage: {coverage_total / n:.1f}%")

    print(f"\nRunning answer evaluation over {len(tests)} tests...")
    accuracy_total = completeness_total = relevance_total = 0.0
    for test, result, progress in evaluate_all_answers(tests):
        accuracy_total += result.accuracy
        completeness_total += result.completeness
        relevance_total += result.relevance
    print(f"  Avg Accuracy: {accuracy_total / n:.2f}/5")
    print(f"  Avg Completeness: {completeness_total / n:.2f}/5")
    print(f"  Avg Relevance: {relevance_total / n:.2f}/5")


if __name__ == "__main__":
    main()
