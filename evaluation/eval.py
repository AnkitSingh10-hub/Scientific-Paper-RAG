import sys
import math
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from evaluation.test import TestQuestion, load_tests
from implementation.answer import answer_question, fetch_context
from openai import OpenAI
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv(override=True)

AZURE_ENDPOINT = (
    "https://ankitsinghtheweeknd691-9348-reso.services.ai.azure.com/openai/v1"
)

MODEL = "DeepSeek-V3.2"
db_name = "vector_database"

# Increased timeout slightly for parallel processing overhead
open_ai = OpenAI(
    base_url=AZURE_ENDPOINT,
    api_key=os.getenv("AZURE_FOUNDRY_API_KEY"),
    default_query={"api-version": "preview"},
    timeout=60,
    max_retries=3,
)


class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance."""

    mrr: float = Field(description="Mean Reciprocal Rank")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain")
    keywords_found: int = Field(description="Number of keywords found")
    total_keywords: int = Field(description="Total number of keywords")
    keyword_coverage: float = Field(description="Percentage of keywords found")


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

    feedback: str = Field(description="Concise feedback")
    accuracy: float = Field(description="1 to 5 scale")
    completeness: float = Field(description="1 to 5 scale")
    relevance: float = Field(description="1 to 5 scale")


# --- Math Helper Functions (Unchanged) ---


def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)
    return dcg


def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 10) -> float:
    keyword_lower = keyword.lower()
    relevances = [
        1 if keyword_lower in doc.page_content.lower() else 0
        for doc in retrieved_docs[:k]
    ]
    dcg = calculate_dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


# --- core Logic Functions ---


def evaluate_retrieval(test: TestQuestion, k: int = 10) -> RetrievalEval:
    retrieved_docs = fetch_context(test.question)
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


def evaluate_answer(test: TestQuestion) -> tuple[AnswerEval, str, list]:
    # 1. RAG Step
    generated_answer, retrieved_docs = answer_question(test.question)

    # 2. Judge Step
    judge_messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator. Evaluate accuracy, completeness, and relevance vs the reference. Only 5/5 for perfect answers.",
        },
        {
            "role": "user",
            "content": f"Question: {test.question}\nGen Answer: {generated_answer}\nRef Answer: {test.reference_answer}\nRate 1-5. If wrong, Accuracy=1.",
        },
    ]

    judge_response = open_ai.beta.chat.completions.parse(
        model=MODEL,
        messages=judge_messages,
        response_format=AnswerEval,
    )
    return judge_response.choices[0].message.parsed, generated_answer, retrieved_docs


# --- Parallel Generator Functions (This is what makes Evaluator.py fast) ---


def evaluate_all_retrieval():
    """Evaluate all retrieval tests in parallel."""
    tests = load_tests()
    total_tests = len(tests)

    # max_workers=10 runs 10 retrievals at once
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_test = {
            executor.submit(evaluate_retrieval, test): test for test in tests
        }

        for index, future in enumerate(as_completed(future_to_test)):
            test = future_to_test[future]
            result = future.result()
            progress = (index + 1) / total_tests
            yield test, result, progress


def evaluate_all_answers():
    """Evaluate all answers using parallel threads."""
    tests = load_tests()
    total_tests = len(tests)

    # We use a lower worker count (5) for answers because it involves two LLM calls
    # (Generation + Judging) and we don't want to hit Rate Limits too hard.
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_test = {
            executor.submit(evaluate_answer, test): test for test in tests
        }

        for index, future in enumerate(as_completed(future_to_test)):
            test = future_to_test[future]
            try:
                result, _, _ = future.result()
                progress = (index + 1) / total_tests
                yield test, result, progress
            except Exception as e:
                print(f"Error evaluating test: {e}")


# --- CLI Helpers (Unchanged) ---


def run_cli_evaluation(test_number: int):
    tests = load_tests()
    if test_number < 0 or test_number >= len(tests):
        sys.exit(1)
    test = tests[test_number]

    ret_res = evaluate_retrieval(test)
    ans_res, gen_ans, _ = evaluate_answer(test)

    print(
        f"\nQuestion: {test.question}\nGen Answer: {gen_ans}\nAccuracy: {ans_res.accuracy}/5"
    )


def main():
    if len(sys.argv) != 2:
        sys.exit(1)
    try:
        run_cli_evaluation(int(sys.argv[1]))
    except ValueError:
        sys.exit(1)


if __name__ == "__main__":
    main()
