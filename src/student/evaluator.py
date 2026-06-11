"""
Evaluation: implements recall@k with the 5% overlap rule.

For each question, a retrieved source counts as "found" if it overlaps
at least 5% with any ground-truth source (character-level overlap).
"""
import json
from typing import List


def _overlap_ratio(
    retrieved_start: int,
    retrieved_end: int,
    truth_start: int,
    truth_end: int,
) -> float:
    """
    Compute the character-level overlap between two ranges,
    expressed as a fraction of the TRUTH range length.

    Example:
        retrieved: [100, 300]
        truth:     [250, 400]
        overlap:   [250, 300] = 50 chars
        truth_len: 150 chars
        ratio: 50/150 = 0.33  ← > 5%, counts as "found"
    """
    overlap_start = max(retrieved_start, truth_start)
    overlap_end   = min(retrieved_end,   truth_end)
    overlap_len   = max(0, overlap_end - overlap_start)

    truth_len = truth_end - truth_start
    if truth_len == 0:
        return 0.0
    return overlap_len / truth_len


def _recall_for_question(
    retrieved_sources: List[dict],
    truth_sources: List[dict],
    min_overlap: float = 0.05,
) -> float:
    """
    Recall for a single question = found_sources / total_truth_sources.
    A truth source is "found" if at least one retrieved chunk overlaps
    it by min_overlap or more AND is in the same file.
    """
    if not truth_sources:
        return 1.0   # nothing to find → perfect score

    found = 0
    for truth in truth_sources:
        truth_file  = truth["file_path"]
        truth_start = truth["first_character_index"]
        truth_end   = truth["last_character_index"]

        for ret in retrieved_sources:
            if ret["file_path"] != truth_file:
                continue
            ratio = _overlap_ratio(
                ret["first_character_index"],
                ret["last_character_index"],
                truth_start,
                truth_end,
            )
            if ratio >= min_overlap:
                found += 1
                break   # this truth source is covered, move to the next

    return found / len(truth_sources)


def run_evaluation(
    student_answer_path: str,
    dataset_path: str,
    k: int = 10,
    max_context_length: int = 2000,
    min_overlap: float = 0.05,
) -> None:
    """Full evaluation: print recall@1/3/5/10."""
    with open(student_answer_path, encoding="utf-8") as f:
        student_data = json.load(f)
    with open(dataset_path, encoding="utf-8") as f:
        truth_data = json.load(f)

    # Build a lookup: question_id → truth sources
    truth_by_id = {}
    for q in truth_data.get("rag_questions", []):
        if "sources" in q:
            truth_by_id[q["question_id"]] = q["sources"]

    student_results = student_data.get("search_results", [])

    valid = [r for r in student_results if r["question_id"] in truth_by_id]
    print(f"Student data is valid: {len(valid) == len(student_results)}")
    print(f"Total number of questions: {len(student_results)}")
    print(f"Total number of questions with sources: {len(truth_by_id)}")
    print(f"Total number of questions with student sources: {len(valid)}")

    # Compute recall at various k values
    recall_at = {1: [], 3: [], 5: [], 10: []}

    for result in valid:
        qid = result["question_id"]
        retrieved = result["retrieved_sources"]
        truth = truth_by_id[qid]

        for cutoff in recall_at:
            score = _recall_for_question(
                retrieved[:cutoff], truth, min_overlap
            )
            recall_at[cutoff].append(score)

    print("\nEvaluation Results")
    print("=" * 40)
    print(f"Questions evaluated: {len(valid)}")
    for cutoff in [1, 3, 5, 10]:
        scores = recall_at[cutoff]
        mean = sum(scores) / len(scores) if scores else 0.0
        print(f"Recall@{cutoff}: {mean:.3f}")
