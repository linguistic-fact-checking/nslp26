#!/usr/bin/env python3
"""
Reads linguistic statements from a CSV, queries an LLM for veracity judgments
with parallel API calls, writes results to CSV, and prints evaluation metrics.

Usage:
    python run_veracity.py --model gpt-5-mini
    python run_veracity.py --model gemini-2-5-flash --parallel 20
    python run_veracity.py --model accounts/fireworks/models/deepseek-v3p1 --input path/to/input.csv

Supported models:
    gpt-5-nano, gpt-5-mini, gpt-5-2                          (OpenAI)
    gemini-3-flash-preview, gemini-2-5-flash                  (Google)
    accounts/fireworks/models/deepseek-v3p1                   (Fireworks)
    accounts/fireworks/models/kimi-k2p5                       (Fireworks)
    accounts/fireworks/models/kimi-k2-instruct-0905           (Fireworks)
"""

from __future__ import annotations

import argparse, json, sys, textwrap, logging, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm.auto import tqdm

from src.pipeline_helpers import call_llm

load_dotenv()

# ========== LOGGING SETUP ==========
LOG_FILE = "logs.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
    force=True,
)
logger = logging.getLogger("logs")

# ========== PROMPT ==========

VERACITY_PROMPT = textwrap.dedent("""\
Role: Linguistic Fact-Checker (Syntax/Morphosyntax).
Task: Evaluate the accuracy of STATEMENT based on descriptive academic linguistics.

Return only JSON:
{{"veredict": true or false,
 "confidence": 0.0-1.0,
 "rationale": "concise reason"
}}
RULES:
- No prescriptive grammar; use descriptive evidence.
- If the statement is a single word or lacks propositional content, veredict is false.
- Lower confidence for theoretical disagreements.
- Return exactly one JSON object per statement, in order.

STATEMENT:
{statement}
""")

# ========== FUNCTIONS ==========

# JSON parsing
def parse_json(text: str) -> dict:
    """Best-effort parse of an LLM JSON response."""
    if not text:
        return {"parse_error": "empty"}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").replace("json", "", 1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"raw": text, "parse_error": "json decode failed"}

# Single-row veracity judgment
def judge_veracity_row(row: pd.Series, model: str) -> dict:
    """Call the LLM and return a dict with the parsed result + raw response."""
    raw = call_llm(row["statement"], VERACITY_PROMPT, model)
    parsed = parse_json(raw)
    parsed["raw_response"] = raw
    return {**row.to_dict(), **parsed}

# Parallel runner
def run_veracity(
    challenge: pd.DataFrame,
    model: str,
    parallel_calls: int = 10,
) -> pd.DataFrame:
    """Run veracity checks with *parallel_calls* concurrent API requests."""
    results: list[dict] = [None] * len(challenge)  # preserve order

    with ThreadPoolExecutor(max_workers=parallel_calls) as pool:
        future_to_idx = {
            pool.submit(judge_veracity_row, row, model): idx
            for idx, (_, row) in enumerate(challenge.iterrows())
        }
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx),
            desc=f"LLM veracity ({model})",
        ):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                # keep the original row data + error info so nothing is lost
                orig = challenge.iloc[idx].to_dict()
                orig["parse_error"] = str(exc)
                results[idx] = orig

    return pd.DataFrame(results)


# Evaluation helpers
def evaluation_verdict(df_eval: pd.DataFrame) -> None:
    """Print overall accuracy / precision / recall / F1."""
    verdict_col = "veredict" if "veredict" in df_eval.columns else "verdict"
    y_true = df_eval["Gold Verification"]
    y_pred = df_eval[verdict_col]

    print("\nEvaluation Results")
    print("-------------------")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_true, y_pred)}")

    errors = df_eval[df_eval[verdict_col] != df_eval["Gold Verification"]]
    print(f"\nNumber of errors: {len(errors)}")


def evaluation_verdict_by_gold_class(df_eval: pd.DataFrame) -> None:
    """Print per-Gold-Class metrics."""
    verdict_col = "veredict" if "veredict" in df_eval.columns else "verdict"
    rows = []
    for gold_class, group in df_eval.groupby("Gold Class"):
        y_true = group["Gold Verification"]
        y_pred = group[verdict_col]
        rows.append({
            "Gold Class": gold_class,
            "Samples": len(group),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
        })
    results_df = pd.DataFrame(rows).sort_values("Gold Class")
    print("\nEvaluation by Gold Class")
    print("-" * 40)
    print(results_df.to_string(index=False))

# Output path helper
def make_output_path(model: str, output_dir: str) -> Path:
    """Build a filesystem-safe output CSV path from the model name."""
    safe_name = (
        model
        .replace("/", "-")
        .replace(" ", "_")
        .replace("accounts-fireworks-models-", "")
    )
    return Path(output_dir) / f"{safe_name}.csv"

# CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM-based linguistic veracity checking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f"""\
            Supported models:
              {chr(10).join('  ' + m for m in os.getenv("VALID_MODELS"))}
        """),
    )
    parser.add_argument(
        "--model", required=True, help="Model identifier (see list above)."
    )
    parser.add_argument(
        "--input",
        default="output_data/linguistic_statements.csv",
        help="Path to input CSV (default: output_data/linguistic_statements.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default="output_data/experiments_fact_check",
        help="Directory for output CSV (default: output_data/experiments_fact_check).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=10,
        help="Number of parallel API calls (default: 10).",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip printing evaluation metrics after completion.",
    )
    return parser.parse_args(argv)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)

    # Validate model
    if args.model not in os.getenv("VALID_MODELS"):
        print(
            f"Warning: '{args.model}' is not in the predefined model list. "
            "Proceeding anyway.",
            file=sys.stderr,
        )

    logger.info("--------------------------------")
    logger.info("STARTING VERACITY CHECK")
    logger.info(f"Model   : {args.model}")

    # Load input
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")
    challenge_df = pd.read_csv(input_path, header=0)
    logger.info(f"Loaded {len(challenge_df)} statements from {input_path}")

    # Run
    judged = run_veracity(challenge_df, args.model, args.parallel)

    # Save
    output_path = make_output_path(args.model, args.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    judged.to_csv(output_path, index=False)

    # Evaluate (if Gold Verification column is present and verdict was parsed)
    verdict_col = "veredict" if "veredict" in judged.columns else "verdict"
    has_gold = "Gold Verification" in judged.columns
    has_verdict = verdict_col in judged.columns

    if not args.no_eval and has_gold and has_verdict:
        eval_df = judged.dropna(subset=["Gold Verification", verdict_col]).copy()
        eval_df["Gold Verification"] = eval_df["Gold Verification"].astype(bool)
        eval_df[verdict_col] = eval_df[verdict_col].astype(bool)

        if len(eval_df) > 0:
            evaluation_verdict(eval_df)
            if "Gold Class" in eval_df.columns:
                evaluation_verdict_by_gold_class(eval_df)
        else:
            logger.info("No rows with both Gold Verification and verdict – skipping eval.")
    elif not args.no_eval:
        logger.info("Skipping evaluation (missing Gold Verification or verdict column).")


if __name__ == "__main__":
    main()
