"""
EXTRACTION MODULE

- Extract statements from segmented JSON files using LLM, saving results to CSV in output directory

Usage:
    python run_extraction.py --model gpt-5-mini
    python run_extraction.py --model gemini-2-5-flash --parallel 20 --seg_batch_words 600
    python run_extraction.py --model accounts/fireworks/models/deepseek-v3p1 --input path/to/input.csv
    python run_extraction.py --model gpt-5-2 --input path/to/input.csv --output-dir path/to/output.csv
"""

import json, os, logging, argparse, textwrap, sys
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from dotenv import load_dotenv

from src.pipeline_helpers import make_word_batches, call_llm, format_extraction_blocks, validate_extraction_batch

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

EXTRACTION_PROMPT = """You are an expert linguist extracting verifiable atomic statements from a linguistics article.

Process each paragraph block below independently.

**Core rules — each extracted statement must:**
1. Be atomic: split coordinations ("A and B") into separate statements when each conjunct is independently meaningful. When atomicity is ambiguous, do not split. Exception: joint reference lists stay together (e.g., "Smith (2020) and Miller (2023) show that…").
2. Be a factual assertion that can be independently understood and assessed.
3. Be directly and literally inferable from the source text — no chained inferences (from "A → B" and "B → C", extract only those two, never "A → C").
4. Be self-contained: resolve all pronouns/references (e.g., "it" → "the subject NP") and expand abbreviations (e.g., "NP" → "noun phrase (NP)").
5. Preserve technical precision. Always include the language name when the context makes it clear (e.g., "The verb agrees…" → "In Spanish, the verb agrees…").
6. Fold linguistic examples into the statement text; remove example numbers, glosses, and translations, keeping only the original-language form.

**Implications:**
7. Pure conditional ("if A then B"): extract only the whole implication as one statement; do not extract A or B separately.
8. Causal / inferential ("A. Hence B"): extract (a) A as a standalone statement, and (b) the implication "A implies B". Do NOT extract B alone.
   - Source: "Prepositions act as cues to retrieve the PP correlate. Hence, P-omission in sluices comes with a processing cost."
   - → (a) "Prepositions act as cues to retrieve the PP correlate." (b) "Prepositions acting as cues to retrieve the PP correlate implies that P-omission in sluices comes with a processing cost."

**Citations & attribution:**
9. If the author attributes a claim to another work, keep the attribution. If the author also endorses the claim, additionally extract the general (non-attributed) version (e.g., "X (2024) uses Y to prove Z" → also "Y can be used to prove Z").
10. Ambiguous citation scope: assume smallest scope (only the immediately preceding clause). Text after the citation is the current author's claim unless marked otherwise.
11. Quoted cited text: preserve verbatim; do not split further.
12. Author's own claims: never prefix with "the author/paper argues/states…".
13. "X followed Y in their analysis of Z" → (a) "Y analyses Z" and (b) "X uses the same methodology as Y for Z".

**Relative clauses:**
14. Non-restrictive "which" clauses yield an additional statement (e.g., "French, which is Indo-European, …" → also "French is an Indo-European language."). If restrictiveness is uncertain, do not split.

{paragraph_blocks}

**Output (JSON only, no extra text):**
Return a JSON array of arrays — one inner array per block, in order. Each inner array contains sentence objects:
```json
[[{{"sentence_idx": 1, "sentence_text": "...", "statements": [{{"statement_idx": 1, "statement": "..."}}]}}]]
```
Final check: every statement must be atomic, self-contained, free of unresolved references, free of author self-attribution, and free of example numbers/glosses/translations."""

# ========== FUNCTIONS ==========

def run_extraction(input, output_dir, seg_batch_words, model_id, parallel_calls):
    logger.info("--------------------------------")
    logger.info("STARTING STATEMENT EXTRACTION FOR ALL ARTICLES")

    final_extraction = pd.DataFrame()
    for filename in os.listdir(input):
        if filename.endswith(".json"):
            predictions = extract_statements_from_json(input, filename, seg_batch_words, model_id, parallel_calls)
            final_extraction = pd.concat([final_extraction, predictions], ignore_index=True)

    final_extraction.to_csv(output_dir, index=False)
    logger.info("COMPLETED STATEMENT EXTRACTION FOR ALL ARTICLES")

# Extract statements from segmented JSON file, saving results to output directory as CSV
def extract_statements_from_json(data_directory, filename, seg_batch_words, model_id, parallel_calls=8):

    logger.info("--------------------------------")
    logger.info(f"START EXTRACTION OF {filename}")

    # ── Load article and build sections ──
    json_path = Path(os.path.join(data_directory, filename))
    with open(json_path, "r", encoding="utf-8") as f:
        segmented_data = json.load(f)
    ext_tasks = []
    task_keys = []
    for sec in segmented_data:
        prev_para = None
        for pd_ in sec['paragraphs']:
                ext_tasks.append({
                    'section': sec['section'],
                    'paragraph': pd_['original_paragraph'],
                    'sentences': pd_['sentences'],
                    'prev_paragraph': prev_para
                })
                prev_para = pd_['original_paragraph']
                task_keys.append((filename, pd_))

    # ── Create word-based batches ──
    ext_batches = make_word_batches(ext_tasks, seg_batch_words, lambda t: len(t['paragraph'].split()))

    #  ── Build batch index ranges ──
    ext_batch_ranges = []
    idx = 0
    for batch in ext_batches:
        ext_batch_ranges.append((list(range(idx, idx + len(batch))), batch))
        idx += len(batch)

    # ── Run extraction in parallel ──
    results = [None] * len(ext_tasks)

    with ThreadPoolExecutor(max_workers=parallel_calls) as executor:
        futures = {}
        for indices, batch in ext_batch_ranges:
            f = executor.submit(call_llm, format_extraction_blocks(batch), EXTRACTION_PROMPT, model_id)
            futures[f] = indices

        for future in tqdm(as_completed(futures), total=len(futures), desc="Extraction batches"):
            indices = futures[future]
            batch_results = future.result()
            validated_batch_results = validate_extraction_batch(batch_results, len(indices))

            for i, flat_idx in enumerate(indices):
                results[flat_idx] = validated_batch_results[i]
            
    # ── Build a new CSV with predicted statements from extraction results ──
    rows = []
    for task_idx, (article, paragraph) in enumerate(task_keys):
        section = ext_tasks[task_idx].get('section', '')
        extraction = results[task_idx]
        if not isinstance(extraction, list):
            extraction = [extraction] if isinstance(extraction, dict) else []
        for sent_obj in extraction:
            if not isinstance(sent_obj, dict):
                continue
            sent_text = sent_obj.get('sentence_text', '')
            sent_idx = sent_obj.get('sentence_idx', '')
            for st in sent_obj.get('statements', []):
                stmt = st['statement'] if isinstance(st, dict) else str(st)
                stmt_idx = st.get('statement_idx', '') if isinstance(st, dict) else ''
                rows.append({
                    'article': article.replace('.json', ''),
                    'section': section,
                    'paragraph': paragraph,
                    'sentence_idx': sent_idx,
                    'sentence_text': sent_text,
                    'statement_idx': stmt_idx,
                    'statement': stmt,
                })

    df_pred = pd.DataFrame(rows)
    return df_pred

# CLI

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run extraction of linguistic statements from segmented JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f"""\
            Supported models:
              {chr(10).join('  ' + m for m in os.getenv("VALID_MODELS").split(','))}
        """),
    )
    parser.add_argument(
        "--model", required=True, help="Model identifier (see list above)."
    )
    parser.add_argument(
        "--input",
        default="output_data/segmented_files",
        help="Path to input json files (default: output_data/segmented_files).",
    )
    parser.add_argument(
        "--output_dir",
        default="output_data/statements.csv",
        help="Directory for output CSV (default: output_data/statements.csv).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=8,
        help="Number of parallel API calls (default: 8).",
    )
    parser.add_argument(
        "--seg_batch_words",
        type=int,
        default=600,
        help="Number of words per segmentation batch (default: 600).",
    )

    return parser.parse_args(argv)

# ========== MAIN ==========

# Extract statements from all segmented JSON files in the input directory, saving results to output directory as CSV
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
    run_extraction(args.input, args.output_dir, args.seg_batch_words, args.model, args.parallel)

if __name__ == "__main__":
    main()