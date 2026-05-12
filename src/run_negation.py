"""
NEGATION MODULE

- 

Usage:
    python run_classification.py --model gpt-5-mini
    python run_classification.py --model gemini-2-5-flash --parallel 20 --seg_batch_words 600
"""

import os, logging, argparse, textwrap, sys, ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from dotenv import load_dotenv
import pandas as pd

from src.pipeline_helpers import call_llm

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

NEGATION_PROMPT = """You are a researcher writing a paper. For each statement, write a statement that is a STRICT logical contradiction of the claim below: both statements cannot be true at the same time.

**Rules**: For each statement,
    - Keep the wording and structure as close as possible. 
    - The change must reverse the truth conditions of the claim (not just express a different opinion). 
    - Do not use "no", "not", or explicit negation. 
    - Keep about the same length. Return a JSON object with a single key "contradiction". 

Statements to contradict: {paragraph_blocks}

**Output Format (JSON only, no extra text):**
    ```json
    [
    {{"contradiction": "The original statement contradicting...",
    }}
    ]
    ```
"""

# ========== FUNCTIONS ==========

def run_negation(statements, seg_batch_words, model_id, parallel_calls=8):

    logger.info("--------------------------------")
    logger.info("STARTING NEGATION GENERATION FOR ALL STATEMENTS")

    # ── Build list of (idx, statement) tuples ──
    all_stmts = list(statements['statement'].items())  # [(row_idx, stmt_text), ...]

    # ── Create batches ──
    batches = [all_stmts[i:i + seg_batch_words] for i in range(0, len(all_stmts), seg_batch_words)]

    # ── Classify in parallel batches ──
    results_by_idx = {}

    with ThreadPoolExecutor(max_workers=parallel_calls) as executor:
        futures = {}
        for batch_idx, batch in enumerate(batches):
            f = executor.submit(call_llm, batch, NEGATION_PROMPT, model_id)
            futures[f] = (batch_idx, batch)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Negation batches"):
            batch_idx, batch = futures[future]
            result = future.result()
            # Add contradictions as new rows to statements dataframe
            new_rows = []

            if isinstance(result, list):
                # Result list is already in same order as batch, so match by index
                for i, (row_idx, stmt_text) in enumerate(batch):
                    if i < len(result):
                        result_item = result[i]
                        
                        if isinstance(result_item, dict) and result_item.get('contradiction'):
                            # Get original row data using .loc (index label) instead of .iloc (position)
                            original_row = statements.loc[row_idx].to_dict()
                            
                            # Create new row with contradiction as statement
                            new_row = original_row.copy()
                            new_row['statement'] = result_item['contradiction']
                            new_rows.append(new_row)

    logger.info(f"END NEGATION GENERATION")

    if new_rows:
        new_statements_df = pd.DataFrame(new_rows)
        return new_statements_df
    else:
        return pd.DataFrame()  

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
        default="output_data/clasified_statements.csv",
        help="Path to input statements CSV (default: output_data/clasified_statements.csv).",
    )
    parser.add_argument(
        "--output_dir",
        default='output_data/negated_statements.csv',
        help="Directory for output CSV (default: output_data/negated_statements.csv).",
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
        default=20,
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

    statements = pd.read_csv(args.input, header=0)
    logger.info(f"Original dataframe shape: {statements.shape}")
    new_statements_df =run_negation(statements, args.seg_batch_words, args.model, args.parallel)
    statements = pd.concat([statements, new_statements_df], ignore_index=True)
    logger.info(f"New dataframe shape: {statements.shape}")
    
    statements.to_csv(args.output_dir, index=False)
    
if __name__ == "__main__":
    main()