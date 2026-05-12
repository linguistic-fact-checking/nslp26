"""
CLASSIFICATION MODULE

- Load segmented JSON files
- Classify statements according to predefined categories

Usage:
    python run_classification.py --model gpt-5-mini
    python run_classification.py --model gemini-2-5-flash --parallel 20 --seg_batch_words 600
    python run_classification.py --model gpt-5-mini --merge_stm True --gold_path input_data/gold_statements.csv
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

CLASSIFICATION_PROMPT = """You are an expert linguist reviewer classifying statements from a linguistics article according to specific guidelines.

**Statements to classify:**
{paragraph_blocks}

**Categories:**

* **C (Citation/Attribution)**: Statements attributing an idea or data to another author or paper.
  - Example: "As noted by Chomsky (1995), the minimal link condition applies."
  - Ambiguity: If the statement is "according to" a paper or person, it is Type C. But if it is "according to" a rule or principle (even if attributed earlier), it is likely Type L (author asserting the rule's validity).
  - Scope: "Ginzburg & Sag (2000) proposed [X]" is C. But the continuation "an interrogative sluice is not derived..." is L (if asserted by the current authors).

* **L (Linguistic Statement)**: The authors themselves assert the claim. Sub-types:

  * **L-Spec (Linguistic - Specific)**: Empirical claims about a *specific* language.
    - MUST NOT make comparisons with other languages.
    - Verifiable on language data (corpora, treebanks).
    - **Language Field**: You MUST specify the language name for L-Spec statements.
    - Example: "Warlpiri is mostly SOV." (Language: Warlpiri)

  * **L-Typo (Linguistic - Typological)**: Cross-linguistic phenomena or comparisons.
    - Describes phenomena across languages ("All VSO languages have prepositions").
    - Compares a set of languages ("Sino-Tibetan languages tend to be more...").
    - Compares specific languages ("Polish is similar to Russian...").

  * **L-Theo (Linguistic - Theoretical)**: Theory-internal generalizations or claims.
    - Not directly verifiable with raw language data (treebanks, corpora).
    - Abstract concepts (e.g., "Move-α is blocked by locality constraints").
    - Evaluating theories ("This mismatch casts doubt on syntactic theories...").

* **S (Structural/Methodological)**: Statements that do not make a scientific claim.
  - Section titles ("3.2 Theoretical Background").
  - Definitions ("We define <term> as...").
  - Meta-discourse/Signposting ("Section X will review...", "The authors studied...", "This article").
  - Data information ("Our dataset contains 1200 sentences").
  - Methodology ("We use the Fisher test...", "We asked 12 native speakers...").
  - Technical/Algorithmic results ("It took 12 seconds", "20% were rejected").

**Important Guidelines for Ambiguity:**
1. If there is a citation in the statement -> Type C.
   (e.g., "Many non-P-stranding languages allow P-omission (Fortin 2007)..." -> Type C).
2. "According to [Rule/Principle]" -> Type L.
3. "According to [Person/Paper]" -> Type C.

**Instructions:**
For each statement, provide:
1. Primary `class` (L-Spec, L-Typo, L-Theo, C, S) and `confidence` (0-1).
2. If L-Spec, provide the `language` (e.g., "French", "Warlpiri"). Dictionary/standard name. If not L-Spec, use null.
3. `secondary_class` and `secondary_confidence`.
4. Keep the `statement_idx`.

**Output Format (JSON only, no extra text):**
```json
[
  {{
    "statement_idx": 1,
    "statement": "The original statement text...",
    "class": "L-Spec",
    "confidence": 0.95,
    "language": "French",
    "secondary_class": "C",
    "secondary_confidence": 0.05
  }},
  {{
    "statement_idx": 2,
    "statement": "We analyze the results...",
    "class": "S",
    "confidence": 0.99,
    "language": null,
    "secondary_class": "Other",
    "secondary_confidence": 0.01
  }}
]
```
"""

# ========== FUNCTIONS ==========

# For each unique gold paragraph, find matching original_paragraphs in statements
# Use first 100 chars to match (avoids footnote number mismatches)
# Remove matched statements and replace with the gold rows
def match_and_replace_statements(statements, gold):

    logger.info("--------------------------------")
    logger.info("ADDING GOLD STATEMENTS TO PREDICTIONS")

    gold_paragraphs = gold['paragraph'].unique()

    MATCH_PREFIX_LEN = 100
    rows_to_drop = []
    gold_rows_to_add = []
    paragraphs_found = 0
    paragraphs_not_found = 0

    for gold_par in gold_paragraphs:
        prefix = gold_par[:MATCH_PREFIX_LEN]
        # Match by article + paragraph prefix to avoid cross-article false matches
        gold_art = gold[gold['paragraph'] == gold_par]['article'].iloc[0]
        mask = (
            (statements['article'] == gold_art) &
            (statements['original_paragraph'].str[:MATCH_PREFIX_LEN] == prefix)
        )
        matched_indices = statements[mask].index.tolist()
        if matched_indices:
            paragraphs_found += 1
            rows_to_drop.extend(matched_indices)
            gold_rows_to_add.append(gold[gold['paragraph'] == gold_par])
        else:
            paragraphs_not_found += 1
            # No matching paragraph in statements — add gold rows anyway
            gold_rows_to_add.append(gold[gold['paragraph'] == gold_par])

    # Drop matched statements
    merged = statements.drop(index=rows_to_drop)
    merged['Gold Class'] = merged.get('Gold Class', pd.NA)
    # Concatenate gold rows
    if gold_rows_to_add:
        gold_replacement = pd.concat(gold_rows_to_add, ignore_index=True)
        merged = pd.concat([merged, gold_replacement], ignore_index=True)
    merged.rename(columns={'Gold Class': 'classification'}, inplace=True)
    logger.info(f"Paragraphs matched and replaced with gold: {paragraphs_found}")
    logger.info(f"Paragraphs not found for gold replacement: {paragraphs_not_found}")
    logger.info(f"COMPLETED ADDING GOLD STATEMENTS")
    return merged

def classify_all_statements(statements, output_dir, seg_batch_words, model_id, parallel_calls=8):

    logger.info("--------------------------------")
    logger.info("STARTING CLASSIFICATION OF STATEMENTS")

    # ── Build list of (idx, statement) tuples ──
    all_stmts = list(statements['statement'].items())  # [(row_idx, stmt_text), ...]

    # ── Create batches ──
    batches = [all_stmts[i:i + seg_batch_words] for i in range(0, len(all_stmts), seg_batch_words)]

    # ── Classify in parallel batches ──
    results_by_idx = {}

    with ThreadPoolExecutor(max_workers=parallel_calls) as executor:
        futures = {}
        for batch_idx, batch in enumerate(batches):
            f = executor.submit(call_llm, batch, CLASSIFICATION_PROMPT, model_id)
            futures[f] = (batch_idx, batch)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Classification batches"):
            batch_idx, batch = futures[future]
            result = future.result()
            if isinstance(result, list):
                result_map = {r.get('statement_idx'): r for r in result if isinstance(r, dict)}
                for i, (row_idx, stmt_text) in enumerate(batch):
                    fresh_idx = i + 1
                    if fresh_idx in result_map:
                        results_by_idx[row_idx] = result_map[fresh_idx]
                    else:
                        results_by_idx[row_idx] = {
                            'class': 'Other', 'confidence': 0.0,
                            'language': None,
                            'secondary_class': '', 'secondary_confidence': 0.0
                        }
            else:
                # Error fallback
                for row_idx, _ in batch:
                    results_by_idx[row_idx] = {
                        'class': 'Other', 'confidence': 0.0,
                        'language': None,
                        'secondary_class': '', 'secondary_confidence': 0.0
                    }

    # ── Merge results back into DataFrame ──
    statements['predicted_class'] = statements.index.map(lambda i: results_by_idx.get(i, {}).get('class', ''))
    statements['confidence'] = statements.index.map(lambda i: results_by_idx.get(i, {}).get('confidence', 0.0))
    statements['language'] = statements.index.map(lambda i: results_by_idx.get(i, {}).get('language', None))
    statements['secondary_class'] = statements.index.map(lambda i: results_by_idx.get(i, {}).get('secondary_class', ''))
    statements['secondary_confidence'] = statements.index.map(lambda i: results_by_idx.get(i, {}).get('secondary_confidence', 0.0))

    statements.to_csv(output_dir, index=False)
    logger.info(f"COMPLETED CLASSIFICATION OF STATEMENTS. Output saved to {output_dir}")

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
        default="output_data/statements.csv",
        help="Path to input statements CSV (default: output_data/statements.csv).",
    )
    parser.add_argument(
        "--output_dir",
        default='output_data/clasified_statements.csv',
        help="Directory for output CSV (default: output_data/clasified_statements.csv).",
    )
    parser.add_argument(
        "--merge_stm",
        default=False,
        help="If statements need to be merged with the gold statements provided.",
    )
    parser.add_argument(
        "--gold_path",
        default="input_data/gold_statements.csv",
        help="Path to gold statements.",
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

    if args.merge_stm == True:
        statements = pd.read_csv(args.input, header=0)
        directory = os.path.dirname(args.output_dir) 

        statements['original_paragraph'] = statements['paragraph'].apply(
            lambda x: ast.literal_eval(x)['original_paragraph']
        )
        gold = pd.read_csv(args.gold_path, header=0)
        merged = match_and_replace_statements(statements, gold)
        merged.to_csv(directory + '/merged_statements.csv', index=False)
    else:
        merged = pd.read_csv(args.input, header=0)
    
    classify_all_statements(merged, args.output_dir, args.seg_batch_words, args.model, args.parallel)

if __name__ == "__main__":
    main()