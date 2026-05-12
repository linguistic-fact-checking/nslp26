"""
SEGMENTATION MODULE

- Extract articles content from JATS XML
- Segment paragraphs in all JSON files in the input directory, saving results to output directory

Usage:
    python run_segmentation.py --model gpt-5-mini
    python run_segmentation.py --model gemini-2-5-flash --parallel 20 --seg_batch_words 600
"""

import json, os, logging, argparse, textwrap, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from dotenv import load_dotenv

from src.jats_to_sections import jats_to_article_json
from src.pipeline_helpers import make_word_batches, call_llm

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

SEGMENTATION_PROMPT = """You are an expert assistant helping a linguist segment a scientific linguistics article into sentences.

**Task:** Split each numbered paragraph below into grammatically complete sentences. Preserve the original text exactly—do not paraphrase, correct, or reorder. Process each paragraph independently.

**Rules:**
1. **Complete sentences only:** Each output unit must be a grammatically complete sentence. Merge fragments (dangling clauses, isolated citations, orphan parentheticals) with the sentence they belong to.
2. **Linguistic examples:** Numbered examples (e.g., "(1)", "(2a)") must be attached to the sentence that introduces or discusses them *without* their glosses/translations. Keep only the original text, removing gloss lines, and translation.
   - Input: "Example (1) shows the problem in Spanish. (1) El niño duerme. 'The boy sleeps.'"
   - Output: ["The Example \\"El niño duerme.\\" shows the problem in Spanish: "]
   Linguistic examples that are not cited or attached to any sentence must be completely removed.
3. **In-text citations:** Keep citations attached to their sentence.
4. **Lists and enumerations:** If a sentence introduces a list, keep short list items with the introducing sentence. Split only if items are full sentences themselves.
5. **Section titles:** Titles/headings may remain as standalone fragments (the only exception to rule 1).
6. **Preserve numbering:** Only include numbers if the original text has them.

**Paragraphs to segment:**
{paragraph_blocks}

**Output format:** Return a JSON array of arrays — one inner array of sentence strings per paragraph, in the same order as the input paragraphs.
Example for 2 paragraphs: [["Sentence from P1.", "Another from P1."], ["Sentence from P2."]]
No markdown, no explanation—just valid JSON."""

# ========== FUNCTIONS ==========

# Extract articles content from JATS XML
def xml_to_json(input_data_directory):

    logger.info("--------------------------------")
    logger.info(f"START EXTRACTION OF XML FILES FROM {input_data_directory}")

    for filename in os.listdir(input_data_directory):

        xml_path = Path(os.path.join(input_data_directory, filename))
        if filename.endswith(".xml") and not xml_path.with_suffix(".json").exists():
            if not xml_path.exists():
                raise FileNotFoundError(f"XML file not found: {xml_path}")
            
            article_data = jats_to_article_json(str(xml_path), strip_urls=True)
            json_path = xml_path.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(article_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Extracted article from: {xml_path}")
            logger.info(f"Saved JSON to: {json_path}")
            logger.info(f"Article Summary:")
            logger.info(f"   Title: {article_data['metadata']['title'][:80]}...")
            logger.info(f"   Authors: {len(article_data['metadata']['authors'])}")
            logger.info(f"   Sections: {len(article_data['content'])}")

    logger.info("END EXTRACTION")

# --- SEGMENTATION ---

# Segment paragraphs in all JSON files in the input directory, saving results to output directory
def segment_all_jsons(input_data_directory, output_dir, seg_batch_words, model_id, parallel_calls):
    logger.info("--------------------------------")
    logger.info(f"START PARAGRAPH SEGMENTATION")

    for filename in os.listdir(input_data_directory):

        if filename.endswith(".json"):
            output_path = os.path.join(output_dir, filename)
            if os.path.exists(output_path):
                logger.info(f"Skipping already segmented file: {filename}")
                continue

            segment_one_json(input_data_directory, filename, output_dir, seg_batch_words, model_id, parallel_calls)

    logger.info("END PARAGRAPH SEGMENTATION")

# Segment one JSON file, saving results to output directory
def segment_one_json(input_data_directory, filename, output_dir, seg_batch_words, model_id, parallel_calls=8):

    logger.info("--------------------------------")
    logger.info(f"START SEGMENTATION OF {filename}")

    # ── Load article and build sections ──
    json_path = Path(os.path.join(input_data_directory, filename))
    with open(json_path, "r", encoding="utf-8") as f:
        article = json.load(f)
    title = article['metadata'].get('title', '')
    abstract = article['metadata'].get('abstract', '') 

    sections_to_process = []
    if title or abstract:
        title_abstract_section = {'section_name': 'Title and Abstract', 'paragraphs': []}
        if title:
            title_abstract_section['paragraphs'].append(title)
        if abstract:
            title_abstract_section['paragraphs'].append(abstract)
        sections_to_process.append(title_abstract_section)

    content_sections = article['content']
    sections_to_process.extend(content_sections)

    flat_tasks = []
    prev_paragraph = None
    for section in sections_to_process:
        section_name = section['section_name']
        for para_text in section['paragraphs']:
            if para_text.strip():
                task = {
                    'section': section_name,
                    'paragraph': para_text,
                    'sentences': [para_text],
                    'previous_paragraph': prev_paragraph
                }
                flat_tasks.append(task)
                prev_paragraph = para_text

    total_paragraphs = len(flat_tasks)

    # ── Create batches and clients ──
    seg_batches = make_word_batches(flat_tasks, seg_batch_words, lambda x: len(x['paragraph'].split()))
    logger.info(f"\n>>> {len(sections_to_process)} sections, {total_paragraphs} paragraphs -> {len(seg_batches)} batches (≤{seg_batch_words} words)")

    # ── Build batch index ranges ──
    batch_ranges = []
    idx = 0
    for batch in seg_batches:
        batch_ranges.append((list(range(idx, idx + len(batch))), batch))
        idx += len(batch)

    # ── Segment in parallel batches ──
    results = {}
    with ThreadPoolExecutor(max_workers=parallel_calls) as executor:
        futures = {}
        for indices, batch in batch_ranges:
            f = executor.submit(call_llm, batch, SEGMENTATION_PROMPT, model_id)
            futures[f] = indices

        for future in tqdm(as_completed(futures), total=len(futures), desc="Segmentation batches"):
            indices = futures[future]
            batch_results = future.result()
            for i, flat_idx in enumerate(indices):
                results[flat_idx] = batch_results[i]

    # ── Reconstruct and save results per model ──
    segmented_data = []
    result_idx = 0
    for section in sections_to_process:
        section_result = {'section': section['section_name'], 'paragraphs': []}
        for para_text in section['paragraphs']:
            if not para_text.strip():
                continue
            section_result['paragraphs'].append({
                    'original_paragraph': para_text,
                    'sentences': results[result_idx]
                })
            result_idx += 1
        segmented_data.append(section_result)
        
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segmented_data, f, indent=2, ensure_ascii=False)

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
        default="input_data",
        help="Path to input json files (default: input_data).",
    )
    parser.add_argument(
        "--output_dir",
        default="output_data/segmented_files",
        help="Directory for output files (default: output_data/segmented_files).",
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
        default=1000,
        help="Number of words per segmentation batch (default: 1000).",
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

    if not Path(args.output_dir).exists():
        os.makedirs(args.output_dir)

    xml_to_json(args.input)
    segment_all_jsons(args.input, args.output_dir, args.seg_batch_words, args.model, args.parallel)

if __name__ == "__main__":
    main()