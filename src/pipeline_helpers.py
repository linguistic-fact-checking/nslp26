"""
Helper functions for the pipeline.

Prompts are passed as parameters.
"""

import json
import time
import os
from dotenv import load_dotenv
from google import genai
import openai as _openai

load_dotenv()

# ── JSON / batch helpers ─────────────────────────────────────

def parse_json_text(text):
    """Strip markdown fences and parse JSON."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:].strip()
    if text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return json.loads(text)


parse_json_response = parse_json_text  # alias used in some cells


def make_word_batches(items, max_words, word_fn):
    """Group items into batches not exceeding max_words each."""
    batches = []
    current_batch, current_words = [], 0
    for item in items:
        w = word_fn(item)
        if current_batch and current_words + w > max_words:
            batches.append(current_batch)
            current_batch, current_words = [], 0
        current_batch.append(item)
        current_words += w
    if current_batch:
        batches.append(current_batch)
    return batches


def validate_seg_batch(result, batch):
    """Ensure segmentation result is array-of-arrays matching batch size."""
    n = len(batch)
    if not isinstance(result, list):
        return [[pt] for _, pt in batch]
    if n == 1 and result and all(isinstance(x, str) for x in result):
        return [result]
    if len(result) == n and all(isinstance(x, list) for x in result):
        return result
    if all(isinstance(x, list) for x in result):
        padded = list(result[:n])
        while len(padded) < n:
            padded.append([batch[len(padded)][1]])
        return padded
    return [[pt] for _, pt in batch]


def format_extraction_blocks(batch_tasks):
    """Format a batch of paragraph tasks into numbered blocks."""
    blocks = []
    for i, task in enumerate(batch_tasks):
        block = f"=== Block {i+1} ==="
        # Handle non-dict items (strings, etc.)
        if not isinstance(task, dict):
            block += f"\n{task}"
            blocks.append(block)
            continue
        if task.get('previous_paragraph'):
            block += f"\n**Previous Paragraph (for resolving references):**\n{task['previous_paragraph']}"
        block += f"\n**Paragraph:**\n{task['paragraph']}"
        sentences_text = "\n".join(f"{j+1}. {s}" for j, s in enumerate(task['sentences']))
        block += f"\n**Sentences:**\n{sentences_text}"
        blocks.append(block)
    return "\n\n".join(blocks)


def validate_extraction_batch(result, n):
    """Ensure extraction result is a list of n extraction results."""
    if not isinstance(result, list):
        return [{"error": "Non-list response"}] * n
    if n == 1 and result and isinstance(result[0], dict):
        return [result]
    if len(result) == n:
        return result
    padded = list(result[:n])
    while len(padded) < n:
        padded.append({"error": "Missing from batch response"})
    return padded


# ── LLM call ───────────────────────────────

# Provider detection

def detect_provider(model: str, VALID_MODELS: list[str]) -> str:
    """Infer API provider from model name."""
    if model.startswith("gpt-"):
        return "openai"
    if model.startswith("gemini"):
        return "google"
    if "fireworks" in model:
        return "fireworks"
    raise ValueError(
        f"Cannot detect provider for model '{model}'. "
        f"Supported models: {VALID_MODELS}"
    )

# API callers  (each creates its own client – safe for threads)

def call_openai(prompt: str, model: str) -> str:

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set (check .env)")
    client = _openai.OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    result = parse_json_text(resp.choices[0].message.content)
    return result


def call_fireworks(prompt: str, model: str) -> str:

    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY not set (check .env)")
    client = _openai.OpenAI(
        api_key=api_key,
        base_url="https://api.fireworks.ai/inference/v1",
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    result = parse_json_text(resp.choices[0].message.content)
    return result


def call_gemini(prompt: str, model: str) -> str:
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set (check .env)")
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(model=model, contents=prompt)
    result = parse_json_text(resp.text)
    return result


_CALLERS = {
    "openai": call_openai,
    "google": call_gemini,
    "fireworks": call_fireworks,
}

def call_llm(batch, prompt: str, model: str) -> str:
    VALID_MODELS = os.getenv("VALID_MODELS")
    if not VALID_MODELS:
        raise RuntimeError("VALID_MODELS not set (check .env)")
    provider = detect_provider(model, VALID_MODELS)
    caller = _CALLERS.get(provider)
    if caller is None:
        raise ValueError(f"Unknown provider '{provider}'")
    filled = prompt.format(paragraph_blocks=batch)
    response = caller(filled, model)
    return response


# ── Statement counting / flattening / reconstruction ──

def count_statements(data):
    """Count total statements in extraction data."""
    total = 0
    for entry in data:
        ext = entry.get('extraction', {})
        if isinstance(ext, list):
            for sent in ext:
                total += len(sent.get('statements', []))
    return total


def flatten_statements(extracted_data):
    """Flatten all statements from extraction data with identity tracking."""
    flat = []
    for entry_idx, entry in enumerate(extracted_data):
        extraction = entry.get('extraction', {})
        if isinstance(extraction, list):
            for sent_pos, sent_data in enumerate(extraction):
                sentence_idx = sent_data.get('sentence_idx', 0)
                sentence_text = sent_data.get('sentence_text', '')
                for local_i, s in enumerate(sent_data.get('statements', [])):
                    flat.append({
                        'entry_idx': entry_idx,
                        'sent_pos': sent_pos,
                        'local_i': local_i,
                        'original_stmt_idx': s.get('statement_idx', local_i + 1),
                        'stmt_text': s.get('statement', ''),
                        'sentence_idx': sentence_idx,
                        'sentence_text': sentence_text,
                    })
    return flat


def reconstruct_from_flat(extracted_data, flat_stmts, classified_stmts):
    """Rebuild classification JSON from flat classified statements."""
    lookup = {}
    for flat_info, cls_result in zip(flat_stmts, classified_stmts):
        key = (flat_info['entry_idx'], flat_info['sent_pos'], flat_info['local_i'])
        lookup[key] = cls_result

    classified_results = []
    for entry_idx, entry in enumerate(extracted_data):
        extraction = entry.get('extraction', {})
        classified_entry = {
            'section': entry['section'],
            'paragraph': entry['paragraph'],
            'sentences': entry['sentences'],
            'classification': []
        }
        if isinstance(extraction, list):
            for sent_pos, sent_data in enumerate(extraction):
                sentence_idx = sent_data.get('sentence_idx', 0)
                sentence_text = sent_data.get('sentence_text', '')
                stmts = []
                for local_i in range(len(sent_data.get('statements', []))):
                    cls = lookup.get((entry_idx, sent_pos, local_i))
                    if cls:
                        stmts.append(cls)
                if stmts:
                    classified_entry['classification'].append({
                        'sentence_idx': sentence_idx,
                        'sentence_text': sentence_text,
                        'statements': stmts
                    })
        elif isinstance(extraction, dict) and 'error' in extraction:
            classified_entry['classification'] = {'error': extraction['error']}
        classified_results.append(classified_entry)
    return classified_results

