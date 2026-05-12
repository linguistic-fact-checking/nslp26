"""
Convert JATS XML to sections with plain text paragraphs.
Uses the `jats` package (pip install jats) for metadata, and custom XML
parsing for body content to handle linguistic examples (gloss lists).
"""

from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import json
import re
import sys
from jats.parser import parse_jats_xml
from lxml import etree


def strip_markdown_links(text: str) -> str:
    """
    Remove markdown-style links, keeping only the link text.
    
    Example:
        "[Merchant 2001](https://doi.org/...)" → "Merchant 2001"
    """
    # Pattern: [text](url) → text
    return re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)


@dataclass
class Paragraph:
    """A single paragraph of text."""
    text: str
    item_type: str = "paragraph"  # paragraph, figure, table, etc.


@dataclass
class Section:
    """A section with a title and paragraphs."""
    title: str
    paragraphs: List[Paragraph]


def _extract_gloss_text(gloss_list: etree._Element) -> str:
    """
    Extract plain text from a <list list-type="gloss"> element.
    
    Handles the nested structure used for linguistic examples in Glossa-style
    JATS XML, including:
    - Simple examples: (1) a. I don't remember *(to) who she was talking.
    - Interlinear glosses with word-by-word alignment
    - Translations in quotes
    
    Args:
        gloss_list: An lxml element for <list list-type="gloss">.
        
    Returns:
        A plain text representation of the linguistic example.
    """
    parts = []
    
    # The top-level list-items in a gloss list:
    # - First list-item: contains example number (e.g. "(1)") and optional sub-label ("a.")
    # - Second list-item: contains the actual example sentence(s)/glosses
    top_items = gloss_list.findall('list-item')
    
    # Extract example number and sub-label from the first list-item
    number_label = ""
    if top_items:
        first_item = top_items[0]
        wordfirst_lists = first_item.findall('list[@list-type="wordfirst"]')
        label_parts = []
        for wf in wordfirst_lists:
            for li in wf.findall('list-item'):
                p = li.find('p')
                if p is not None:
                    text = ''.join(p.itertext()).strip()
                    # Skip non-breaking spaces used as placeholders
                    if text and text != '\xa0' and text != '\u00a0':
                        label_parts.append(text)
        number_label = " ".join(label_parts)
    
    if number_label:
        parts.append(number_label)
    
    # Extract the example content from the second list-item
    if len(top_items) > 1:
        second_item = top_items[1]
        sentence_gloss = second_item.find('list[@list-type="sentence-gloss"]')
        if sentence_gloss is not None:
            example_parts = _extract_sentence_gloss(sentence_gloss)
            if example_parts:
                parts.append(example_parts)
    
    return " ".join(parts)


def _extract_sentence_gloss(sg_elem: etree._Element) -> str:
    """
    Extract text from a <list list-type="sentence-gloss"> element.
    
    This may contain:
    - <list list-type="final-sentence"> with the example sentence or translation
    - <list list-type="word"> elements for interlinear glosses
    """
    result_parts = []
    
    for item in sg_elem.findall('list-item'):
        # Check for final-sentence (simple example or translation)
        final_sentences = item.findall('list[@list-type="final-sentence"]')
        for fs in final_sentences:
            for li in fs.findall('list-item'):
                p = li.find('p')
                if p is not None:
                    text = ''.join(p.itertext()).strip()
                    # Skip lines that are only whitespace/em-spaces (used for alignment)
                    cleaned = text.replace('\u2003', '').replace('\u00a0', '').strip()
                    if cleaned:
                        result_parts.append(text)
        
        # Check for interlinear word glosses
        word_lists = item.findall('list[@list-type="word"]')
        if word_lists:
            source_words = []
            gloss_words = []
            for wl in word_lists:
                word_items = wl.findall('list-item')
                if len(word_items) >= 1:
                    p0 = word_items[0].find('p')
                    if p0 is not None:
                        source_words.append(''.join(p0.itertext()).strip())
                if len(word_items) >= 2:
                    p1 = word_items[1].find('p')
                    if p1 is not None:
                        gloss_words.append(''.join(p1.itertext()).strip())
            
            if source_words:
                result_parts.append(" ".join(source_words))
            if gloss_words:
                result_parts.append(" ".join(gloss_words))
    
    return " / ".join(result_parts) if result_parts else ""


def _extract_paragraph_text(p_elem: etree._Element) -> str:
    """
    Extract plain text from a <p> element, handling inline elements like
    <xref>, <italic>, <bold>, <strike>, <sc>, etc.
    """
    return ''.join(p_elem.itertext()).strip()


def _parse_section_with_glosses(sec_elem: etree._Element) -> Section:
    """
    Parse a <sec> element, extracting both paragraphs and linguistic examples
    (gloss lists). Gloss lists are appended to the preceding paragraph.
    
    Args:
        sec_elem: An lxml element for <sec>.
        
    Returns:
        A Section with paragraphs that include linguistic examples.
    """
    # Get section title
    title_elem = sec_elem.find('title')
    title = ''.join(title_elem.itertext()).strip() if title_elem is not None else "Untitled"
    
    paragraphs = []
    
    # Walk through direct children of <sec> in document order
    for child in sec_elem:
        if child.tag == 'p':
            para_text = _extract_paragraph_text(child)
            if para_text:
                paragraphs.append(Paragraph(text=para_text, item_type="paragraph"))
        
        elif child.tag == 'list' and child.get('list-type') == 'gloss':
            # Extract the linguistic example text
            gloss_text = _extract_gloss_text(child)
            if gloss_text:
                if paragraphs:
                    # Append to the preceding paragraph
                    paragraphs[-1].text += "\n" + gloss_text
                else:
                    # No preceding paragraph; create a new one
                    paragraphs.append(Paragraph(text=gloss_text, item_type="paragraph"))
        
        elif child.tag == 'sec':
            # Nested sections are handled separately (they'll be found by
            # the recursive findall in parse_jats_to_sections)
            continue
    
    return Section(title=title, paragraphs=paragraphs)


def parse_jats_to_sections(xml_path: str | Path) -> Dict[str, Any]:
    """
    Parse a JATS XML file and extract sections with paragraphs.
    
    Uses the `jats` package for metadata extraction, and custom XML parsing
    for body content to properly handle linguistic examples encoded as
    <list list-type="gloss"> elements. These examples are appended to the
    preceding paragraph text.
    
    Args:
        xml_path: Path to the JATS XML file.
        
    Returns:
        Dict with 'metadata' (title, authors, abstract, etc.) and 'sections' list.
    """
    
    xml_path = Path(xml_path)
    article = parse_jats_xml(xml_path)
    
    # Extract metadata using the jats package (works fine for front matter)
    metadata = {
        "title": article.title,
        "authors": [
            {
                "given_names": a.given_names,
                "surname": a.surname,
                "orcid": a.orcid,
                "affiliation": a.affiliation
            }
            for a in (article.authors or [])
        ],
        "abstract": article.abstract,
        "article_id": article.article_id,
    }
    
    # Parse body sections directly from XML to handle gloss lists
    tree = etree.parse(str(xml_path))
    root = tree.getroot()
    body = root.find('.//body')
    
    sections = []
    if body is not None:
        for sec_elem in body.findall('.//sec'):
            section = _parse_section_with_glosses(sec_elem)
            if section.paragraphs or section.title:
                sections.append(section)
    
    return {
        "metadata": metadata,
        "sections": sections
    }


def sections_to_plain_text(data: Dict[str, Any], include_metadata: bool = True) -> str:
    """
    Convert parsed sections to plain text with preserved newlines.
    
    Args:
        data: Output from parse_jats_to_sections().
        include_metadata: Whether to include title/abstract at the top.
        
    Returns:
        Plain text with sections and paragraphs.
    """
    lines = []
    
    if include_metadata:
        meta = data["metadata"]
        if meta.get("title"):
            lines.append(f"# {meta['title']}")
            lines.append("")
        if meta.get("abstract"):
            lines.append("## Abstract")
            lines.append(meta["abstract"])
            lines.append("")
    
    for section in data["sections"]:
        lines.append(f"## {section.title}")
        lines.append("")
        for para in section.paragraphs:
            lines.append(para.text)
            lines.append("")
    
    return "\n".join(lines)


def sections_to_json(data: Dict[str, Any]) -> str:
    """
    Convert parsed sections to JSON format (for Input_articles.json compatibility).
    """
    # Convert dataclasses to dicts
    result = {
        "metadata": data["metadata"],
        "content": [
            {
                "section_name": s.title,
                "paragraphs": [p.text for p in s.paragraphs]
            }
            for s in data["sections"]
        ]
    }
    return json.dumps(result, indent=2, ensure_ascii=False)


def jats_to_article_json(xml_path: str | Path, strip_urls: bool = True) -> Dict[str, Any]:
    """
    Convert JATS XML to the same format as Input_articles.json entries.
    
    Args:
        xml_path: Path to the JATS XML file.
        strip_urls: If True (default), remove markdown-style links like 
                   [Merchant 2001](https://...) → Merchant 2001
    
    Returns a dict compatible with the existing article processing pipeline.
    """
    data = parse_jats_to_sections(xml_path)
    
    def process_text(text: str) -> str:
        if strip_urls:
            return strip_markdown_links(text)
        return text
    
    # Also strip URLs from abstract if present
    metadata = data["metadata"].copy()
    if metadata.get("abstract") and strip_urls:
        metadata["abstract"] = strip_markdown_links(metadata["abstract"])
    
    return {
        "article_id": metadata.get("article_id", Path(xml_path).stem),
        "metadata": metadata,
        "content": [
            {
                "section_name": s.title,
                "paragraphs": [process_text(p.text) for p in s.paragraphs]
            }
            for s in data["sections"]
        ]
    }


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python jats_to_sections.py <input.xml> [--json|--text]")
        sys.exit(1)
    
    xml_path = sys.argv[1]
    output_format = sys.argv[2] if len(sys.argv) > 2 else "--text"
    
    data = parse_jats_to_sections(xml_path)
    
    if output_format == "--json":
        print(sections_to_json(data))
    else:
        print(sections_to_plain_text(data))
