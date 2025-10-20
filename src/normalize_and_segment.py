import re
import json
import os
import shutil
import tempfile
import time
from openai import OpenAI
from dotenv import load_dotenv 

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from report_generator import BoardMember, CompanyReport, CoreCompetency, DDRGenerator, FinancialData
from embeddings import search_sections, build_section_embeddings, append_next_sections

HEADING_RE = re.compile(r'^(#{1,6})\s+(.*)$', re.M)
TABLE_BLOCK_RE = re.compile(r'(?:^\|.*\|\s*\n)+^\|(?:\s*:?-+:?\s*\|)+\s*\n(?:^\|.*\|\s*\n)+', re.M)

def split_by_h2(md_text: str) -> List[Dict]:
    sections: List[Dict] = []
    last_pos = 0
    last = None

    for m in HEADING_RE.finditer(md_text):
        title = m.group(2).strip()
        start = m.start()
        if last is not None:
            last["content"] = md_text[last_pos:start].strip()
        last = {"title": title, "content": ""}
        sections.append(last)
        last_pos = m.end()

    if last is not None:
        last["content"] = md_text[last_pos:].strip()
    else:
        if md_text.strip():
            sections.append({"title": "_preamble_", "content": md_text.strip()})

    return sections

def slugify(text):
    """Convert text to a URL-friendly slug"""
    # Remove special chars, convert to lowercase, replace spaces with hyphens
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    slug = re.sub(r'[\s_-]+', '-', slug)
    return slug.strip('-')

def extract_tables_from_lines(lines, start_idx, end_idx):
    """Extract markdown tables and their line ranges from a section"""
    tables = []
    in_table = False
    table_start = None
    table_lines = []

    for i, line in enumerate(lines[start_idx:end_idx], start=start_idx):
        line = line.strip()

        # Check if line contains table markers
        if '|' in line and line.count('|') >= 2:
            if not in_table:
                in_table = True
                table_start = i + 1
                table_lines = [line]
            else:
                table_lines.append(line)
        else:
            if in_table:
                # End of table
                tables.append({
                    'start_line': table_start,
                    'end_line': i,
                    'content': '\n'.join(table_lines),
                    'row_count': len([l for l in table_lines if '|' in l])
                })
                in_table = False
                table_lines = []

    # Handle table at end of section
    if in_table and table_lines:
        tables.append({
            'start_line': table_start,
            'end_line': start_idx + len(lines[start_idx:end_idx]) - 1,
            'content': '\n'.join(table_lines),
            'row_count': len([l for l in table_lines if '|' in l])
        })

    return tables

def normalize_and_segment_markdown(markdown_text, doc_filename):
    """
    Normalize & segment markdown into consistent sections
    Returns sections array and saves JSONL file
    """
    lines = markdown_text.split('\n')
    sections = []

    # Since all headings are ##, just look for those
    heading_pattern = r'^##\s+(.+)$'

    current_section = None
    section_start_line = 0

    for line_idx, line in enumerate(lines):
        match = re.match(heading_pattern, line.strip())

        if match:
            # Close previous section if exists
            if current_section is not None:
                current_section['end_line'] = line_idx - 1
                current_section['content'] = '\n'.join(lines[section_start_line:line_idx])

                # Extract tables from this section
                current_section['tables'] = extract_tables_from_lines(
                    lines, section_start_line, line_idx
                )

                sections.append(current_section)

            # Start new section
            title = match.group(1).strip()
            section_id = slugify(title)

            current_section = {
                'section_id': section_id,
                'title': title,
                'section_number': len(sections) + 1,  
                'start_line': line_idx + 1,
                'end_line': None, 
                'content': None,  
                'tables': [],
                'lang': 'EN'  
            }
            section_start_line = line_idx

    # Close final section
    if current_section is not None:
        current_section['end_line'] = len(lines) - 1
        current_section['content'] = '\n'.join(lines[section_start_line:])
        current_section['tables'] = extract_tables_from_lines(
            lines, section_start_line, len(lines)
        )
        sections.append(current_section)

    # Ensure parsed directory exists
    Path("data/sections_report").mkdir(parents=True, exist_ok=True)

    # Save as JSONL file
    jsonl_file = f"data/sections_report/{doc_filename}.jsonl"
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for section in sections:
            # Create record for JSONL
            record = {
                'section_id': section['section_id'],
                'title': section['title'],
                'section_number': section['section_number'],  # section numbering 
                'lines': [section['start_line'], section['end_line']],
                'tables': [
                    {
                        'lines': [t['start_line'], t['end_line']],
                        'row_count': t['row_count']
                    } for t in section['tables']
                ],
                'lang': section['lang'],
                'char_count': len(section['content']) if section['content'] else 0
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"JSONL saved to: {jsonl_file}")
    return sections