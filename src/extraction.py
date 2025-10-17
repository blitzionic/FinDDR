import re
import json
import os
from openai import OpenAI
from dotenv import load_dotenv 

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from report_generator import CompanyReport, DDRGenerator, FinancialData
from embeddings import search_sections, build_section_embeddings, append_next_sections


import argparse

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

def extract_md_tables(text: str) -> List[str]:
    """Extract markdown tables using regex"""
    tables: List[str] = []
    for m in TABLE_BLOCK_RE.finditer(text + "\n"):
        tables.append(m.group(0).strip())
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
                'lang': 'en'  # Default language
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

# --------------------------- Above are MD to JSONL helpers --------------------------------------

def get_text_from_lines(md_file: str, start_line: int, end_line: int) -> str:
    """Extract text from specific line ranges in markdown"""
    with open(f"data/parsed/{md_file}.md", "r", encoding="utf-8") as f:
      markdown_text = f.read()
    lines = markdown_text.split('\n')
    return '\n'.join(lines[start_line - 1 :end_line + 1])

# ----------------------- S 1.1 Basic Information ----------------------------------------------------------------

def query_company_name(jsonl_file, md_file) -> str:

    with open(f"data/parsed/{md_file}.md", "r", encoding="utf-8") as f:
      markdown_text = f.read()
    preview_text = markdown_text[:8000]
    
    try:
        prompt = f"""
        Please analyze the following text from a business document and extract the primary company name mentioned. 
        Return only the company name, nothing else.
        
        Text:
        {preview_text} 
        """

        response = client.chat.completions.create(model="gpt-3.5-turbo",  
        messages=[
            {"role": "system", "content": "You are an expert at extracting company names from business documents."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0)

        company_name = response.choices[0].message.content.strip()
        print(company_name)
        return company_name

    except Exception as e:
        print(f"Error querying LLM: {e}")
        return f"Error: {str(e)}"
    
def query_company_hq(jsonl_file, md_file) -> str:
  
    keywords = "company headquarters office location"
    company_name_sections = search_sections(keywords, 3, md_file)
    
    candidate_sections_text = []
    for result in company_name_sections:
      start_line, end_line = result["lines"]
      section_text = get_text_from_lines(md_file, start_line, end_line)
      
      print(f"{result['rank']}. {result['title']} (distance: {result['distance']:.3f})")
      print(f"   Section ID: {result['section_id']}")
      print(f"   Lines: {result['lines'][0]}-{result['lines'][1]}")
      print(f"   Chars: {result['char_count']:,}")
      print(f"   Original text:\n {get_text_from_lines(md_file, result['lines'][0], result['lines'][1])}")
      
      if (end_line - start_line) <= 1:
        print("⚠️ Detected one-liner section — appending next 5 sections for context.")
        section_text = append_next_sections(md_file, result["section_id"], num_next=5)
      candidate_sections_text.append(section_text)
    print(f"--------------------------------")
    combined_sections_text = "\n\n".join(candidate_sections_text)
    print(combined_sections_text)
    
    try:
        prompt = f"""
        Please analyze the following text from a business document and extract the primary company headquarters mentioned. 
        Return only the company headquarters: city, state/province/county, Country
        Example: "Santa Clara, California, USA"
        

        Text:
        {combined_sections_text}
        """

        response = client.chat.completions.create(model="gpt-3.5-turbo",  
        messages=[
            {"role": "system", "content": "You are an expert at extracting company headquarters from business documents."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0)

        company_hq = response.choices[0].message.content.strip()
        return company_hq

    except Exception as e:
        print(f"Error querying LLM: {e}")
        return f"Error: {str(e)}"


def query_establishment_date(jsonl_file, md_file) -> str:
  
    keywords = "company founding date establishment incorporation founded year origin history inception formation established"
    company_name_sections = search_sections(keywords, 3, md_file)
    
    candidate_sections_text = []
    for result in company_name_sections:
      start_line, end_line = result["lines"]
      section_text = get_text_from_lines(md_file, start_line, end_line)
      
      if (end_line - start_line) <= 1:
        print("⚠️ Detected one-liner section — appending next 5 sections for context.")
        section_text = append_next_sections(md_file, result["section_id"], num_next=5)
      candidate_sections_text.append(section_text)
    
    combined_sections_text = "\n\n".join(candidate_sections_text)
    print(combined_sections_text)
    
    try:
        prompt = f"""
        Analyze the following text from a company's official filing or business report and
        extract the **establishment or incorporation date**.

        Follow these strict rules:
        - Return **only one date**, the **earliest** if multiple are found.
        - The date must be formatted as one of:
            • "Month DD, YYYY"  (e.g., April 15, 1993)
            • "Month, YYYY"     (e.g., April, 1993)
            • "YYYY"            (e.g., 1993)
        - Use natural month names (January, February, etc.).
        - If the date is unknown or not found, return exactly: "N/A".
        - Do not include any explanation or text other than the date.
    
        Text:
        {combined_sections_text}
        """

        response = client.chat.completions.create(model="gpt-3.5-turbo",  
        messages=[
            {"role": "system", "content": "You are an expert in extracting founding and incorporation information from company filings."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0)

        establishment_date = response.choices[0].message.content.strip()
        return establishment_date

    except Exception as e:
        print(f"Error querying LLM: {e}")
        return f"Error: {str(e)}"

# --------------------------- S 1.2: Core Competencies -------------------------------------------
PERSPECTIVE_KWS = {
  "Innovation Advantages": [
    "innovation", "R&D", "technology leadership", "platform", "pipeline", "IP", "patents", "engineering", "value chain"
  ],
  "Product Advantages": [
    "products", "portfolio", "solutions", "capabilities", "quality", "safety", "reliability", "performance", "standards", "compliance",
    "countermeasures", "energetics", "solutions", "capabilities"
  ],
  "Brand Recognition": [
    "brand", "reputation", "preferred supplier", "niche", "high barriers", "sole source", "market-leading", "category leader", "award"
  ],
  "Reputation Ratings": [
    "ESG rating", "MSCI", "Sustainalytics", "CDP", "FTSE4Good", "Dow Jones Sustainability", "award", "ranking", "corporate responsibility", "sustainability",
    "corporate responsibility"
  ],
}

WIDE_BACKSTOP = [
    # fallback context if keywords fail
    "Our Company", "Business Strategies", "Strategy", "Overview", "ESG", "Sustainability",
    "Human Capital", "Products", "Platform", "Data Center", "Gaming", "Professional Visualization", "Automotive"
]

def _gather_candidates_for_keywords(md_file: str, keywords: list, top_k: int = 6) -> str:
    """
    Use FAISS search to gather candidate snippets for the keyword set.
    Expands one-liner sections with next N sections for context.
    """
    query = " ".join(keywords)
    hits = search_sections(query, top_k=top_k, md_file=md_file)
    chunks = []
    for h in hits:
        s, e = h["lines"]
        txt = get_text_from_lines(md_file, s, e)
        if (e - s) <= 1:  # header-only → expand
            txt = append_next_sections(md_file, h["section_id"], num_next=5) or txt
        if txt:
            chunks.append(txt[:8000])  # keep individual chunks bounded
    return "\n\n".join(chunks)

def _gather_backstop(md_file: str, top_k: int = 6) -> str:
    hits = search_sections(" ".join(WIDE_BACKSTOP), top_k=top_k, md_file=md_file)
    chunks = []
    for h in hits:
        s, e = h["lines"]
        txt = get_text_from_lines(md_file, s, e)
        if (e - s) <= 1:
            txt = append_next_sections(md_file, h["section_id"], num_next=5) or txt
        if txt:
            chunks.append(txt[:6000])
    return "\n\n".join(chunks)

def _llm_summarize_core_competency(perspective: str, year_text: str) -> str:
    """
    Ask LLM to produce a short, precise, non-hallucinatory summary for a single perspective.
    Returns <= ~2 sentences. If nothing, return "N/A".
    """
    # Guard against empty
    if not year_text or not year_text.strip():
        return "N/A"

    prompt = f"""
    You are summarizing ONE perspective of a company's core competencies based ONLY on the provided text.

    Perspective: {perspective}

    Rules:
    - Use only the provided text; do NOT invent or use outside knowledge.
    - Write a concise summary of at most 2 sentences. Be specific (e.g., name of platforms/tech) but brief.
    - No markdown, no bullets, no qualifiers—just the summary.

    Text:
    {year_text}
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": "You summarize filings precisely without hallucination."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=120,
            temperature=0
        )
        return (resp.choices[0].message.content or "").strip() or "N/A"
    except Exception as e:
        print(f"[S1.2] LLM error ({perspective}): {e}")
        return "N/A"

def extract_core_competencies(md_file: str) -> dict:
    """
    Extracts core competencies for a single report (one year).

    Returns:
    {
      "Innovation Advantages": "...",
      "Product Advantages":    "...",
      "Brand Recognition":     "...",
      "Reputation Ratings":    "..."
    }
    """
    out = {}
    backstop = _gather_backstop(md_file)

    for perspective, kws in PERSPECTIVE_KWS.items():
        # 1) gather candidate snippets
        blob = _gather_candidates_for_keywords(md_file, kws, top_k=7)
        if not blob:
            blob = backstop

        # 2) LLM summarize
        summary = _llm_summarize_core_competency(perspective, blob)

        # 3) tidy up
        def _tidy(s):
            return s.strip()[:550] if s and s.strip() and s != "N/A" else "N/A"

        out[perspective] = _tidy(summary)

    return out

def merge_core_competencies(
    core_2024: dict,
    core_2023: dict
) -> dict:
    """
    Merge core competency dicts from 2024 and 2023 into a unified structure.

    Returns:
    {
      "Innovation Advantages": {"2024": "...", "2023": "..."},
      "Product Advantages":    {"2024": "...", "2023": "..."},
      "Brand Recognition":     {"2024": "...", "2023": "..."},
      "Reputation Ratings":    {"2024": "...", "2023": "..."},
    }
    """
    merged = {}
    for key in PERSPECTIVE_KWS.keys():
        merged[key] = {
            "2024": core_2024.get(key, "N/A"),
            "2023": core_2023.get(key, "N/A")
        }
    return merged

# --------------------------- S 1.3: Mission & Vision -------------------------------------------

MISSION_KWS = [
    "mission", "our mission", "purpose", "purpose statement",
    "we exist to", "we are here to", "why we exist"
]

VISION_KWS = [
    "vision", "our vision", "we envision", "aspire to", "long-term vision",
    "what we aspire", "north star", "long term ambition"
]

VALUES_KWS = [
    "values", "core values", "our values", "what we value",
    "principles", "guiding principles", "code of conduct", "ethics and integrity"
]

# Broader catch-all to find a few relevant early sections if the exact words don't appear.
WIDE_KWS = [
    "strategy", "our company", "who we are", "what we do", "business overview",
    "culture", "governance", "sustainability", "esg", "purpose", "innovation"
]

def _gather_text_for(concept_keywords, md_file, top_k=5):
    """
    Retrieve candidate sections for a group of keywords; expand one-liners.
    Returns a text blob.
    """
    # Join keywords for a single semantic query
    query = " ".join(concept_keywords)
    hits = search_sections(query, top_k=top_k, md_file=md_file)

    chunks = []
    for hit in hits:
        s, e = hit["lines"]
        txt = get_text_from_lines(md_file, s, e)
        if (e - s) <= 1:  # safety expansion for headers/stubs
            txt = append_next_sections(md_file, hit["section_id"], num_next=5) or txt
        # Keep chunks modest to avoid token bloat
        if txt and len(txt) > 0:
            chunks.append(txt[:8000])
    return "\n\n".join(chunks)

def _heuristic_trim(s: str) -> str:
    """Clean up overly long or boilerplate-y text."""
    if not s:
        return s
    s = s.strip()
    # Drop trailing headings or obvious junk lines
    s = s.replace("Table of Contents", "").strip()
    # Cap length to something reasonable for a single cell
    return s[:500].strip()

def extract_mission_vision_values(jsonl_file: str, md_file: str) -> dict:
    """
    Returns dict:
      {
        "mission": "... or N/A",
        "vision": "... or N/A",
        "core_values": "... or N/A"
      }
    """
    # 1) Gather focused candidate text per concept
    mission_blob = _gather_text_for(MISSION_KWS, md_file, top_k=5)
    vision_blob  = _gather_text_for(VISION_KWS,  md_file, top_k=5)
    values_blob  = _gather_text_for(VALUES_KWS,  md_file, top_k=5)

    # 2) If everything is too thin, widen context using broader terms (esp. early report)
    if not any([mission_blob, vision_blob, values_blob]):
        wide_blob = _gather_text_for(WIDE_KWS, md_file, top_k=6)
    else:
        wide_blob = ""

    # 3) Build a combined context for the LLM with sections separated to reduce cross-bleed
    combined = "\n\n".join([
        "=== MISSION CANDIDATES ===\n" + mission_blob if mission_blob else "",
        "=== VISION CANDIDATES ===\n"  + vision_blob  if vision_blob  else "",
        "=== VALUES CANDIDATES ===\n"  + values_blob  if values_blob  else "",
        "=== EXTRA CONTEXT ===\n"     + wide_blob    if wide_blob    else "",
    ]).strip()

    # 4) Ask the model to extract strictly formatted fields
    #    IMPORTANT: we forbid invention; require "N/A" when not present.
    prompt = f"""
You are extracting **Mission Statement**, **Vision Statement**, and **Core Values** from a company's official report text.

Rules:
- Read ONLY the provided text. If a field is not clearly present, return "N/A" for that field.
- Prefer short, declarative sentences or concise phrases from the text.
- For Core Values, prefer a comma-separated list if present (e.g., "Integrity, Innovation, Customer Focus"). If not explicit, return "N/A".
- Do not invent, generalize, or use world knowledge. Use the closest in-text phrasing (even if approximate).
- Output **exactly** this JSON object (no extra text):

{{
  "mission": "<string or N/A>",
  "vision": "<string or N/A>",
  "core_values": "<comma-separated list or N/A>"
}}

Text:
{combined}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",  
            messages=[
                {"role": "system", "content": "You extract structured fields from filings without hallucination."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=250,
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        # Be defensive: parse JSON even if the model wraps it with fencing
        import json, re
        m = re.search(r"\{.*\}", raw, flags=re.S)
        data = json.loads(m.group(0)) if m else {}

        mission = _heuristic_trim(data.get("mission", "").strip() or "N/A")
        vision  = _heuristic_trim(data.get("vision", "").strip() or "N/A")
        values  = _heuristic_trim(data.get("core_values", "").strip() or "N/A")

        # 5) Light heuristic fallback: look for common patterns if still N/A
        def _regex_fallback(blob, pats):
            if not blob:
                return None
            for p in pats:
                match = re.search(p, blob, flags=re.I)
                if match:
                    return _heuristic_trim(match.group(1))
            return None

        if mission == "N/A":
            mission_fb = _regex_fallback(
                mission_blob,
                [
                    r"\bmission\b[:\-]\s*(.+?$)",
                    r"\bour mission\b\s*(?:is|:)\s*(.+?$)",
                    r"\bpurpose\b\s*(?:is|:)\s*(.+?$)",
                ]
            )
            if mission_fb: mission = mission_fb

        if vision == "N/A":
            vision_fb = _regex_fallback(
                vision_blob,
                [
                    r"\bvision\b[:\-]\s*(.+?$)",
                    r"\bour vision\b\s*(?:is|:)\s*(.+?$)",
                    r"\bwe envision\b\s*(.+?$)",
                    r"\baspire to\b\s*(.+?$)",
                ]
            )
            if vision_fb: vision = vision_fb

        if values == "N/A":
            values_fb = _regex_fallback(
                values_blob,
                [
                    r"\bcore values\b[:\-]\s*(.+?$)",
                    r"\bour values\b[:\-]\s*(.+?$)",
                    r"\bvalues\b[:\-]\s*(.+?$)",
                    r"\bguiding principles\b[:\-]\s*(.+?$)",
                ]
            )
            if values_fb: values = values_fb

        # Normalize empties to "N/A"
        mission = mission if mission else "N/A"
        vision  = vision  if vision  else "N/A"
        values  = values  if values  else "N/A"

        return {
            "mission": mission,
            "vision": vision,
            "core_values": values
        }

    except Exception as e:
        print(f"[S1.3] Error extracting Mission/Vision/Values: {e}")
        # Safe fallback
        return {"mission": "N/A", "vision": "N/A", "core_values": "N/A"}


# --------------------------- S 2.1 Income Statement -------------------------------------------
# INCOME_KWS = [
#     # EN
#     "income statement consolidated statement of operations statement of comprehensive income statement of profit or loss results of operations",
#     "revenue gross profit operating expense operating income net profit net income income before income taxes income tax expense interest expense finance costs",
#     # ZH
#     "利润表 综合损益表 营业收入 毛利 营业费用 营业利润 净利润 所得税前利润 所得税费用 利息费用 融资费用",
#     # ID
#     "laporan laba rugi pendapatan laba kotor beban operasi laba usaha laba bersih sebelum pajak beban pajak beban bunga biaya keuangan",
# ]

# def _gather_income_context(md_file: str, top_k: int = 12) -> str:
#     """
#     Use the FAISS index to pull likely income-statement sections.
#     Expand header-only sections. Dedupe by (start,end).
#     """
#     seen = set()
#     chunks = []
#     for q in INCOME_KWS:
#         hits = search_sections(q, top_k=top_k, md_file=md_file)
#         for h in hits:
#             s, e = h["lines"]
#             span = (int(s), int(e))
#             if span in seen:
#                 continue
#             seen.add(span)

#             txt = get_text_from_lines(md_file, s, e)
#             if (e - s) <= 1:
#                 # section looks like a single-line header → expand forward
#                 expanded = append_next_sections(md_file, h["section_id"], num_next=5)
#                 if expanded:
#                     txt = expanded
#             if txt:
#                 chunks.append(txt[:8000])  # cap per chunk
#     # fuse & trim global budget
#     return ("\n\n" + ("-"*40) + "\n\n").join(chunks)[:18000]


def _income_prompt(years: list[int], text: str) -> str:
    """
    Build a strict JSON-only extraction prompt.
    """
    years_csv = ", ".join(str(y) for y in years)
    return f"""
        You are given text snippets from a company's annual reports. Extract an Income Statement for years [{years_csv}].

        STRICT RULES
        - Use ONLY the provided text. If a value is not clearly present for a year, return "N/A" for that year.
        - Prefer CONSOLIDATED totals. If a table shows segments (e.g., 'Compute & Networking', 'Graphics'), do NOT sum them;
        use the consolidated line/column if available. If only segments exist with no consolidated total, return "N/A".
        - Parse negatives shown in parentheses, e.g., (257) → -257.
        - Detect currency and multiplier from headers like: "$ in millions", "$m", "£m", "€m", "SGD in millions", etc.
        Output multiplier as one of: "Units", "Thousands", "Millions", "Billions".
        Output currency as a 3-letter code if clear (USD, GBP, EUR, SGD, IDR, AUD, MYR, CNY, HKD), else best textual code.
        - DO NOT invent numbers. If conflicting tables exist, choose the one that explicitly matches the requested years.

        OUTPUT (JSON ONLY, no extra text):
        {{
        "years": [{years_csv}],
        "multiplier": "<Units|Thousands|Millions|Billions>",
        "currency": "<e.g., USD, GBP, EUR>",
        "fields": {{
            "Revenue": {{"{years[0]}": "N/A"}},
            "Cost of Goods Sold": {{}},
            "Gross Profit": {{}},
            "Operating Expense": {{}},
            "Operating Income": {{}},
            "Net Profit": {{}},
            "Income before income taxes": {{}},
            "Income tax expense(benefit)": {{}},
            "Interest Expense": {{}}
        }}
        }}

        Ensure every field has a mapping for every requested year with either a number (no commas) or "N/A".

        TEXT:
        {text}
        """.strip()

def _safe_json_from_llm(s: str) -> dict:
    """
    Extract the first JSON object from an LLM string.
    """
    import json, re
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

def _coerce_number_or_na(v):
    """
    Coerce strings like '(257)', '1,234', '  45.6 ' to float; keep 'N/A' as-is.
    """
    if v is None:
        return "N/A"
    if isinstance(v, (int, float)):
        return v
    s = str(v).strip()
    if s.upper() == "N/A":
        return "N/A"
    # parentheses negative
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("() ").replace(",", "")
    try:
        num = float(s)
        return -num if neg else num
    except Exception:
        return "N/A"

def extract_income_statement(income_text, years: list[int] = [2024, 2023, 2022]) -> dict:
    """
    Uses FAISS index to gather candidate sections, then LLM to extract S2.1 fields.
    Returns a dict:
    {
      "years": [...],
      "multiplier": "Millions",
      "currency": "USD",
      "fields": {
         "Revenue": { "2024": 60922, "2023": 26974, ... },
         ...
      }
    }
    """
    text = income_text
    if not text:
        # No context; produce empty shell
        return {
            "years": years,
            "multiplier": "Units",
            "currency": "USD",
            "fields": {k: {str(y): "N/A" for y in years} for k in [
                "Revenue","Cost of Goods Sold","Gross Profit","Operating Expense",
                "Operating Income","Net Profit","Income before income taxes",
                "Income tax expense(benefit)","Interest Expense"
            ]}
        }

    prompt = _income_prompt(years, text)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[
                {"role": "system", "content": "You extract precise financials in strict JSON and never invent data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=800
        )
        data = _safe_json_from_llm(resp.choices[0].message.content.strip())
    except Exception as e:
        print(f"[S2.1] LLM error: {e}")
        data = {}

    # Build a safe, normalized output
    target_fields = [
        "Revenue","Cost of Goods Sold","Gross Profit","Operating Expense",
        "Operating Income","Net Profit","Income before income taxes",
        "Income tax expense(benefit)","Interest Expense"
    ]
    out = {
        "years": years,
        "multiplier": (data.get("multiplier") or "Units").strip(),
        "currency": (data.get("currency") or "USD").strip(),
        "fields": {f: {} for f in target_fields}
    }

    fields = data.get("fields") or {}
    for f in target_fields:
        per_year = fields.get(f, {})
        for y in years:
            raw = per_year.get(str(y), "N/A")
            out["fields"][f][str(y)] = _coerce_number_or_na(raw)

    return out

# ---------- One formatter for all financial cells ----------
def format_financial_cell(value: Any) -> str:
    # mirrors DDRGenerator.format_financial_value (no lambdas/ternaries)
    if value is None or value == "N/A":
        return "N/A"

    if isinstance(value, str):
        s = value.strip()
        if s.upper() == "N/A":
            return "N/A"
        try:
            # handle "(335.8)" and "1,234.56"
            is_paren_negative = s.startswith("(") and s.endswith(")")
            s2 = s.strip("() ").replace(",", "")
            num = float(s2)
            if is_paren_negative or num < 0:
                num_abs = abs(num)
                return f"({num_abs:,})"
            return f"{num:,}"
        except Exception:
            return s

    if isinstance(value, (int, float)):
        if value < 0:
            num_abs = abs(value)
            return f"({num_abs:,})"
        return f"{value:,}"

    return str(value)

def print_income_statement_table(inc: dict):
    years = [str(y) for y in inc["years"]]
    header = ["Field"] + years

    # Print header
    print(" | ".join(header))
    print("-" * (len(" | ".join(header))))

    # Print each field row
    for field, per_year in inc["fields"].items():
        row = [field] + [format_financial_cell(per_year.get(y, "N/A")) for y in years]
        print(" | ".join(row))

    # Footer info
    print(f"\nCurrency: {inc['currency']}  |  Multiplier: {inc['multiplier']}") 
    
# LLLM identify 10 most likely sections to contain income statements 
# potential issue here: JSON files can have duplicate section ids and titles 
def _extract_json_array(text: str):
    """
    Extract first JSON array from a text blob (LLMs sometimes wrap JSON).
    Returns Python object or raises.
    """
    m = re.search(r"\[\s*(?:\{|\[).*?\]\s*", text, flags=re.S)
    if not m:
        raise ValueError("No JSON array found in LLM response.")
    return json.loads(m.group(0))

def llm_pick_income_statements_sections(jsonl_path: str, top_k: int = 10, batch_size: int = 150, model: str = "gpt-4o-mini") -> List[str]:
    """
    Use an LLM to choose the top-k sections most likely to contain any of these Income Statement fields:
      - Revenue
      - Cost of Goods Sold
      - Gross Profit
      - Operating Expense
      - Operating income
      - Net Profit
      - Income before income taxes
      - Income tax expense(benefit)
      - Interest Expense

    Returns a list of section_id strings (ranked best->worst).
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 1) load sections (title + id only)
    sections: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
            except Exception:
                continue
            sid = rec.get("section_id", "") or ""
            title = rec.get("title", "") or ""
            if sid or title:
                sections.append({"section_id": sid, "title": title})

    if not sections:
        return []

    # 2) batching
    def _batches(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    scored: List[Tuple[str, float]] = []

    # 3) ask LLM to score/select per batch
    TARGET_FIELDS = [
        "Revenue",
        "Cost of Goods Sold",
        "Gross Profit",
        "Operating Expense",
        "Operating income",
        "Net Profit",
        "Income before income taxes",
        "Income tax expense(benefit)",
        "Interest Expense",
    ]

    for chunk in _batches(sections, batch_size):
        compact = [
            {"section_id": s["section_id"], "title": s["title"][:180]}
            for s in chunk
        ]

        system_msg = (
            "You are a precise classifier for annual report sections. "
            "You never hallucinate, and you always return valid JSON arrays only."
        )

        user_prompt = f"""
            You will receive a list of sections from an annual report (title + section_id).
            Task: select the entries most likely to contain **any** of these Income Statement fields:

            - Revenue
            - Cost of Goods Sold
            - Gross Profit
            - Operating Expense
            - Operating income
            - Net Profit
            - Income before income taxes
            - Income tax expense (benefit)
            - Interest Expense

            Interpretations and synonyms you should treat as matches:
            - Revenue: "sales", "turnover"
            - Cost of Goods Sold: "cost of sales"
            - Gross Profit: "gross margin"
            - Operating Expense: "operating costs", "selling, general and administrative", "SG&A", "research and development"
            - Operating income: "operating profit", "EBIT"
            - Net Profit: "profit for the year", "profit attributable", "net income"
            - Income before income taxes: "profit before tax", "PBT"
            - Income tax expense(benefit): "taxation", "income tax"
            - Interest Expense: "finance costs", "interest payable", "borrowing costs"

            Also favor canonical statement containers:
            - "Consolidated income statement", "Statement of profit or loss", "Results of operations",
            "Financial statements", "Notes to the financial statements", "MD&A" sections specifically discussing results/operations.

            Avoid sections clearly unrelated (sustainability-only, governance-only, remuneration-only, auditor’s opinion text without numbers).

            Return a JSON array of objects, each with:
            - "section_id": string
            - "score": a number from 0.0 to 1.0 (higher = more likely)

            Return ONLY the JSON array. Do not include any explanatory text.

            Sections:
            {json.dumps(compact, ensure_ascii=False)}
                    """.strip()

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=800,
        )

        raw = (resp.choices[0].message.content or "").strip()
        try:
            arr = _extract_json_array(raw)
            for obj in arr:
                sid = str(obj.get("section_id", "")).strip()
                sc = float(obj.get("score", 0.0))
                if sid:
                    scored.append((sid, sc))
        except Exception:
            # silently skip bad batch
            pass

    if not scored:
        # 4) fallback: keyword heuristic focused on the requested fields & canonical containers
        KEYWORDS = [
            # direct fields
            "revenue", "sales", "turnover",
            "cost of goods sold", "cost of sales",
            "gross profit", "gross margin",
            "operating expense", "operating expenses", "operating costs", "sg&a", "selling general administrative", "research and development",
            "operating income", "operating profit", "ebit",
            "net profit", "profit for the year", "net income", "profit attributable",
            "income before income taxes", "profit before tax", "pbt",
            "income tax expense", "income tax benefit", "taxation", "income tax",
            "interest expense", "finance costs", "interest payable", "borrowing costs",
            # canonical containers
            "consolidated income statement", "statement of profit or loss",
            "results of operations", "financial statements", "financial statement",
            "notes to the financial statements", "md&a", "management discussion and analysis"
        ]

        def _score_heuristic(t: str) -> float:
            t2 = t.lower()
            # simple count of keyword hits
            count = 0
            for k in KEYWORDS:
                if k in t2:
                    count += 1
            return float(count)

        ranked = sorted(
            sections,
            key=lambda s: _score_heuristic((s.get("title") or "") + " " + (s.get("section_id") or "")),
            reverse=True
        )
        # return top_k section_ids
        return [str(s.get("section_id", "")) for s in ranked[:top_k] if s.get("section_id")]

    # 5) aggregate duplicate section_ids by max score, then take top_k
    best: Dict[str, float] = {}
    for sid, sc in scored:
        if sid not in best or sc > best[sid]:
            best[sid] = sc

    best: Dict[str, float] = {}
    for sid, sc in scored:
        if sid not in best or sc > best[sid]:
            best[sid] = sc

    # ✅ Print all section IDs and their scores before filtering
    print("\n=== All identified Income Statement–related sections ===")
    for sid, sc in sorted(best.items(), key=lambda x: x[1], reverse=True):
        print(f"{sid:50s} | score: {sc:.3f}")
    print("========================================================\n")
   
    ranked_ids = sorted(best.items(), key=lambda x: x[1], reverse=True)
    
    return [sid for sid, _ in ranked_ids[:top_k]]

def _read_jsonl_sections(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    Reads a sections JSONL. Each line should include:
      - title: str
      - section_id: str
      - section_number: int (optional but helpful)
      - lines: [start_line, end_line]
    Returns the list in file order. If section_number exists, we re-sort by it for safety.
    """
    records: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            records.append(rec)

    # Prefer document order by section_number if present (stable)
    def _order_key(rec: Dict[str, Any]) -> Tuple[int, int]:
        # Fallback to file index if section_number missing or not int
        num = rec.get("section_number", None)
        if isinstance(num, int):
            return (0, num)
        return (1, 0)

    records.sort(key=_order_key)
    return records


def _normalize_title(t: str) -> str:
    if t is None:
        return ""
    return t.strip().casefold()

def _build_id_index(records: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """
    Map section_id (both raw and casefolded) -> list of indices.
    We store two keys per id:
      - raw id: e.g., "chief-financial-officer"
      - casefolded key: "__CF__" + id.casefold()
    """
    index: Dict[str, List[int]] = {}
    i = 0
    while i < len(records):
        sid = records[i].get("section_id", "")
        if isinstance(sid, str) and sid != "":
            # raw
            if sid not in index:
                index[sid] = []
            index[sid].append(i)
            # casefolded
            sid_cf = sid.casefold()
            k = "__CF__" + sid_cf
            if k not in index:
                index[k] = []
            index[k].append(i)
        i += 1
    return index

def _load_md_lines(markdown_path: str) -> List[str]:
    with open(markdown_path, "r", encoding="utf-8") as f:
        return f.readlines()


def _clip(n: int, low: int, high: int) -> int:
    if n < low:
        return low
    if n > high:
        return high
    return n

def _window_indices(start_idx: int, total: int, window_size: int) -> List[int]:
    """
    Returns [start_idx, start_idx+1, ..., start_idx+(window_size-1)] clipped to total.
    """
    result: List[int] = []
    i = 0
    while i < window_size:
        j = start_idx + i
        if j >= total:
            break
        result.append(j)
        i += 1
    return result

def _window_line_span(records: List[Dict[str, Any]], idxs: List[int]) -> Optional[Tuple[int, int]]:
    """
    From a list of record indices, compute inclusive [start_line, end_line] span
    using each record's `lines = [start, end]`.
    Returns None if any `lines` are missing or malformed.
    """
    if len(idxs) == 0:
        return None

    has_bad = False
    min_start = None
    max_end = None

    k = 0
    while k < len(idxs):
        rec = records[idxs[k]]
        lines = rec.get("lines", None)
        if not isinstance(lines, list) or len(lines) != 2:
            has_bad = True
            break
        start_line = lines[0]
        end_line = lines[1]
        if not isinstance(start_line, int) or not isinstance(end_line, int):
            has_bad = True
            break

        if min_start is None or start_line < min_start:
            min_start = start_line
        if max_end is None or end_line > max_end:
            max_end = end_line
        k += 1

    if has_bad:
        return None
    return (min_start, max_end)

def _slice_markdown_lines(md_lines: List[str], start_line_inclusive: int, end_line_inclusive: int, one_based: bool = True) -> str:
    """
    Slices the markdown lines by the given inclusive line numbers.
    """
    if one_based:
        start_idx = start_line_inclusive - 1
        end_idx = end_line_inclusive
    else:
        start_idx = start_line_inclusive
        end_idx = end_line_inclusive + 1

    start_idx = _clip(start_idx, 0, len(md_lines))
    end_idx = _clip(end_idx, 0, len(md_lines))
    return "".join(md_lines[start_idx:end_idx])


_TABLE_RULE_ROW_RE = re.compile(r"[\|\s:\-]*$")

def _normalize_table_line(line: str) -> str:
    # If it's a markdown table row (has >=2 pipes), trim each cell
    if line.count("|") >= 2:
        parts = line.split("|")
        parts = [p.strip().replace("\xa0", " ") for p in parts]
        # Rejoin with single spaces around pipes
        line = " | ".join(parts)
        # Normalize any odd spacing around pipes again
        line = re.sub(r"\s*\|\s*", " | ", line).strip()
        # If this line is just pipes/spaces/dashes/colons (header rule rows), keep it compact
        if _TABLE_RULE_ROW_RE.fullmatch(line):
            line = re.sub(r"\s+", "", line)  # "|---|:---:|" style, no spaces
    return line

def _normalize_block(s: str) -> str:
    # 1) unify line endings & normalize NBSP/tabs
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\xa0", " ").replace("\t", " ")
    # 2) trim trailing spaces per line
    lines = [ln.rstrip() for ln in s.split("\n")]
    # 3) normalize table rows cell-by-cell
    lines = [_normalize_table_line(ln) for ln in lines]
    # 4) collapse long runs of spaces inside non-table lines
    lines = [re.sub(r" {2,}", " ", ln) if ln.count("|") < 2 else ln for ln in lines]
    # 5) collapse 3+ blank lines to max 1 blank line
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    # 6) strip outer whitespace
    return out.strip()

def assemble_financial_statement_windows_from_ids(
    top5_ids: List[str],
    sections_jsonl_path: str,
    original_markdown_path: str,
    window_size: int = 10,
    one_based_lines: bool = True,
    choose_first_match_only: bool = False,  # <-- default to include ALL occurrences
    debug: bool = False
) -> Tuple[List[Dict[str, Any]], str]:
    """
    For each seed section_id in `top5_ids`:
      - find *all* occurrences in sections JSONL by section_id only (raw + casefolded)
      - for each occurrence, take that section plus the next (window_size - 1) sections
      - compute start/end line span from JSONL `lines`
      - slice the original markdown and collect text
    """
    # 1) Load and index
    records = _read_jsonl_sections(sections_jsonl_path)
    id_index = _build_id_index(records)
    md_lines = _load_md_lines(original_markdown_path)
    

    windows_info: List[Dict[str, Any]] = []
    combined_parts: List[str] = []

    total = len(records)
    if debug:
        print(f"[assemble/ids] Loaded {total} records")
        print(f"[assemble/ids] Markdown: {original_markdown_path}")

    i = 0
    while i < len(top5_ids):
        seed_id = str(top5_ids[i])
        
        # Collect all matches: raw + casefolded, then dedupe + sort in doc order
        all_matches: List[int] = []
        if seed_id in id_index:
            all_matches.extend(id_index[seed_id])

        seed_id_cf = seed_id.casefold()
        key_cf = "__CF__" + seed_id_cf
        if key_cf in id_index:
            all_matches.extend(id_index[key_cf])

        # Deduplicate while preserving order
        seen = set()
        unique_matches: List[int] = []
        j = 0
        while j < len(all_matches):
            idx = all_matches[j]
            if idx not in seen:
                unique_matches.append(idx)
                seen.add(idx)
            j += 1

        if len(unique_matches) == 0:
            if debug:
                print(f"[warn] section_id '{seed_id}' not found. Skipping.")
            i += 1
            continue

        if choose_first_match_only:
            # only keep the earliest occurrence
            unique_matches = [unique_matches[0]]

        if debug:
            print(f"[match] '{seed_id}' -> occurrences: {unique_matches}")

        # For each occurrence: build a window and append text
        occ_num = 0
        while occ_num < len(unique_matches):
            seed_idx = unique_matches[occ_num]

            # window indices
            idxs = _window_indices(seed_idx, total, window_size)
            if debug:
                print(f"[window] '{seed_id}' occ#{occ_num+1} -> seed_idx={seed_idx}, size={len(idxs)}")

            # compute span
            span = _window_line_span(records, idxs)
            if span is None:
                if debug:
                    print(f"[warn] '{seed_id}' occ#{occ_num+1} has missing/malformed line spans. Skipping.")
                occ_num += 1
                continue

            start_line = span[0]
            end_line = span[1]
            if debug:
                print(f"[span] '{seed_id}' occ#{occ_num+1} -> {start_line}..{end_line}")

            # collect ids/titles for reference
            window_ids: List[str] = []
            window_titles: List[str] = []
            k = 0
            while k < len(idxs):
                rec = records[idxs[k]]
                window_ids.append(str(rec.get("section_id", "")))
                window_titles.append(str(rec.get("title", "")))
                k += 1

            # slice markdown
            # text_blob = _slice_markdown_lines(md_lines, start_line, end_line, one_based=one_based_lines).strip("\n")
            
            text_blob = _slice_markdown_lines(
                md_lines, start_line, end_line, one_based=one_based_lines
            )
            text_blob = _normalize_block(text_blob)

            # bundle
            bundle = {
                "seed_section_id": seed_id,
                "occurrence_index": seed_idx,
                "window_indices": idxs,
                "window_section_ids": window_ids,
                "window_titles": window_titles,
                "start_line": start_line,
                "end_line": end_line,
                "text": text_blob
            }
            windows_info.append(bundle)

            # append to combined
            combined_parts.append(
                "\n\n"
                + "===== WINDOW FOR section_id: " + seed_id
                + " (occurrence seed_idx " + str(seed_idx) + ") =====\n"
                + "Start line: " + str(start_line) + " | End line: " + str(end_line) + "\n\n"
                + text_blob
                + "\n===== END WINDOW =====\n"
            )

            occ_num += 1

        i += 1

    combined_text = "".join(combined_parts)
    if debug:
        print(f"[result] windows={len(windows_info)} | combined_len={len(combined_text)}")
    return windows_info, combined_text

# merge two tables (2023 + 2024)

def merge_income_statements_per_year_priority(
    inc_2024: Dict[str, Any],
    inc_2023: Dict[str, Any],
    years: List[int] = [2024, 2023, 2022],
    debug: bool = False
) -> Dict[str, Any]:
    """
    Merge two extracted income-statement dicts with a global priority policy:
      - For ALL requested years, prefer values from inc_2024; if a value is "N/A", fall back to inc_2023.

    Also:
      - Normalize values to a common multiplier (prefer inc_2024's; else inc_2023's; fallback "Units").
      - If currencies disagree (both present and different), mark currency as "MIXED".

    Returns the same schema as extract_income_statement:
    {
      "years": [2024, 2023, 2022],
      "multiplier": "Millions",
      "currency": "GBP",
      "fields": {
        "Revenue": {"2024": 510.4, "2023": 472.6, "2022": 442.8},
        ...
      }
    }
    """

    # Canonical field order
    target_fields = [
        "Revenue",
        "Cost of Goods Sold",
        "Gross Profit",
        "Operating Expense",
        "Operating Income",
        "Net Profit",
        "Income before income taxes",
        "Income tax expense(benefit)",
        "Interest Expense"
    ]

    # ---------- helpers ----------
    def _canon_mult(m: Optional[str]) -> str:
        if m is None:
            return "Units"
        s = str(m).strip().lower()
        if s == "unit" or s == "units":
            return "Units"
        if s == "thousand" or s == "thousands" or s == "k":
            return "Thousands"
        if s == "million" or s == "millions" or s == "mm" or s == "m":
            return "Millions"
        if s == "billion" or s == "billions" or s == "bn" or s == "b":
            return "Billions"
        return "Units"

    def _mult_factor(mult: str) -> float:
        if mult == "Units":
            return 1.0
        if mult == "Thousands":
            return 1_000.0
        if mult == "Millions":
            return 1_000_000.0
        if mult == "Billions":
            return 1_000_000_000.0
        return 1.0

    def _normalize_value_to_multiplier(value: Any, from_mult: str, to_mult: str) -> Any:
        if value is None:
            return "N/A"
        if value == "N/A":
            return "N/A"
        if isinstance(value, str):
            s = value.strip()
            if s.upper() == "N/A":
                return "N/A"
            try:
                value = float(s)
            except Exception:
                return "N/A"
        if isinstance(value, (int, float)):
            f_from = _mult_factor(from_mult)
            f_to = _mult_factor(to_mult)
            if f_to == 0.0:
                return "N/A"
            scaled = float(value) * (f_from / f_to)
            return scaled
        return "N/A"

    def _get_field_value(src: Dict[str, Any], field: str, year: int) -> Any:
        fields = src.get("fields", {})
        per_year = fields.get(field, {})
        val = per_year.get(str(year), "N/A")
        return val

    # ---------- choose global multiplier and currency ----------
    mult_2024 = _canon_mult(inc_2024.get("multiplier"))
    mult_2023 = _canon_mult(inc_2023.get("multiplier"))

    if mult_2024 is not None and mult_2024 != "":
        target_multiplier = mult_2024
    else:
        target_multiplier = mult_2023
    if target_multiplier is None or target_multiplier == "":
        target_multiplier = "Units"

    cur_2024 = ""
    cur_2023 = ""
    if inc_2024.get("currency") is not None:
        cur_2024 = str(inc_2024.get("currency")).strip()
    if inc_2023.get("currency") is not None:
        cur_2023 = str(inc_2023.get("currency")).strip()

    if cur_2024 != "" and cur_2023 != "" and cur_2024 != cur_2023:
        target_currency = "MIXED"
    elif cur_2024 != "":
        target_currency = cur_2024
    elif cur_2023 != "":
        target_currency = cur_2023
    else:
        target_currency = "USD"

    # ---------- per-year source preference map ----------
    # Maps year -> ("primary", "fallback")
    # primary is the report whose value we prefer for that year.
    year_source_pref: Dict[int, str] = {}
    k = 0
    while k < len(years):
        y = years[k]
        if y == 2024:
            year_source_pref[y] = "inc_2024_first"
        elif y == 2023:
            year_source_pref[y] = "inc_2023_first"
        else:
            # for 2022 and other older years, prefer the older report (inc_2023) first
            year_source_pref[y] = "inc_2023_first"
        k += 1

    # ---------- build merged ----------
    merged: Dict[str, Any] = {
        "years": years,
        "multiplier": target_multiplier,
        "currency": target_currency,
        "fields": {}
    }

    i = 0
    while i < len(target_fields):
        field = target_fields[i]
        merged["fields"][field] = {}
        j = 0
        while j < len(years):
            y = years[j]

            # pull raw values
            v24_raw = _get_field_value(inc_2024, field, y)
            v23_raw = _get_field_value(inc_2023, field, y)

            # normalize both to target multiplier (so comparisons are apples-to-apples)
            v24_norm = _normalize_value_to_multiplier(v24_raw, _canon_mult(inc_2024.get("multiplier")), target_multiplier)
            v23_norm = _normalize_value_to_multiplier(v23_raw, _canon_mult(inc_2023.get("multiplier")), target_multiplier)

            if v24_norm != "N/A":
                chosen = v24_norm
            else:
                if v23_norm != "N/A":
                    chosen = v23_norm
                else:
                    chosen = "N/A"

            merged["fields"][field][str(y)] = chosen
            j += 1
        i += 1

    if debug:
        print("[merge-income] multiplier:", target_multiplier)
        print("[merge-income] currency:", target_currency)
        print("[merge-income] policy: 2024 overrides 2023 for all years")
        
    return merged


# --------------------------- S 2.2 Balance Sheet (2024 + 2023 + 2022) -------------------------------------------

def _balance_prompt(years: list[int], text: str) -> str:
    """
    Build a strict JSON-only extraction prompt for Balance Sheet.
    """
    years_csv = ", ".join(str(y) for y in years)
    return f"""
        You are given text snippets from a company's annual reports. Extract a Balance Sheet for years [{years_csv}].

        STRICT RULES
        - Use ONLY the provided text. If a value is not clearly present for a year, return "N/A" for that year.
        - Prefer CONSOLIDATED totals. Do NOT add up segments; use the consolidated line/column when available.
        - Parse negatives shown in parentheses, e.g., (335.8) → -335.8.
        - Detect currency and multiplier from headers like: "$ in millions", "$m", "£m", "€m", "SGD in millions", etc.
          Output multiplier as one of: "Units", "Thousands", "Millions", "Billions".
          Output currency as a 3-letter code if clear (USD, GBP, EUR, SGD, IDR, AUD, MYR, CNY, HKD), else best textual code.
        - DO NOT invent numbers. If conflicting tables exist, choose the one that explicitly matches the requested years.
        - If a field truly does not appear, use "N/A" for all years for that field.

        FIELDS (exact keys):
        - Total Assets
        - Current Assets
        - Non-Current Assets
        - Total Liabilities
        - Current Liabilities
        - Non-Current Liabilities
        - Shareholders' Equity
        - Retained Earnings
        - Total Equity and Liabilities
        - Inventories
        - Prepaid Expenses

        OUTPUT (JSON ONLY, no extra text):
        {{
        "years": [{years_csv}],
        "multiplier": "<Units|Thousands|Millions|Billions>",
        "currency": "<e.g., USD, GBP, EUR>",
        "fields": {{
            "Total Assets": {{"{years[0]}": "N/A"}},
            "Current Assets": {{}},
            "Non-Current Assets": {{}},
            "Total Liabilities": {{}},
            "Current Liabilities": {{}},
            "Non-Current Liabilities": {{}},
            "Shareholders' Equity": {{}},
            "Retained Earnings": {{}},
            "Total Equity and Liabilities": {{}},
            "Inventories": {{}},
            "Prepaid Expenses": {{}}
        }}
        }}

        Ensure every field has a mapping for every requested year with either a number (no commas) or "N/A".

        TEXT:
        {text}
        """.strip()
        
def extract_balance_sheet(bs_text: str, years: list[int] = [2024, 2023, 2022]) -> dict:
    """
    LLM extracts Balance Sheet fields as strict JSON. Mirrors S2.1 extractor style.
    """
    text = bs_text
    if not text:
        # empty shell
        target_fields = [
            "Total Assets","Current Assets","Non-Current Assets","Total Liabilities",
            "Current Liabilities","Non-Current Liabilities","Shareholders' Equity",
            "Retained Earnings","Total Equity and Liabilities","Inventories","Prepaid Expenses"
        ]
        empty = {f: {str(y): "N/A" for y in years} for f in target_fields}
        return {
            "years": years,
            "multiplier": "Units",
            "currency": "USD",
            "fields": empty
        }

    prompt = _balance_prompt(years, text)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You extract precise financials in strict JSON and never invent data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1000
        )
        data = _safe_json_from_llm(resp.choices[0].message.content.strip())
    except Exception as e:
        print(f"[S2.2] LLM error: {e}")
        data = {}

    target_fields = [
        "Total Assets","Current Assets","Non-Current Assets","Total Liabilities",
        "Current Liabilities","Non-Current Liabilities","Shareholders' Equity",
        "Retained Earnings","Total Equity and Liabilities","Inventories","Prepaid Expenses"
    ]

    out = {
        "years": years,
        "multiplier": (data.get("multiplier") or "Units").strip(),
        "currency": (data.get("currency") or "USD").strip(),
        "fields": {f: {} for f in target_fields}
    }

    fields = data.get("fields") or {}
    i = 0
    while i < len(target_fields):
        f = target_fields[i]
        per_year = fields.get(f, {})
        j = 0
        while j < len(years):
            y = years[j]
            raw = per_year.get(str(y), "N/A")
            out["fields"][f][str(y)] = _coerce_number_or_na(raw)
            j += 1
        i += 1
    print(f"[S2.2] Extracted Balance Sheet: {json.dumps(out)}")
    return out

def llm_pick_balance_sheet_sections(
    jsonl_path: str,
    top_k: int = 20,
    batch_size: int = 150,
    model: str = "gpt-4o-mini"
) -> List[str]:
    """
    Use an LLM to choose top-k sections likely to contain Balance Sheet lines.
    Returns section_ids ranked best->worst.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # load section titles + ids
    sections: List[Dict[str, str]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            sid = rec.get("section_id", "") or ""
            title = rec.get("title", "") or ""
            if sid != "" or title != "":
                sections.append({"section_id": sid, "title": title})

    if len(sections) == 0:
        return []

    def _batches(lst, n):
        i = 0
        while i < len(lst):
            yield lst[i:i+n]
            i += n

    TARGET_FIELDS = [
        "Total Assets","Current Assets","Non-Current Assets","Total Liabilities",
        "Current Liabilities","Non-Current Liabilities","Shareholders' Equity",
        "Retained Earnings","Total Equity and Liabilities","Inventories","Prepaid Expenses"
    ]

    scored: List[Tuple[str, float]] = []

    for chunk in _batches(sections, batch_size):
        compact = []
        i = 0
        while i < len(chunk):
            item = chunk[i]
            compact.append({
                "section_id": item["section_id"],
                "title": item["title"][:180]
            })
            i += 1

        system_msg = "You are a precise classifier for annual report sections. Return ONLY a JSON array."
        user_prompt = (
            "You will receive a list of sections (title + section_id). "
            "Select entries most likely to contain Balance Sheet lines:\n\n"
            + "\n".join(f"- {t}" for t in TARGET_FIELDS) +
            "\n\nStrong signals:\n"
            "- 'Consolidated balance sheet', 'Statement of financial position', 'Financial statements', "
            "'Notes to the financial statements' discussing assets, liabilities, equity.\n"
            "Avoid remuneration-only, governance-only, ESG-only, auditor opinion.\n\n"
            "Return ONLY a JSON array of objects:\n"
            "[{\"section_id\": \"...\", \"score\": 0.0..1.0}, ...]\n\n"
            "Sections:\n" + json.dumps(compact, ensure_ascii=False)
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=800,
        )

        raw = (resp.choices[0].message.content or "").strip()
        try:
            arr = _extract_json_array(raw)
            j = 0
            while j < len(arr):
                obj = arr[j]
                sid = str(obj.get("section_id", "")).strip()
                try:
                    sc = float(obj.get("score", 0.0))
                except Exception:
                    sc = 0.0
                if sid != "":
                    scored.append((sid, sc))
                j += 1
        except Exception:
            # fallback handled below if nothing scored
            pass

    if len(scored) == 0:
        # keyword fallback
        KEYWORDS = [
            "balance sheet", "statement of financial position",
            "assets", "liabilities", "equity", "retained earnings",
            "current assets", "non-current assets",
            "current liabilities", "non-current liabilities",
            "inventories", "prepaid"
        ]

        def _score_heur(t: str) -> float:
            t2 = t.lower()
            count = 0
            i = 0
            while i < len(KEYWORDS):
                if KEYWORDS[i] in t2:
                    count += 1
                i += 1
            return float(count)

        ranked = sorted(
            sections,
            key=lambda s: _score_heur((s.get("title") or "") + " " + (s.get("section_id") or "")),
            reverse=True
        )
        out = []
        i = 0
        while i < min(top_k, len(ranked)):
            sid = str(ranked[i].get("section_id", ""))
            if sid != "":
                out.append(sid)
            i += 1
        return out

    # aggregate by max score
    best: Dict[str, float] = {}
    i = 0
    while i < len(scored):
        sid, sc = scored[i]
        if sid not in best:
            best[sid] = sc
        else:
            if sc > best[sid]:
                best[sid] = sc
        i += 1

    ranked_ids = sorted(best.items(), key=lambda x: x[1], reverse=True)
    out = []
    i = 0
    limit = top_k if top_k < len(ranked_ids) else len(ranked_ids)
    while i < limit:
        out.append(ranked_ids[i][0])
        i += 1
    return out

def merge_balance_sheet_per_year_priority(
    bs_2024: Dict[str, Any],
    bs_2023: Dict[str, Any],
    years: List[int] = [2024, 2023, 2022],
    debug: bool = False
) -> Dict[str, Any]:
    """
    Merge two extracted balance-sheet dicts with a global priority:
      - For ALL years (2024, 2023, 2022...), prefer bs_2024; fallback to bs_2023.
    Normalizes to a common multiplier; resolves currency (2024 if present, else 2023; 'MIXED' if both present and different).
    """

    target_fields = [
        "Total Assets", "Current Assets", "Non-Current Assets", "Total Liabilities",
        "Current Liabilities", "Non-Current Liabilities", "Shareholders' Equity",
        "Retained Earnings", "Total Equity and Liabilities", "Inventories", "Prepaid Expenses"
    ]

    def _canon_mult(m: Optional[str]) -> str:
        if m is None:
            return "Units"
        s = str(m).strip().lower()
        if s == "unit" or s == "units":
            return "Units"
        if s == "thousand" or s == "thousands" or s == "k":
            return "Thousands"
        if s == "million" or s == "millions" or s == "mm" or s == "m":
            return "Millions"
        if s == "billion" or s == "billions" or s == "bn" or s == "b":
            return "Billions"
        return "Units"

    def _mult_factor(mult: str) -> float:
        if mult == "Units":
            return 1.0
        if mult == "Thousands":
            return 1_000.0
        if mult == "Millions":
            return 1_000_000.0
        if mult == "Billions":
            return 1_000_000_000.0
        return 1.0

    def _normalize_value_to_multiplier(value: Any, from_mult: str, to_mult: str) -> Any:
        if value is None:
            return "N/A"
        if isinstance(value, str):
            s = value.strip()
            if s.upper() == "N/A":
                return "N/A"
            coerced = _coerce_number_or_na(s)
            if isinstance(coerced, str) and coerced == "N/A":
                return "N/A"
            value_num = coerced
        elif isinstance(value, (int, float)):
            value_num = float(value)
        else:
            return "N/A"

        f_from = _mult_factor(from_mult)
        f_to = _mult_factor(to_mult)
        if f_to == 0.0:
            return "N/A"
        return float(value_num) * (f_from / f_to)

    def _get_field_value(src: Dict[str, Any], field: str, year: int) -> Any:
        fields = src.get("fields", {})
        per_year = fields.get(field, {})
        return per_year.get(str(year), "N/A")

    # Choose global multiplier and currency (prefer 2024’s metadata if present)
    mult_2024 = _canon_mult(bs_2024.get("multiplier"))
    mult_2023 = _canon_mult(bs_2023.get("multiplier"))
    if mult_2024 is not None and mult_2024 != "":
        target_multiplier = mult_2024
    else:
        target_multiplier = mult_2023
    if target_multiplier is None or target_multiplier == "":
        target_multiplier = "Units"

    cur_2024 = str(bs_2024.get("currency") or "").strip()
    cur_2023 = str(bs_2023.get("currency") or "").strip()
    if cur_2024 != "" and cur_2023 != "" and cur_2024 != cur_2023:
        target_currency = "MIXED"
    elif cur_2024 != "":
        target_currency = cur_2024
    elif cur_2023 != "":
        target_currency = cur_2023
    else:
        target_currency = "USD"

    merged: Dict[str, Any] = {
        "years": years,
        "multiplier": target_multiplier,
        "currency": target_currency,
        "fields": {}
    }

    i = 0
    while i < len(target_fields):
        field = target_fields[i]
        merged["fields"][field] = {}

        j = 0
        while j < len(years):
            y = years[j]

            # Always prefer the value extracted from the 2024 report (for any year),
            # and only if it's "N/A" fall back to the 2023 report's value.
            v24_raw = _get_field_value(bs_2024, field, y)
            v23_raw = _get_field_value(bs_2023, field, y)

            v24_norm = _normalize_value_to_multiplier(v24_raw, mult_2024, target_multiplier)
            v23_norm = _normalize_value_to_multiplier(v23_raw, mult_2023, target_multiplier)

            if v24_norm != "N/A":
                chosen = v24_norm
            else:
                if v23_norm != "N/A":
                    chosen = v23_norm
                else:
                    chosen = "N/A"

            merged["fields"][field][str(y)] = chosen
            j += 1

        i += 1

    if debug:
        print("[merge-balance] multiplier:", target_multiplier)
        print("[merge-balance] currency:", target_currency)
        print("[merge-balance] policy: 2024 overrides 2023 for all years")

    return merged
        
# def _format_balance_cell(value: Any) -> str:
#     # mirror DDRGenerator.format_financial_value
#     if value is None or value == "N/A":
#         return "N/A"
#     if isinstance(value, str):
#         s = value.strip()
#         if s.upper() == "N/A":
#             return "N/A"
#         # if string looks numeric (maybe from JSON), coerce
#         try:
#             # handle strings like '(335.8)' and '1,234.5'
#             neg = s.startswith("(") and s.endswith(")")
#             s2 = s.strip("() ").replace(",", "")
#             num = float(s2)
#             if neg or num < 0:
#                 return f"({abs(num):,})"
#             return f"{num:,}"
#         except Exception:
#             return s
#     if isinstance(value, (int, float)):
#         if value < 0:
#             return f"({abs(value):,})"
#         return f"{value:,}"
#     return str(value)

def print_balance_sheet_table(bs: dict):
    years = [str(y) for y in bs.get("years", [2024, 2023, 2022])]
    # Header: include Multiplier, Currency columns 
    header = ["Field"] + years + ["Multiplier", "Currency"]
    print(" | ".join(header))
    print("-" * (len(" | ".join(header))))

    multiplier = bs.get("multiplier", "Units")
    currency = bs.get("currency", "USD")

    # rows in deterministic order
    order = [
        "Total Assets","Current Assets","Non-Current Assets","Total Liabilities",
        "Current Liabilities","Non-Current Liabilities","Shareholders' Equity",
        "Retained Earnings","Total Equity and Liabilities","Inventories","Prepaid Expenses"
    ]

    fields = bs.get("fields", {})
    for field in order:
        per_year = fields.get(field, {})
        row_values = []
        i = 0
        while i < len(years):
            y = years[i]
            row_values.append(format_financial_cell(per_year.get(y, "N/A")))
            i += 1
        # repeat multiplier/currency per row
        row = [field] + row_values + [multiplier, currency]
        print(" | ".join(row))
        
    
# --------------------------- S 2.3 Cash Flow Statement (2024 + 2023 + 2022) -------------------------------------------       
def _cashflow_prompt(years: list[int], text: str) -> str:
    """
    Build a strict JSON-only extraction prompt for Cash Flow Statement.
    """
    years_csv = ", ".join(str(y) for y in years)
    return f"""
        You are given text snippets from a company's annual reports. Extract a Cash Flow Statement for years [{years_csv}].

        STRICT RULES
        - Use ONLY the provided text. If a value is not clearly present for a year, return "N/A" for that year.
        - Prefer CONSOLIDATED totals; do NOT sum segments.
        - Parse negatives shown in parentheses, e.g., (47.6) → -47.6.
        - Detect currency and multiplier from headers like: "$ in millions", "$m", "£m", "€m", etc.
          Output multiplier as one of: "Units", "Thousands", "Millions", "Billions".
          Output currency as a 3-letter code if clear (USD, GBP, EUR, SGD, IDR, AUD, MYR, CNY, HKD), else best textual code.
        - DO NOT invent numbers. If conflicting tables exist, choose the one that explicitly matches the requested years.

        FIELDS (exact keys):
        - Net Cash Flow from Operations
        - Net Cash Flow from Investing
        - Net Cash Flow from Financing
        - Net Increase/Decrease in Cash
        - Dividends

        OUTPUT (JSON ONLY, no extra text):
        {{
        "years": [{years_csv}],
        "multiplier": "<Units|Thousands|Millions|Billions>",
        "currency": "<e.g., USD, GBP, EUR>",
        "fields": {{
            "Net Cash Flow from Operations": {{"{years[0]}": "N/A"}},
            "Net Cash Flow from Investing": {{}},
            "Net Cash Flow from Financing": {{}},
            "Net Increase/Decrease in Cash": {{}},
            "Dividends": {{}}
        }}
        }}

        Ensure every field has a mapping for every requested year with either a number (no commas) or "N/A".

        TEXT:
        {text}
        """.strip()
      
def extract_cash_flow_statement(cf_text: str, years: list[int] = [2024, 2023, 2022]) -> dict:
    """
    LLM extracts Cash Flow Statement fields as strict JSON.
    """
    text = cf_text
    target_fields = [
        "Net Cash Flow from Operations",
        "Net Cash Flow from Investing",
        "Net Cash Flow from Financing",
        "Net Increase/Decrease in Cash",
        "Dividends"
    ]

    if not text:
        empty = {f: {str(y): "N/A" for y in years} for f in target_fields}
        return {
            "years": years,
            "multiplier": "Units",
            "currency": "USD",
            "fields": empty
        }

    prompt = _cashflow_prompt(years, text)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You extract precise financials in strict JSON and never invent data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=900
        )
        data = _safe_json_from_llm(resp.choices[0].message.content.strip())
    except Exception as e:
        print(f"[S2.3] LLM error: {e}")
        data = {}

    out = {
        "years": years,
        "multiplier": (data.get("multiplier") or "Units").strip(),
        "currency": (data.get("currency") or "USD").strip(),
        "fields": {f: {} for f in target_fields}
    }

    fields = data.get("fields") or {}
    i = 0
    while i < len(target_fields):
        f = target_fields[i]
        per_year = fields.get(f, {})
        j = 0
        while j < len(years):
            y = years[j]
            raw = per_year.get(str(y), "N/A")
            out["fields"][f][str(y)] = _coerce_number_or_na(raw)
            j += 1
        i += 1

    return out

def llm_pick_cash_flow_sections(
    jsonl_path: str,
    top_k: int = 20,
    batch_size: int = 150,
    model: str = "gpt-4o-mini"
) -> List[str]:
    """
    Use an LLM to choose top-k sections likely to contain Cash Flow Statement lines.
    Returns section_ids ranked best->worst.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # load section titles + ids
    sections: List[Dict[str, str]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            sid = rec.get("section_id", "") or ""
            title = rec.get("title", "") or ""
            if sid != "" or title != "":
                sections.append({"section_id": sid, "title": title})

    if len(sections) == 0:
        return []

    def _batches(lst, n):
        i = 0
        while i < len(lst):
            yield lst[i:i+n]
            i += n

    TARGET_FIELDS = [
        "Net Cash Flow from Operations",
        "Net Cash Flow from Investing",
        "Net Cash Flow from Financing",
        "Net Increase/Decrease in Cash",
        "Dividends"
    ]

    scored: List[Tuple[str, float]] = []

    for chunk in _batches(sections, batch_size):
        compact = []
        i = 0
        while i < len(chunk):
            item = chunk[i]
            compact.append({
                "section_id": item["section_id"],
                "title": item["title"][:180]
            })
            i += 1

        system_msg = "You are a precise classifier for annual report sections. Return ONLY a JSON array."
        user_prompt = (
            "You will receive a list of sections (title + section_id). "
            "Select entries most likely to contain Cash Flow Statement lines.\n\n"
            "Strong signals:\n"
            "- 'Consolidated cash flow statement', 'Consolidated statement of cash flows',\n"
            "  'Cash flows from operating activities', 'Cash flows from investing/financing activities',\n"
            "  'Net increase (decrease) in cash and cash equivalents', 'Dividends paid'.\n"
            "Avoid remuneration-only, governance-only, ESG-only, auditor opinion.\n\n"
            "Fields of interest:\n" + "\n".join(f"- {t}" for t in TARGET_FIELDS) + "\n\n"
            "Return ONLY a JSON array of objects:\n"
            "[{\"section_id\": \"...\", \"score\": 0.0..1.0}, ...]\n\n"
            "Sections:\n" + json.dumps(compact, ensure_ascii=False)
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=800,
        )

        raw = (resp.choices[0].message.content or "").strip()
        try:
            arr = _extract_json_array(raw)
            j = 0
            while j < len(arr):
                obj = arr[j]
                sid = str(obj.get("section_id", "")).strip()
                try:
                    sc = float(obj.get("score", 0.0))
                except Exception:
                    sc = 0.0
                if sid != "":
                    scored.append((sid, sc))
                j += 1
        except Exception:
            pass

    if len(scored) == 0:
        # keyword fallback
        KEYWORDS = [
            "cash flow", "statement of cash flows", "cash flows",
            "operating activities", "investing activities", "financing activities",
            "net increase", "net decrease", "cash and cash equivalents",
            "dividends", "dividends paid"
        ]

        def _score_heur(t: str) -> float:
            t2 = t.lower()
            count = 0
            i = 0
            while i < len(KEYWORDS):
                if KEYWORDS[i] in t2:
                    count += 1
                i += 1
            return float(count)

        ranked = sorted(
            sections,
            key=lambda s: _score_heur((s.get("title") or "") + " " + (s.get("section_id") or "")),
            reverse=True
        )
        out = []
        i = 0
        limit = top_k if top_k < len(ranked) else len(ranked)
        while i < limit:
            sid = str(ranked[i].get("section_id", ""))
            if sid != "":
                out.append(sid)
            i += 1
        return out

    # aggregate by max score then take top_k
    best: Dict[str, float] = {}
    i = 0
    while i < len(scored):
        sid, sc = scored[i]
        if sid not in best or sc > best[sid]:
            best[sid] = sc
        i += 1

    ranked_ids = sorted(best.items(), key=lambda x: x[1], reverse=True)
    out = []
    i = 0
    limit = top_k if top_k < len(ranked_ids) else len(ranked_ids)
    while i < limit:
        out.append(ranked_ids[i][0])
        i += 1
    return out

def merge_cash_flow_per_year_priority(
    cf_2024: Dict[str, Any],
    cf_2023: Dict[str, Any],
    years: List[int] = [2024, 2023, 2022],
    debug: bool = False
) -> Dict[str, Any]:
    """
    Global priority: for ALL years, prefer cf_2024; fallback to cf_2023.
    Normalizes to a common multiplier; resolves currency similar to S2.1/S2.2.
    """

    target_fields = [
        "Net Cash Flow from Operations",
        "Net Cash Flow from Investing",
        "Net Cash Flow from Financing",
        "Net Increase/Decrease in Cash",
        "Dividends"
    ]

    def _canon_mult(m: Optional[str]) -> str:
        if m is None:
            return "Units"
        s = str(m).strip().lower()
        if s == "unit" or s == "units":
            return "Units"
        if s == "thousand" or s == "thousands" or s == "k":
            return "Thousands"
        if s == "million" or s == "millions" or s == "mm" or s == "m":
            return "Millions"
        if s == "billion" or s == "billions" or s == "bn" or s == "b":
            return "Billions"
        return "Units"

    def _mult_factor(mult: str) -> float:
        if mult == "Units":
            return 1.0
        if mult == "Thousands":
            return 1_000.0
        if mult == "Millions":
            return 1_000_000.0
        if mult == "Billions":
            return 1_000_000_000.0
        return 1.0

    def _normalize_value_to_multiplier(value: Any, from_mult: str, to_mult: str) -> Any:
        if value is None:
            return "N/A"
        if value == "N/A":
            return "N/A"
        if isinstance(value, str):
            s = value.strip()
            if s.upper() == "N/A":
                return "N/A"
            try:
                value = float(s)
            except Exception:
                return "N/A"
        if isinstance(value, (int, float)):
            f_from = _mult_factor(from_mult)
            f_to = _mult_factor(to_mult)
            if f_to == 0.0:
                return "N/A"
            return float(value) * (f_from / f_to)
        return "N/A"

    def _get_field_value(src: Dict[str, Any], field: str, year: int) -> Any:
        fields = src.get("fields", {})
        per_year = fields.get(field, {})
        return per_year.get(str(year), "N/A")

    mult_2024 = _canon_mult(cf_2024.get("multiplier"))
    mult_2023 = _canon_mult(cf_2023.get("multiplier"))
    if mult_2024 is not None and mult_2024 != "":
        target_multiplier = mult_2024
    else:
        target_multiplier = mult_2023
    if target_multiplier is None or target_multiplier == "":
        target_multiplier = "Units"

    cur_2024 = str(cf_2024.get("currency") or "").strip()
    cur_2023 = str(cf_2023.get("currency") or "").strip()
    if cur_2024 != "" and cur_2023 != "" and cur_2024 != cur_2023:
        target_currency = "MIXED"
    elif cur_2024 != "":
        target_currency = cur_2024
    elif cur_2023 != "":
        target_currency = cur_2023
    else:
        target_currency = "USD"

    merged: Dict[str, Any] = {
        "years": years,
        "multiplier": target_multiplier,
        "currency": target_currency,
        "fields": {}
    }

    i = 0
    while i < len(target_fields):
        field = target_fields[i]
        merged["fields"][field] = {}
        j = 0
        while j < len(years):
            y = years[j]
            v24_raw = _get_field_value(cf_2024, field, y)
            v23_raw = _get_field_value(cf_2023, field, y)

            v24_norm = _normalize_value_to_multiplier(v24_raw, mult_2024, target_multiplier)
            v23_norm = _normalize_value_to_multiplier(v23_raw, mult_2023, target_multiplier)

            if v24_norm != "N/A":
                chosen = v24_norm
            else:
                if v23_norm != "N/A":
                    chosen = v23_norm
                else:
                    chosen = "N/A"

            merged["fields"][field][str(y)] = chosen
            j += 1
        i += 1

    if debug:
        print("[merge-cashflow] multiplier:", target_multiplier)
        print("[merge-cashflow] currency:", target_currency)
        print("[merge-cashflow] policy: 2024 overrides 2023 for all years")

    return merged

def print_cash_flow_table(cf: dict):
    years = [str(y) for y in cf.get("years", [2024, 2023, 2022])]
    header = ["Field"] + years + ["Multiplier", "Currency"]
    print(" | ".join(header))
    print("-" * (len(" | ".join(header))))

    multiplier = cf.get("multiplier", "Units")
    currency = cf.get("currency", "USD")

    order = [
        "Net Cash Flow from Operations",
        "Net Cash Flow from Investing",
        "Net Cash Flow from Financing",
        "Net Increase/Decrease in Cash",
        "Dividends"
    ]

    fields = cf.get("fields", {})
    i = 0
    while i < len(order):
        field = order[i]
        per_year = fields.get(field, {})
        row_values = []
        j = 0
        while j < len(years):
            y = years[j]
            row_values.append(format_financial_cell(per_year.get(y, "N/A")))
            j += 1
        row = [field] + row_values + [multiplier, currency]
        print(" | ".join(row))
        i += 1
 
        
# --------------------------- S 2.4 Cash Flow Statement (2024 + 2023 + 2022) -------------------------------------------       

def _as_num(v):
    if v is None or v == "N/A":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s.upper() == "N/A":
        return None
    # handle '(335.8)' and '1,234'
    neg = s.startswith("(") and s.endswith(")")
    s2 = s.strip("() ").replace(",", "")
    try:
        n = float(s2)
        return -n if neg else n
    except Exception:
        return None

def _pct(num, den, *, allow_zero_den=False):
    """
    Return num/den as percentage (x100). If invalid -> "N/A".
    By default, disallow zero denominator.
    """
    n = _as_num(num)
    d = _as_num(den)
    if n is None or d is None:
        return "N/A"
    if (not allow_zero_den and d == 0) or d is None:
        return "N/A"
    try:
        return (n / d) * 100.0
    except Exception:
        return "N/A"

def _pct_abs(num, den):
    """Percentage using absolute denominator"""
    n = _as_num(num)
    d = _as_num(den)
    if n is None or d is None:
        return "N/A"
    d = abs(d)
    if d == 0:
        return "N/A"
    try:
        return (n / d) * 100.0
    except Exception:
        return "N/A"

def _safe_sub(*args):
    """Return a - b - c ... using numeric coercion; if any None -> 'N/A'."""
    total = None
    for i, v in enumerate(args):
        x = _as_num(v)
        if x is None:
            return "N/A"
        total = x if i == 0 else (total - x)
    return total

def _avg(a, b):
    """Average of two numerics; if any None -> 'N/A'."""
    a1 = _as_num(a)
    b1 = _as_num(b)
    if a1 is None or b1 is None:
        return "N/A"
    return (a1 + b1) / 2.0

def _fmt_percent_value(v):
    """Keep raw number or 'N/A' (the printing step can add % sign)."""
    return v if v == "N/A" else float(v)

def compute_key_metrics_from_tables(
    inc: Dict[str, Any],
    bal: Dict[str, Any],
    cf:  Dict[str, Any],
    years: List[int] = [2024, 2023, 2022]
) -> Dict[str, Any]:
    """
    Build S2.4 Key Financial Metrics from S2.1/S2.2/S2.3 merged tables (no LLM).
    Assumptions to match:
      - Use year-end values when prior-year average is missing.
      - For 2024 and 2023, use two-year averages for asset-based metrics where possible.
      - Treat liabilities and dividends as magnitudes in denominators/numerators (abs()).
      - If any input needed is N/A, output 'N/A' for that metric/year.
    """
    fields = {
        "Gross Margin": {},       
        "Operating Margin": {},
        "Net Profit Margin": {},
        "Current Ratio": {},
        "Quick Ratio": {},
        "Debt-to-Equity": {},
        "Interest Coverage": {},
        "Asset Turnover": {},
        "Return on Equity": {},
        "Return on Assets": {},
        "Effective Tax Rate": {},
        "Dividend Payout Ratio": {},
    }

    def inc_val(field, y): return (inc.get("fields", {}).get(field, {}) or {}).get(str(y), "N/A")
    def bal_val(field, y): return (bal.get("fields", {}).get(field, {}) or {}).get(str(y), "N/A")
    def cf_val(field, y):  return (cf.get("fields",  {}).get(field,  {}) or {}).get(str(y), "N/A")

    # Pre-calc lookups
    def rev(y):      return inc_val("Revenue", y)
    def opinc(y):    return inc_val("Operating Income", y)
    def netprof(y):  return inc_val("Net Profit", y)
    def pbt(y):      return inc_val("Income before income taxes", y)
    def tax(y):      return inc_val("Income tax expense(benefit)", y)
    def intr(y):     return inc_val("Interest Expense", y)

    def tot_assets(y): return bal_val("Total Assets", y)
    def cur_assets(y): return bal_val("Current Assets", y)
    def cur_liab(y):   return bal_val("Current Liabilities", y)
    def inv(y):        return bal_val("Inventories", y)
    def prepaids(y):   return bal_val("Prepaid Expenses", y)  # may be N/A -> treat as 0 in quick ratio
    def tot_liab(y):   return bal_val("Total Liabilities", y)
    def equity(y):     return bal_val("Shareholders' Equity", y)

    def divs(y): return cf_val("Dividends", y)  # usually negative in cash flow

    # For averages (asset/equity) we’ll use: avg(current year, previous year) if previous exists, else year-end.
    def avg_assets(y):
        idx = years.index(y)
        if idx + 1 < len(years):  # previous year exists in the list
            prev = years[idx + 1]
            a = _avg(tot_assets(y), tot_assets(prev))
            return a if a != "N/A" else _as_num(tot_assets(y))
        return _as_num(tot_assets(y))

    def avg_equity(y):
        idx = years.index(y)
        if idx + 1 < len(years):
            prev = years[idx + 1]
            a = _avg(equity(y), equity(prev))
            return a if a != "N/A" else _as_num(equity(y))
        return _as_num(equity(y))

    for y in years:
        # 1) Gross Margin -> not computable with current inputs
        fields["Gross Margin"][str(y)] = "N/A"

        # 2) Operating Margin = Operating Income / Revenue
        fields["Operating Margin"][str(y)] = _fmt_percent_value(_pct(opinc(y), rev(y)))

        # 3) Net Profit Margin = Net Profit / Revenue
        fields["Net Profit Margin"][str(y)] = _fmt_percent_value(_pct(netprof(y), rev(y)))

        # 4) Current Ratio = Current Assets / Current Liabilities  (use abs(liabilities))
        fields["Current Ratio"][str(y)] = _fmt_percent_value(_pct_abs(cur_assets(y), cur_liab(y)))

        # 5) Quick Ratio = (Current Assets - Inventories - Prepaids) / Current Liabilities    
        inv_v = inv(y)
        inv_v = 0.0 if inv_v == "N/A" else inv_v
        pre  = prepaids(y)
        pre  = 0.0 if pre == "N/A" else pre
        quick_num = _safe_sub(cur_assets(y), inv_v, pre)
        fields["Quick Ratio"][str(y)] = _fmt_percent_value(_pct_abs(quick_num, cur_liab(y)))

        # 6) Debt-to-Equity = Total Liabilities / Shareholders' Equity (use magnitudes)
        liab_val = _as_num(tot_liab(y))
        eq_val   = _as_num(equity(y))
        liab_mag = abs(liab_val) if liab_val is not None else None
        eq_mag   = abs(eq_val)   if eq_val   is not None else None
        fields["Debt-to-Equity"][str(y)] = _fmt_percent_value(_pct(liab_mag, eq_mag))

        # 7) Interest Coverage ≈ Operating Income / Interest Expense
        # If interest expense is negative (rare), use magnitude.
        intr_mag = abs(_as_num(intr(y))) if _as_num(intr(y)) is not None else None
        fields["Interest Coverage"][str(y)] = _fmt_percent_value(_pct(opinc(y), intr_mag))

        # 8) Asset Turnover = Revenue / Average Total Assets (2-year avg if available, else year-end)
        at_den = avg_assets(y)
        fields["Asset Turnover"][str(y)] = _fmt_percent_value(_pct(rev(y), at_den))

        # 9) Return on Equity = Net Profit / Average Equity
        roe_den = avg_equity(y)
        fields["Return on Equity"][str(y)] = _fmt_percent_value(_pct(netprof(y), roe_den))

        # 10) Return on Assets = Net Profit / Average Total Assets
        roa_den = avg_assets(y)
        fields["Return on Assets"][str(y)] = _fmt_percent_value(_pct(netprof(y), roa_den))

        # 11) Effective Tax Rate = Income Tax Expense / Income before income taxes
        # Use magnitudes in denominator if needed (usually positive).
        pbt_mag = abs(_as_num(pbt(y))) if _as_num(pbt(y)) is not None else None
        fields["Effective Tax Rate"][str(y)] = _fmt_percent_value(_pct(tax(y), pbt_mag))

        # 12) Dividend Payout Ratio = Dividends / Net Profit  (use |Dividends|)
        div_mag = abs(_as_num(divs(y))) if _as_num(divs(y)) is not None else None
        fields["Dividend Payout Ratio"][str(y)] = _fmt_percent_value(_pct(div_mag, netprof(y)))

    return {
        "years": years,
        "multiplier": "Percent",
        "currency": "N/A",
        "fields": fields,
    }

# Optional: pretty printer (adds % sign)
def print_key_metrics_table(metrics: dict):
    years = [str(y) for y in metrics["years"]]
    header = ["Field"] + years + ["Multiplier", "Currency"]
    print(" | ".join(header))
    print("-" * (len(" | ".join(header))))
    for field, per_year in metrics["fields"].items():
        row_vals = []
        for y in years:
            v = per_year.get(y, "N/A")
            if v == "N/A":
                row_vals.append("N/A")
            else:
                row_vals.append(f"{float(v):.2f}%")
        print(" | ".join([field] + row_vals + [metrics["multiplier"], metrics["currency"]]))


# --------------------------- S 2.5 Cash Flow Statement (2024 + 2023 + 2022) -------------------------------------------       

def llm_pick_operating_performance_sections(
    jsonl_path: str,
    top_k: int = 10,
    batch_size: int = 150,
    model: str = "gpt-4o-mini"
) -> List[str]:
    """
    Use an LLM to choose top-k sections likely to contain segment/geography revenue breakdowns:
    - 'Sensors & Information', 'Countermeasures & Energetics'
    - 'Revenue by destination' / 'Geographic split' (UK, US, Europe, Asia Pacific, Rest of the world)
    Returns list of section_id strings ranked best->worst.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sections: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
            except Exception:
                continue
            sid = rec.get("section_id", "") or ""
            title = rec.get("title", "") or ""
            if sid or title:
                sections.append({"section_id": sid, "title": title})

    if not sections:
        return []

    def _batches(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    scored: List[Tuple[str, float]] = []
    for chunk in _batches(sections, batch_size):
        compact = [{"section_id": s["section_id"], "title": s["title"][:180]} for s in chunk]

        system_msg = (
            "You are a precise classifier for annual report sections. "
            "Return JSON arrays only."
        )

        user_prompt = f"""
You will receive annual report sections (title + section_id).
Select entries most likely to contain:
- Segment revenue: "Sensors & Information", "Countermeasures & Energetics", by year totals
- Geographic revenue: lines like "UK", "US", "Europe", "Asia Pacific", "Rest of the world"
Also consider phrases: "revenue by destination", "geographical analysis", "operating segments".

Return a JSON array of objects: {{ "section_id": string, "score": 0..1 }}

Sections:
{json.dumps(compact, ensure_ascii=False)}
""".strip()

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=800,
        )

        raw = (resp.choices[0].message.content or "").strip()
        try:
            arr = _extract_json_array(raw)
            for obj in arr:
                sid = str(obj.get("section_id", "")).strip()
                sc = float(obj.get("score", 0.0))
                if sid:
                    scored.append((sid, sc))
        except Exception:
            pass

    if not scored:
        # simple keyword fallback
        KEYWORDS = [
            "sensors & information", "countermeasures & energetics",
            "revenue by destination", "geographic", "geographical",
            "uk", "united kingdom", "us", "united states", "europe",
            "asia pacific", "rest of the world", "segment revenue", "operating segments"
        ]
        def _score_heuristic(t: str) -> float:
            t2 = t.lower()
            return float(sum(1 for k in KEYWORDS if k in t2))

        ranked = sorted(
            sections,
            key=lambda s: _score_heuristic((s.get("title") or "") + " " + (s.get("section_id") or "")),
            reverse=True
        )
        return [str(s.get("section_id", "")) for s in ranked[:top_k] if s.get("section_id")]

    best: Dict[str, float] = {}
    for sid, sc in scored:
        if sid not in best or sc > best[sid]:
            best[sid] = sc

    ranked_ids = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in ranked_ids[:top_k]]

def llm_pick_operating_performance_sections(
    jsonl_path: str,
    top_k: int = 10,
    batch_size: int = 150,
    model: str = "gpt-4o-mini"
) -> List[str]:
    """
    Use an LLM to choose top-k sections likely to contain segment/geography revenue breakdowns:
    - 'Sensors & Information', 'Countermeasures & Energetics'
    - 'Revenue by destination' / 'Geographic split' (UK, US, Europe, Asia Pacific, Rest of the world)
    Returns list of section_id strings ranked best->worst.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sections: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
            except Exception:
                continue
            sid = rec.get("section_id", "") or ""
            title = rec.get("title", "") or ""
            if sid or title:
                sections.append({"section_id": sid, "title": title})

    if not sections:
        return []

    def _batches(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    scored: List[Tuple[str, float]] = []
    for chunk in _batches(sections, batch_size):
        compact = [{"section_id": s["section_id"], "title": s["title"][:180]} for s in chunk]

        system_msg = (
            "You are a precise classifier for annual report sections. "
            "Return JSON arrays only."
        )

        user_prompt = f"""
            You will receive annual report sections (title + section_id).
            Select entries most likely to contain:
            - Segment revenue: "Sensors & Information", "Countermeasures & Energetics", by year totals
            - Geographic revenue: lines like "UK", "US", "Europe", "Asia Pacific", "Rest of the world"
            Also consider phrases: "revenue by destination", "geographical analysis", "operating segments".

            Return a JSON array of objects: {{ "section_id": string, "score": 0..1 }}

            Sections:
            {json.dumps(compact, ensure_ascii=False)}
            """.strip()

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=800,
        )

        raw = (resp.choices[0].message.content or "").strip()
        try:
            arr = _extract_json_array(raw)
            for obj in arr:
                sid = str(obj.get("section_id", "")).strip()
                sc = float(obj.get("score", 0.0))
                if sid:
                    scored.append((sid, sc))
        except Exception:
            pass

    if not scored:
        # simple keyword fallback
        KEYWORDS = [
            "sensors & information", "countermeasures & energetics",
            "revenue by destination", "geographic", "geographical",
            "uk", "united kingdom", "us", "united states", "europe",
            "asia pacific", "rest of the world", "segment revenue", "operating segments"
        ]
        def _score_heuristic(t: str) -> float:
            t2 = t.lower()
            return float(sum(1 for k in KEYWORDS if k in t2))

        ranked = sorted(
            sections,
            key=lambda s: _score_heuristic((s.get("title") or "") + " " + (s.get("section_id") or "")),
            reverse=True
        )
        return [str(s.get("section_id", "")) for s in ranked[:top_k] if s.get("section_id")]

    best: Dict[str, float] = {}
    for sid, sc in scored:
        if sid not in best or sc > best[sid]:
            best[sid] = sc

    ranked_ids = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in ranked_ids[:top_k]]

def _operating_perf_prompt(years: List[int], text: str) -> str:
    years_csv = ", ".join(str(y) for y in years)
    return f"""
        You are given text snippets from a company's annual reports that include segment and geographic revenue tables.

        TASK
        Extract S2.5 Operating Performance for years [{years_csv}] with these two fields only:
        - "Revenue by Product/Service"
        - "Revenue by Geographic Region"

        STRICT RULES
        - Use ONLY the provided text. If a year’s breakdown cannot be found, return "N/A" for that year.
        - For Product/Service, prefer a concise, single-line summary per year in this pattern (if GBP in millions):
        "Sensors & Information: £212.0m, Countermeasures & Energetics: £298.4m"
        - For Geographic Region, one line per year, pattern:
        "UK: £229.2m, US: £172.6m, Europe: £86.0m, Asia Pacific: £16.7m, Rest of the world: £5.9m"
        - Preserve '£' symbol and 'm' if shown as millions; round as the text shows.
        - Do NOT invent categories; if something is missing in text for a year, put "N/A".

        OUTPUT (JSON ONLY, no extra text):
        {{
        "years": [{years_csv}],
        "multiplier": "Millions",
        "currency": "GBP",
        "fields": {{
            "Revenue by Product/Service": {{"{years[0]}": "N/A"}},
            "Revenue by Geographic Region": {{}}
        }}
        }}

        TEXT:
        {text}
        """.strip()

def extract_operating_performance(oper_text: str, years: List[int] = [2024, 2023, 2022]) -> Dict[str, Any]:
    """
    LLM extraction for S2.5 operating performance (two string fields).
    Returns:
    {
      "years": [...],
      "multiplier": "Millions",
      "currency": "GBP",
      "fields": {
        "Revenue by Product/Service": { "2024": "Sensors & Information: £212.0m, ..." , ... },
        "Revenue by Geographic Region": { "2024": "UK: £229.2m, US: ...", ... }
      }
    }
    """
    if not oper_text:
        return {
            "years": years,
            "multiplier": "Millions",
            "currency": "GBP",
            "fields": {
                "Revenue by Product/Service": {str(y): "N/A" for y in years},
                "Revenue by Geographic Region": {str(y): "N/A" for y in years},
            }
        }

    prompt = _operating_perf_prompt(years, oper_text)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You extract segment/geography revenue strings in strict JSON and never invent categories."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=800
        )
        data = _safe_json_from_llm(resp.choices[0].message.content.strip())
    except Exception as e:
        print(f"[S2.5] LLM error: {e}")
        data = {}

    out = {
        "years": years,
        "multiplier": (data.get("multiplier") or "Millions").strip(),
        "currency": (data.get("currency") or "GBP").strip(),
        "fields": {
            "Revenue by Product/Service": {},
            "Revenue by Geographic Region": {}
        }
    }
    fields = data.get("fields") or {}
    for key in ["Revenue by Product/Service", "Revenue by Geographic Region"]:
        per_year = fields.get(key, {}) or {}
        for y in years:
            out["fields"][key][str(y)] = per_year.get(str(y), "N/A") or "N/A"

    return out

def merge_operating_performance_per_year_priority(
    op_2024: Dict[str, Any],
    op_2023: Dict[str, Any],
    years: List[int] = [2024, 2023, 2022],
    debug: bool = False
) -> Dict[str, Any]:
    """
    Merge two S2.5 dicts:
      For ALL years, prefer op_2024; if that year's value is 'N/A', fall back to op_2023.
    """
    fields = ["Revenue by Product/Service", "Revenue by Geographic Region"]

    # Multiplier/currency: pick 2024's if present, else 2023's
    mult = (op_2024.get("multiplier") or op_2023.get("multiplier") or "Millions").strip()
    curr = (op_2024.get("currency") or op_2023.get("currency") or "GBP").strip()

    merged = {
        "years": years,
        "multiplier": mult,
        "currency": curr,
        "fields": {k: {} for k in fields}
    }

    for k in fields:
        for y in years:
            v24 = (op_2024.get("fields", {}).get(k, {}) or {}).get(str(y), "N/A")
            if v24 and v24 != "N/A":
                merged["fields"][k][str(y)] = v24
            else:
                v23 = (op_2023.get("fields", {}).get(k, {}) or {}).get(str(y), "N/A")
                merged["fields"][k][str(y)] = v23 if v23 else "N/A"

    if debug:
        print("[merge-oper] multiplier:", mult)
        print("[merge-oper] currency:", curr)
        print("[merge-oper] policy: 2024 overrides 2023 for all years")

    return merged

def print_operating_performance_table(op: Dict[str, Any]) -> None:
    years = [str(y) for y in op["years"]]
    header = ["Field"] + years + ["Multiplier", "Currency"]
    print(" | ".join(header))
    print("-" * (len(" | ".join(header))))
    for field, per_year in op["fields"].items():
        row = [field] + [per_year.get(y, "N/A") for y in years] + [op["multiplier"], op["currency"]]
        print(" | ".join(row))
    

# --------------------------- S 3.1 Profitability Analysis -------------------------------------------

def _round_floats_in_fields(table: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optional: make numeric fields shorter for prompt readability (no effect on correctness).
    Leaves strings ('N/A') untouched.
    """
    out = {"years": table.get("years", []), "multiplier": table.get("multiplier"), "currency": table.get("currency"), "fields": {}}
    fields = table.get("fields", {}) or {}
    for k, per_year in fields.items():
        row = {}
        for y, v in (per_year or {}).items():
            try:
                if isinstance(v, (int, float)):
                    row[y] = float(f"{float(v):.4g}")  # compact
                else:
                    # keep N/A / strings verbatim
                    row[y] = v
            except Exception:
                row[y] = v
        out["fields"][k] = row
    return out

def _serialize_operating(s25_operating: Optional[Dict[str, Dict[str, str]]]) -> Dict[str, Dict[str, str]]:
    """
    Expecting:
      {
        "Revenue by Product/Service": {"2024": "...", "2023": "...", "2022": "..."},
        "Revenue by Geographic Region": {"2024": "...", "2023": "...", "2022": "..."}
      }
    Returns {} if None.
    """
    return s25_operating or {}

def _s31_prompt(
    inc: Dict[str, Any],
    bal: Dict[str, Any],
    cf: Dict[str, Any],
    metrics: Dict[str, Any],
    operating: Dict[str, Dict[str, str]],
    years: List[int],
) -> str:
    """
    Build a STRICT JSON-only prompt for S3.1 based only on S2 tables.
    """
    years_csv = ", ".join(str(y) for y in years)
    inc_c = _round_floats_in_fields(inc)
    bal_c = _round_floats_in_fields(bal)
    cf_c  = _round_floats_in_fields(cf)
    met_c = _round_floats_in_fields(metrics)

    operating_json = json.dumps(operating, ensure_ascii=False)
    return f"""
        You are given **only** the company's Section 2 tables for years [{years_csv}]:
        - S2.1: Income Statement (consolidated)
        - S2.2: Balance Sheet (consolidated)
        - S2.3: Cash Flow Statement (consolidated)
        - S2.4: Key Financial Metrics (derived from the above tables; already computed)
        - S2.5: Operating Performance (revenue splits by product/service and geography; optional)

        Your task: Produce S3.1 **Profitability Analysis** as three short, evidence-based paragraphs following this rubric:

        1) "Revenue & Direct-Cost Dynamics"
        - Key metrics to reference (when available in the inputs): Revenue growth, Gross Margin, Revenue by Product/Service, Revenue by Geographic Region.
        - Summarize revenue trend **from 2022→2023→2024** using specific values from S2.1.
        - If Gross Margin in S2.4 is N/A, explicitly note that it’s not available.
        - If S2.5 splits are present, briefly discuss both the product/service and geographic composition:
            - For Product/Service: summarize the relative contribution of each segment 
            - For Geographic Region: explicitly mention which regions grew or declined from 2023 to 2024 (e.g., “growth in the UK and Europe, while the US saw a slight decline”).
        - Always include at least one sentence referencing geographic performance if region data exists in S2.5.
        - Stay within 2–4 sentences.

        2) "Operating Efficiency"
        - Key metric: Operating Margin (from S2.4).
        - Describe the change **2022→2023→2024** (e.g., “decreased … then increased …”).
        - Quote the percentages with two decimals and the year.
        - 1–2 sentences.

        3) "External & One-Off Impact"
        - Key metrics: Effective Tax Rate (from S2.4) and any visible non-recurring items **seen in the S2 tables only**.
        - If no non-recurring items can be inferred from the provided tables, say so plainly.
        - Briefly relate movements in the effective tax rate to movements in net profit margin (from S2.4), without speculating beyond the tables.
        - 2–3 sentences.

        STRICT RULES
        - Use ONLY the numbers in the provided S2 tables below. Do NOT invent or import outside information.
        - Be concise, neutral, and analytical (no marketing language).
        - Where you quote numbers:
        - Use the currency in S2.1 for revenue (e.g., “£510.4m” or “$60,922m”) and add “m” (millions) since inputs are in millions.
        - Use two decimals for percentages (e.g., “11.38%”).
        - If a requested metric is N/A in the inputs, state it’s not available rather than guessing.
        - Return **JSON ONLY** with exactly this schema:

        {{
        "perspectives": {{
            "Revenue & Direct-Cost Dynamics": "<2-4 sentences>",
            "Operating Efficiency": "<1-2 sentences>",
            "External & One-Off Impact": "<2-3 sentences>"
        }}
        }}

        DATA (use these only):
        S2.1 Income Statement (JSON): {json.dumps(inc_c, ensure_ascii=False)}
        S2.2 Balance Sheet   (JSON): {json.dumps(bal_c, ensure_ascii=False)}
        S2.3 Cash Flow       (JSON): {json.dumps(cf_c, ensure_ascii=False)}
        S2.4 Key Metrics     (JSON): {json.dumps(met_c, ensure_ascii=False)}
        S2.5 Operating Perf. (JSON): {operating_json}
        """.strip()

def llm_build_profitability_analysis(
    report: "CompanyReport",
    merged_income: Dict[str, Any],
    merged_balance: Dict[str, Any],
    merged_cashflow: Dict[str, Any],
    derived_metrics: Dict[str, Any],
    s25_operating: Optional[Dict[str, Dict[str, str]]] = None,
    years: List[int] = [2024, 2023, 2022],
    model: str = "gpt-4-turbo",
) -> Dict[str, str]:
    """
    Calls the LLM to generate S3.1 text from Section 2 tables (no hardcoding).
    Writes results into report.profitability_analysis and returns the dict.
    """
    # Currency symbol for revenue phrases (default GBP if missing)
    cur = (merged_income.get("currency") or "GBP").strip().upper()
    # Build prompt
    prompt = _s31_prompt(
        inc=merged_income,
        bal=merged_balance,
        cf=merged_cashflow,
        metrics=derived_metrics,
        operating=_serialize_operating(s25_operating),
        years=years,
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You write concise, accurate financial analysis. Always return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=700,
        )
        data = _safe_json_from_llm(resp.choices[0].message.content.strip())
    except Exception as e:
        # Fallback: very short deterministic placeholders (still respect rubric)
        data = {
            "perspectives": {
                "Revenue & Direct-Cost Dynamics": "Revenue information could not be generated due to an LLM error.",
                "Operating Efficiency": "Operating margin commentary is unavailable.",
                "External & One-Off Impact": "Tax rate and non-recurring items commentary is unavailable.",
            }
        }

    pers = (data.get("perspectives") or {})
    rev_direct = pers.get("Revenue & Direct-Cost Dynamics", "N/A")
    op_eff     = pers.get("Operating Efficiency", "N/A")
    ext_oneoff = pers.get("External & One-Off Impact", "N/A")

    # Write into dataclass
    report.profitability_analysis.revenue_direct_cost_dynamics = rev_direct
    report.profitability_analysis.operating_efficiency = op_eff
    report.profitability_analysis.external_oneoff_impact = ext_oneoff

    return {
        "Revenue & Direct-Cost Dynamics": rev_direct,
        "Operating Efficiency": op_eff,
        "External & One-Off Impact": ext_oneoff,
    }


# --------------------------- S 3.2 Financial Performance Summary -------------------------------------------
def _s32_round_fields(table: Dict[str, Any]) -> Dict[str, Any]:
    """Compact numerics for prompt readability; keep 'N/A' as-is."""
    out = {"years": table.get("years", []), "multiplier": table.get("multiplier"), "currency": table.get("currency"), "fields": {}}
    fields = table.get("fields", {}) or {}
    for k, per_year in fields.items():
        row = {}
        for y, v in (per_year or {}).items():
            try:
                if isinstance(v, (int, float)):
                    row[y] = float(f"{float(v):.4g}")
                else:
                    row[y] = v
            except Exception:
                row[y] = v
        out["fields"][k] = row
    return out

def _s32_operating_to_json(operating: Optional[Dict[str, Dict[str, str]]]) -> str:
    """
    Expect:
      {
        "Revenue by Product/Service": {"2024": "...", "2023": "...", "2022": "..."},
        "Revenue by Geographic Region": {"2024": "...", "2023": "...", "2022": "..."}
      }
    """
    return json.dumps(operating or {}, ensure_ascii=False)

def _s32_prompt(
    inc: Dict[str, Any],
    bal: Dict[str, Any],
    cf: Dict[str, Any],
    metrics: Dict[str, Any],
    operating: Optional[Dict[str, Dict[str, str]]],
    years: List[int],
    company_name: Optional[str] = None,
    establishment_date: Optional[str] = None,
    company_hq: Optional[str] = None,
) -> str:
    years_csv = ", ".join(str(y) for y in years)
    inc_c = _s32_round_fields(inc)
    bal_c = _s32_round_fields(bal)
    cf_c  = _s32_round_fields(cf)
    met_c = _s32_round_fields(metrics)
    op_js = _s32_operating_to_json(operating)

    # --- Compose company descriptor (optional fields included only if present) ---
    company_info_lines = []
    if company_name:
        company_info_lines.append(f"Company: {company_name}")
    if establishment_date:
        company_info_lines.append(f"Established: {establishment_date}")
    if company_hq:
        company_info_lines.append(f"Headquarters: {company_hq}")
    company_info = " | ".join(company_info_lines) if company_info_lines else "Company information unavailable"

    return f"""
        You are analyzing financial data for **{company_name or "the company"}**.
        {company_info}
        You are given ONLY the company’s Section 2 tables for years [{years_csv}]:
        - S2.1: Income Statement
        - S2.2: Balance Sheet
        - S2.3: Cash Flow Statement
        - S2.4: Key Financial Metrics (derived from the above)
        - S2.5: Operating Performance (optional: product/service & geography splits)

        Write **S3.2: Financial Performance Summary** as 5 perspectives, each with a concise 2024 Report and 2023 Report.
        Use numbers strictly from the provided tables (no outside info, no guessing).

        Perspectives:
        1) Comprehensive financial health
        - Reference revenue (S2.1), total assets/liabilities/equity (S2.2). Mention leverage/liquidity only if supported by S2.4 ratios.
        2) Profitability and earnings quality
        - Use Net Profit Margin, Operating Margin, ROE, ROA, Effective Tax Rate (S2.4). Keep it succinct and comparative vs prior year.
        3) Operational efficiency
        - Use Operating Margin (S2.4), Asset Turnover (S2.4), and Net Cash Flow from Operations (S2.3). Keep to 1–3 sentences.
        4) Financial risk identification and early warning
        - Use Debt-to-Equity, Current Ratio, Quick Ratio, Interest Coverage (S2.4). Call out material moves.
        5) Future financial performance projection
        - Ground only in S2 tables: revenue trends (S2.1), cash used in investing (S2.3), S2.5 mixes (if provided), dividend payout ratio (S2.4).
        - Forward-looking phrasing must be conditional and tied directly to observed trends (no speculation beyond the tables).

        Formatting rules:
        - Be neutral, compact, and analytical (no marketing tone).
        - Use the currency from S2.1 (e.g., “£510.4m” or “$60,922m”) and include “m” (millions).
        - Percentages must use two decimals (e.g., 11.38%).
        - If a referenced metric is N/A, say it’s not available rather than inferring.

        Return **JSON ONLY** in this schema:
        {{
        "summary": {{
            "Comprehensive financial health": {{"2024 Report": "<text>", "2023 Report": "<text>"}},
            "Profitability and earnings quality": {{"2024 Report": "<text>", "2023 Report": "<text>"}},
            "Operational efficiency": {{"2024 Report": "<text>", "2023 Report": "<text>"}},
            "Financial risk identification and early warning": {{"2024 Report": "<text>", "2023 Report": "<text>"}},
            "Future financial performance projection": {{"2024 Report": "<text>", "2023 Report": "<text>"}}
        }}
        }}

        DATA (use these):
        S2.1 Income Statement: {json.dumps(inc_c, ensure_ascii=False)}
        S2.2 Balance Sheet: {json.dumps(bal_c, ensure_ascii=False)}
        S2.3 Cash Flow: {json.dumps(cf_c, ensure_ascii=False)}
        S2.4 Key Metrics: {json.dumps(met_c, ensure_ascii=False)}
        S2.5 Operating Performance: {op_js}
        """.strip()

def llm_build_financial_performance_summary(
    report: "CompanyReport",
    merged_income: Dict[str, Any],
    merged_balance: Dict[str, Any],
    merged_cashflow: Dict[str, Any],
    derived_metrics: Dict[str, Any],
    s25_operating: Optional[Dict[str, Dict[str, str]]] = None,
    years: List[int] = [2024, 2023, 2022],
    company_name: Optional[str] = None,
    establishment_date: Optional[str] = None,
    company_hq: Optional[str] = None,
    model: str = "gpt-4-turbo",
) -> Dict[str, Dict[str, str]]:
    """
    Build S3.2 using the LLM, grounded strictly in S2.1–S2.5. Writes into
    report.financial_performance_summary.*.report_2024 / report_2023.
    Returns the same dict for convenience.
    """
    prompt = _s32_prompt(
        inc=merged_income,
        bal=merged_balance,
        cf=merged_cashflow,
        metrics=derived_metrics,
        operating=s25_operating,
        years=years,
        company_name=company_name,
        establishment_date=establishment_date,
        company_hq=company_hq,
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You write concise, accurate financial analysis and return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=1200,
        )
        data = _safe_json_from_llm(resp.choices[0].message.content.strip())
    except Exception as e:
        data = {"summary": {}}

    summary = data.get("summary", {}) or {}

    def _get(pair: Dict[str, str], k: str) -> str:
        return (pair or {}).get(k, "N/A")

    # 1) Comprehensive financial health
    cfh = summary.get("Comprehensive financial health", {})
    report.financial_performance_summary.comprehensive_financial_health.report_2024 = _get(cfh, "2024 Report")
    report.financial_performance_summary.comprehensive_financial_health.report_2023 = _get(cfh, "2023 Report")

    # 2) Profitability and earnings quality
    peq = summary.get("Profitability and earnings quality", {})
    report.financial_performance_summary.profitability_earnings_quality.report_2024 = _get(peq, "2024 Report")
    report.financial_performance_summary.profitability_earnings_quality.report_2023 = _get(peq, "2023 Report")

    # 3) Operational efficiency
    oe = summary.get("Operational efficiency", {})
    report.financial_performance_summary.operational_efficiency.report_2024 = _get(oe, "2024 Report")
    report.financial_performance_summary.operational_efficiency.report_2023 = _get(oe, "2023 Report")

    # 4) Financial risk identification and early warning
    fri = summary.get("Financial risk identification and early warning", {})
    report.financial_performance_summary.financial_risk_identification.report_2024 = _get(fri, "2024 Report")
    report.financial_performance_summary.financial_risk_identification.report_2023 = _get(fri, "2023 Report")

    # 5) Future financial performance projection
    ffp = summary.get("Future financial performance projection", {})
    report.financial_performance_summary.future_financial_performance_projection.report_2024 = _get(ffp, "2024 Report")
    report.financial_performance_summary.future_financial_performance_projection.report_2023 = _get(ffp, "2023 Report")

    return {
        "Comprehensive financial health": {
            "2024 Report": report.financial_performance_summary.comprehensive_financial_health.report_2024,
            "2023 Report": report.financial_performance_summary.comprehensive_financial_health.report_2023,
        },
        "Profitability and earnings quality": {
            "2024 Report": report.financial_performance_summary.profitability_earnings_quality.report_2024,
            "2023 Report": report.financial_performance_summary.profitability_earnings_quality.report_2023,
        },
        "Operational efficiency": {
            "2024 Report": report.financial_performance_summary.operational_efficiency.report_2024,
            "2023 Report": report.financial_performance_summary.operational_efficiency.report_2023,
        },
        "Financial risk identification and early warning": {
            "2024 Report": report.financial_performance_summary.financial_risk_identification.report_2024,
            "2023 Report": report.financial_performance_summary.financial_risk_identification.report_2023,
        },
        "Future financial performance projection": {
            "2024 Report": report.financial_performance_summary.future_financial_performance_projection.report_2024,
            "2023 Report": report.financial_performance_summary.future_financial_performance_projection.report_2023,
        },
    }


# --------------------------- S 3.3 Business Competitiveness (2024 + 2023)-------------------------------------------

def _compact_s1(report: "CompanyReport") -> Dict[str, Any]:
    """Prepare compact Section 1 context for the LLM (Basic Info, Core Competencies, Mission/Vision)."""
    s1 = {
        "basic_info": {
            "company_name": report.basic_info.company_name,
            "establishment_date": report.basic_info.establishment_date,
            "headquarters_location": report.basic_info.headquarters_location,
        },
        "core_competencies": {
            "innovation_advantages": {
                "2024": report.core_competencies.innovation_advantages.report_2024,
                "2023": report.core_competencies.innovation_advantages.report_2023,
            },
            "product_advantages": {
                "2024": report.core_competencies.product_advantages.report_2024,
                "2023": report.core_competencies.product_advantages.report_2023,
            },
            "brand_recognition": {
                "2024": report.core_competencies.brand_recognition.report_2024,
                "2023": report.core_competencies.brand_recognition.report_2023,
            },
            "reputation_ratings": {
                "2024": report.core_competencies.reputation_ratings.report_2024,
                "2023": report.core_competencies.reputation_ratings.report_2023,
            },
        },
        "mission_vision": {
            "mission_statement": report.mission_vision.mission_statement,
            "vision_statement": report.mission_vision.vision_statement,
            "core_values": report.mission_vision.core_values,
        },
    }
    return s1

def llm_pick_competitiveness_sections(
    jsonl_path: str,
    top_k: int = 20,
    batch_size: int = 150,
    model: str = "gpt-4o-mini"
) -> List[str]:
    """
    Rank sections likely to describe Business Model / Market Position.
    Signals (titles/ids): "business model", "strategy", "operating model", "go-to-market",
    "market share", "competitive landscape", "market position", "industry leadership",
    "customers", "segments", "IFRS 8 Operating Segments", "MD&A" overview, etc.
    Returns ranked section_ids.
    """
    # Lightweight JSONL scan of titles
    sections: List[Dict[str, str]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
            except Exception:
                continue
            sid = (rec.get("section_id") or "").strip()
            title = (rec.get("title") or "").strip()
            if sid or title:
                sections.append({"section_id": sid, "title": title})

    if not sections:
        return []

    def _batches(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    TARGET_CUES = [
        "business model","operating model","go-to-market","value proposition",
        "market position","market share","competitive landscape","competition",
        "industry leadership","segment overview","operating segments","IFRS 8",
        "strategy","strategic priorities","MD&A","management discussion and analysis",
        "products and services","customers","geographic markets","niche markets",
    ]

    # Ask LLM to score section titles for relevance
    scored: List[Tuple[str,float]] = []
    for chunk in _batches(sections, batch_size):
        compact = [{"section_id": s["section_id"], "title": s["title"][:180]} for s in chunk]
        system_msg = "You rank annual report sections for 'Business Model' and 'Market Position' relevance. Return JSON array only."
        user_prompt = f"""
            Rank the following sections by their likelihood to contain **Business Model** and/or **Market Position** info.
            Return ONLY a JSON array of objects: [{{"section_id": str, "score": number}}], score 0.0..1.0.

            Signals to favor: {", ".join(TARGET_CUES)}.

            Sections:
            {json.dumps(compact, ensure_ascii=False)}
            """.strip()

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":system_msg},{"role":"user","content":user_prompt}],
                temperature=0,
                max_tokens=600
            )
            raw = (resp.choices[0].message.content or "").strip()
            arr = json.loads(raw) if raw.startswith("[") else []
            for obj in arr:
                sid = str(obj.get("section_id","")).strip()
                sc = float(obj.get("score",0.0))
                if sid:
                    scored.append((sid, sc))
        except Exception:
            # fallback heuristic if batch fails
            pass

    if not scored:
        # Heuristic fallback if LLM ranking fails
        KEYS = [k.lower() for k in TARGET_CUES]
        def _score(title: str) -> float:
            t = (title or "").lower()
            return sum(1 for k in KEYS if k in t)
        ranked = sorted(sections, key=lambda s: _score(s.get("title","")), reverse=True)
        return [s.get("section_id","") for s in ranked[:top_k] if s.get("section_id")]

    # de-dup by max score
    best: Dict[str,float] = {}
    for sid, sc in scored:
        if sid and (sid not in best or sc > best[sid]):
            best[sid] = sc

    ranked_ids = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return [sid for sid,_ in ranked_ids[:top_k]]

def _s33_prompt_builder_one_year_s1_only(
    s1_context: dict,
    windows_text: str,
    year: int,
    currency_label: str = "m",
) -> str:
    """
    Strict prompt to extract Business Model + Market Position for a single year,
    grounded in Section 1 ONLY + the year’s windows text.
    """
    return f"""
        You will synthesize **S3.3: Business Competitiveness** for **{year}** using ONLY:
        - Section 1 (Basic Info, Core Competencies, Mission/Vision)
        - High-relevance text windows from the company's {year} annual report.

        Rubric (NO speculation):
        1) Business Model — state the primary business model (sales/subscription/service-led/technology-driven B2G/B2B), how it creates/delivers value (R&D, manufacturing, services, long-term programs, partnerships), and revenue drivers. Use concrete phrases from sources.
        2) Market Position — say whether the company is leader/challenger/niche; include market share/leadership claims ONLY if explicitly stated; mention key geographies/segments if explicit.

        Formatting rules:
        - Neutral, analytic tone; 2–5 concise sentences per cell.
        - If you mention monetary values, include suffix '{currency_label}' (e.g., "£510.4{currency_label}").
        - DO NOT invent metrics or shares; omit any claim that isn’t supported.
        - If unclear, provide a concise, sourced description without numbers rather than guessing.

        Return **JSON ONLY** with this exact schema:
        {{
        "S3_3": {{
            "Business Model": "{year} Report text",
            "Market Position": "{year} Report text"
        }}
        }}

        SECTION 1 (context):
        {json.dumps(s1_context, ensure_ascii=False)}

        TEXT WINDOWS ({year}, truncated):
        {windows_text}
        """.strip()

def llm_build_business_competitiveness_for_year(
    report: "CompanyReport",
    year: int,
    windows_text: str,
    *,
    model: str = "gpt-4o-mini",
    max_chars: int = 16000,
    currency_label: str = "m",
    debug: bool = False,
) -> dict:
    """
    Build S3.3 (Business Competitiveness) for a SINGLE year using pre-assembled windows,
    grounded ONLY in Section 1 context.
    Writes into report.business_competitiveness.<year> fields.
    """

    # Section 1 compact context (you already have _compact_s1 implemented)
    s1_ctx = _compact_s1(report)

    # Build prompt
    prompt = _s33_prompt_builder_one_year_s1_only(
        s1_context=s1_ctx,
        windows_text=windows_text,
        year=year,
        currency_label=currency_label,
    )

    # LLM call in JSON mode
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"You synthesize precise, grounded business analysis and return valid JSON only."},
                {"role":"user","content":prompt}
            ],
            temperature=0,
            max_tokens=1200,
            response_format={"type":"json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        try:
            data = json.loads(raw)
        except Exception:
            if debug:
                print(f"[S3.3:{year}] JSON parse failed. Raw head:\n", raw[:800])
            data = {}
    except Exception as e:
        if debug:
            print(f"[S3.3:{year}] LLM error:", e)
        data = {}
        
    # --- Extract result safely ---
    s33 = data.get("S3_3") or {}
    bm_text = s33.get("Business Model", "N/A") or "N/A"
    mp_text = s33.get("Market Position", "N/A") or "N/A"

    return {
        "Business Model": bm_text,
        "Market Position": mp_text,
    }


# --------------------------- S 4.1 Risk Factors (2024 + 2023)-------------------------------------------

def _risk_prompt(year: int, text: str) -> str:
    """
    Build a strict JSON-only extraction prompt for S4.1 Risk Factors.
    Output one concise (<=2 sentence) summary per category per year.
    """
    
    return f"""
    You will extract *verbatim-faithful* Risk Factors for year {year} from the TEXT.

    NON-NEGOTIABLE RULES
    - Use ONLY the TEXT below. Do NOT add external context or examples unless those exact words appear in TEXT.
    - Prefer the exact nouns/phrases that appear in TEXT. Paraphrase minimally; preserve concepts and vocabulary.
    - Produce exactly **one sentence per category** (a semicolon to join two clauses is allowed).
    - If a category is not supported by TEXT, output "N/A".

    CATEGORIES (exact keys):
    - "Market Risks"
    - "Operational Risks"
    - "Financial Risks"
    - "Compliance Risks"

    COVERAGE CHECKLIST (include when present in TEXT):
    - Market: political considerations; fiscal/budget constraints; yearly fluctuations; downward pressure.
    - Operational: safety risks; plant failures; supplier interruptions; quality issues; employee absences; climate impact risks.
    - Financial: foreign exchange movements; group-specific risks from operational disruption; failure to deliver strategic objectives; customer payment defaults.
    - Compliance: operates in more than 50 countries (if stated); highly regulated environment; subject to applicable laws and regulations of each jurisdiction.

    OUTPUT (JSON ONLY, no extra text):
    {{
    "years": [{year}],
    "factors": {{
        "Market Risks": {{"{year}": "N/A"}},
        "Operational Risks": {{"{year}": "N/A"}},
        "Financial Risks": {{"{year}": "N/A"}},
        "Compliance Risks": {{"{year}": "N/A"}}
    }}
    }}

    TEXT:
    {text}
    """.strip()


def _coerce_text_or_na(v) -> str:
    """
    Normalize any value to a printable short string; keep 'N/A' as-is.
    """
    if v is None:
        return "N/A"
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return "N/A"
        if s.upper() == "N/A":
            return "N/A"
        return s
    try:
        return str(v)
    except Exception:
        return "N/A"


def extract_risk_factor(risk_text: str, year: int) -> dict:
    """
    LLM extraction for S4.1 Risk Factors.
    Schema:
    {
      "years": [...],
      "factors": {
        "Market Risks": {"2024": "...", "2023": "..."},
        "Operational Risks": {...},
        "Financial Risks": {...},
        "Compliance Risks": {...}
      }
    }
    """
    empty = {
            "years": [year],
            "factors": {
                "Market Risks": {str(year): "N/A"},
                "Operational Risks": {str(year): "N/A"},
                "Financial Risks": {str(year): "N/A"},
                "Compliance Risks": {str(year): "N/A"},
            }
        }
    if not risk_text:
        return empty

    prompt = _risk_prompt(year, risk_text) 
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You extract precise, source-grounded risks in strict JSON and never invent data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=900
        )
        data = _safe_json_from_llm(resp.choices[0].message.content.strip())
    except Exception as e:
        print(f"[S4.1] LLM error: {e}")
        data = {}

    out = empty
    cats = ["Market Risks","Operational Risks","Financial Risks","Compliance Risks"]
    for c in cats:
        v = (data.get("factors", {}).get(c, {}).get(str(year), "N/A"))
        out["factors"][c][str(year)] = _coerce_text_or_na(v)
    return out

def print_risk_factors_table(risks: dict):
    """
    Pretty-print S4.1 table (Markdown-friendly).
    Columns: Perspective | 2024 Report | 2023 Report
    """
    years = risks.get("years", [2024, 2023])
    years = [str(y) for y in years]
    # Guard to exactly two columns in order [2024, 2023] if present
    # but keep generic if caller provided different order
    header = ["S4.1: Risk Factors", ""]  # Title row
    print("S4.1: Risk Factors")
    print("Perspective | " + " | ".join(f"{y} Report" for y in years))
    print(":-- | " + " | ".join([":--"] * len(years)))

    categories = ["Market Risks", "Operational Risks", "Financial Risks", "Compliance Risks"]
    i = 0
    while i < len(categories):
        c = categories[i]
        row_values = []
        j = 0
        while j < len(years):
            y = years[j]
            val = risks.get("factors", {}).get(c, {}).get(y, "N/A")
            if val is None or str(val).strip() == "":
                val = "N/A"
            row_values.append(val)
            j += 1
        print(f"{c} | " + " | ".join(row_values))
        i += 1

def llm_pick_risk_sections(jsonl_path: str, top_k: int = 10, batch_size: int = 150, model: str = "gpt-4o-mini") -> List[str]:
    """
    Use an LLM to choose the top-k sections most likely to contain Risk Factors
    (principal risks, market/operational/financial/compliance, uncertainties, HSE/HSE risk,
     regulatory/export-control/sanctions, risk management).
    Returns a list of section_id strings (ranked best->worst).
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Load minimal section metadata
    sections: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
            except Exception:
                continue
            sid = rec.get("section_id", "") or ""
            title = rec.get("title", "") or ""
            if sid or title:
                sections.append({"section_id": sid, "title": title})

    if not sections:
        return []

    def _batches(lst, n):
        i = 0
        while i < len(lst):
            yield lst[i:i+n]
            i += n

    scored: List[Tuple[str, float]] = []

    for chunk in _batches(sections, batch_size):
        compact = [{"section_id": s["section_id"], "title": s["title"][:180]} for s in chunk]

        system_msg = (
            "You are a precise classifier for annual report sections. "
            "Return valid JSON arrays only."
        )

        user_prompt = f"""
        You will receive a list of sections from an annual report (title + section_id).
        Task: select entries most likely to contain **Risk Factors** content (principal risks and uncertainties),
        including market, operational, financial, and compliance/regulatory risks, HSE, ESG, export controls/sanctions,
        and any 'risk management' subsections.

        Favor titles like:
        - "Principal risks and uncertainties", "Strategic risks", "Risk Factors", "Risk management",
        - "Health, Safety and Environment risks", "Occupational and process safety",
        - "Environmental laws and regulations", "Regulatory compliance", "Export controls", "Sanctions",
        - "Financial risk management", "Market/credit/liquidity/FX risk".

        Avoid unrelated: remuneration, corporate governance boilerplate-only, auditor's opinion without risk detail, generic sustainability without risk framing.

        Return a JSON array of objects:
        {{ "section_id": string, "score": number in [0,1] }}

        Return ONLY the JSON array.

        Sections:
        {json.dumps(compact, ensure_ascii=False)}
        """.strip()

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=800,
            )
            raw = (resp.choices[0].message.content or "").strip()

            # Reuse your array extractor if you have one; inline minimal here:
            import re, json as _json
            m = re.search(r"\[\s*(?:\{|\[).*?\]\s*", raw, flags=re.S)
            if not m:
                arr = []
            else:
                arr = _json.loads(m.group(0))

            for obj in arr:
                sid = str(obj.get("section_id", "")).strip()
                try:
                    sc = float(obj.get("score", 0.0))
                except Exception:
                    sc = 0.0
                if sid != "":
                    scored.append((sid, sc))
        except Exception as e:
            print(f"[pick-risk] LLM error: {e}")
            continue

    if len(scored) == 0:
        # Fallback: keyword heuristic
        KEYWORDS = [
            "risk", "risks", "principal risks", "uncertainties", "risk management",
            "market risk", "operational risk", "financial risk", "compliance risk",
            "regulatory", "export control", "sanctions", "anti-bribery", "anti corruption",
            "health safety environment", "HSE", "ESG", "environmental regulations",
            "occupational safety", "process safety", "financial risk management",
            "credit risk", "liquidity risk", "foreign exchange", "FX"
        ]
        def _score(t: str) -> float:
            t2 = t.lower()
            count = 0
            k = 0
            while k < len(KEYWORDS):
                if KEYWORDS[k] in t2:
                    count += 1
                k += 1
            return float(count)

        ranked = sorted(
            sections,
            key=lambda s: _score((s.get("title") or "") + " " + (s.get("section_id") or "")),
            reverse=True
        )
        return [str(s.get("section_id", "")) for s in ranked[:top_k] if s.get("section_id")]

    # Aggregate max score per section_id
    best: Dict[str, float] = {}
    i = 0
    while i < len(scored):
        sid, sc = scored[i]
        if (sid not in best) or (sc > best[sid]):
            best[sid] = sc
        i += 1

    # Debug print (optional)
    print("\n=== All identified Risk-related sections ===")
    for sid, sc in sorted(best.items(), key=lambda x: x[1], reverse=True):
        print(f"{sid:50s} | score: {sc:.3f}")
    print("============================================\n")

    ranked_ids = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in ranked_ids[:top_k]]


# --------------------------- S 5.1 Board Composition (default 2024) -------------------------------------------


    
# -----------------------------------------------------------------------------------------
# ----------------- test extraction -------------------------------------------------------
# -----------------------------------------------------------------------------------------
def extract(md_file1: str, md_file2):
    report = CompanyReport()
    md_path_2024 = Path(md_file1)
    md_path_2023 = Path(md_file2)
    md_file_2024 = md_path_2024.stem
    md_file_2023 = md_path_2023.stem 
    md_file_path_2024 = f"data/parsed/{md_file_2024}.md"
    md_file_path_2023 = f"data/parsed/{md_file_2023}.md"

    # load 2024 and 2023 markdown files 
    try: 
        with open(md_path_2024, "r", encoding="utf-8") as f:
            md_text_2024 = f.read() 
            print(f"Successfully loaded {len(md_text_2024)} characters from {md_file_2024}")
    except FileNotFoundError: 
        print(f"File not found: {md_file_2024}")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    try: 
        with open(md_path_2023, "r", encoding="utf-8") as f:
            md_text_2023 = f.read() 
            print(f"Successfully loaded {len(md_text_2023)} characters from {md_file_2023}")
    except FileNotFoundError: 
        print(f"File not found: {md_file_2023}")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print("\n=== Testing Advanced Segmentation ===")
    # Load markdown content first
    normalize_and_segment_markdown(md_text_2024, Path(md_file_2024).stem)
    normalize_and_segment_markdown(md_text_2023, Path(md_file_2023).stem)

    # JSONL files will be created as:
    jsonl_file_2024_path = f"data/sections_report/{Path(md_file_2024).stem}.jsonl"
    jsonl_file_2023_path = f"data/sections_report/{Path(md_file_2023).stem}.jsonl"

    # Build embeddings with correct paths
    build_section_embeddings(jsonl_file_2024_path, f"data/parsed/{Path(md_file_2024).stem}.md")
    build_section_embeddings(jsonl_file_2023_path, f"data/parsed/{Path(md_file_2023).stem}.md")

    # --- S1.1: Basic Information (2024 ONLY) --- 
    company_name = query_company_name(jsonl_file_2024_path, md_file_2024)
    establishment_date = query_establishment_date(jsonl_file_2024_path, md_file_2024)
    company_hq = query_company_hq(jsonl_file_2024_path, md_file_2024)
    # save
    report.basic_info.company_name = company_name
    report.basic_info.establishment_date = establishment_date
    report.basic_info.headquarters_location = company_hq

    # --- S1.2: Core Competencies (2024 + 2023) --- 
    core_comp_2024 = extract_core_competencies(md_file_2024)
    core_comp_2023 = extract_core_competencies(md_file_2023)

    core_comp = merge_core_competencies(core_comp_2024, core_comp_2023)
    # save
    report.core_competencies.innovation_advantages.report_2024 = str((core_comp.get("Innovation Advantages", {}) or {}).get("2024", "N/A") or "N/A")
    report.core_competencies.innovation_advantages.report_2023 = str((core_comp.get("Innovation Advantages", {}) or {}).get("2023", "N/A") or "N/A")
    report.core_competencies.product_advantages.report_2024 = str((core_comp.get("Product Advantages", {}) or {}).get("2024", "N/A") or "N/A")
    report.core_competencies.product_advantages.report_2023 = str((core_comp.get("Product Advantages", {}) or {}).get("2023", "N/A") or "N/A")
    report.core_competencies.brand_recognition.report_2024 = str((core_comp.get("Brand Recognition", {}) or {}).get("2024", "N/A") or "N/A")
    report.core_competencies.brand_recognition.report_2023 = str((core_comp.get("Brand Recognition", {}) or {}).get("2023", "N/A") or "N/A")
    report.core_competencies.reputation_ratings.report_2024 = str((core_comp.get("Reputation Ratings", {}) or {}).get("2024", "N/A") or "N/A")
    report.core_competencies.reputation_ratings.report_2023 = str((core_comp.get("Reputation Ratings", {}) or {}).get("2023", "N/A") or "N/A")

    # Render the final table
    print("\n=== S1.2: Core Competencies ===")
    for perspective in ["Innovation Advantages", "Product Advantages", "Brand Recognition", "Reputation Ratings"]:
        print(perspective)
        print("2024:", core_comp[perspective]["2024"])
        print("2023:", core_comp[perspective]["2023"])
        print()
            
    # --- S1.3: Mission & Vision (2024 ONLY) --- 
    mv = extract_mission_vision_values(jsonl_file_2024_path, md_file_2024)
    print(f"Mission: {mv['mission']}")
    print(f"Vision: {mv['vision']}")
    print(f"Core Values: {mv['core_values']}")
    # save
    report.mission_vision.mission_statement = mv['mission']
    report.mission_vision.vision_statement = mv['vision']
    report.mission_vision.core_values = mv['core_values']

    # --- S2.1: Income Statement (2024 + 2023) ---  
    # do one for 2023, then one for 2024, then merge
    # is_topK_2024 = llm_pick_income_statements_sections(jsonl_file_2024_path, top_k=25)
    # is_topK_2023 = llm_pick_income_statements_sections(jsonl_file_2023_path, top_k=25)
    # windows_info_2024, income_text_2024 = assemble_financial_statement_windows_from_ids(is_topK_2024, jsonl_file_2024_path, md_file_path_2024, window_size=15, one_based_lines=True, choose_first_match_only=True)
    # windows_info_2023, income_text_2023 = assemble_financial_statement_windows_from_ids(is_topK_2023, jsonl_file_2023_path, md_file_path_2023, window_size=15, one_based_lines=True, choose_first_match_only=True)
    # income_2024 = extract_income_statement(income_text_2024, years=[2024, 2023, 2022])
    # income_2023 = extract_income_statement(income_text_2023, years=[2024, 2023, 2022])

    # merged_income = merge_income_statements_per_year_priority(income_2024, income_2023, years=[2024, 2023, 2022], debug=True)
    

    # print("\n=== Merged Income Statement ===")
    # print_income_statement_table(merged_income)
    
    # for field, values in merged_income["fields"].items():
    #     field_obj = getattr(report.income_statement, field.replace(" ", "_").lower(), None)
    #     if isinstance(field_obj, FinancialData):
    #         field_obj.year_2024 = values.get("2024", "N/A")
    #         field_obj.year_2023 = values.get("2023", "N/A")
    #         field_obj.year_2022 = values.get("2022", "N/A")
    #         field_obj.multiplier = merged_income.get("multiplier", "Units")
    #         field_obj.currency = merged_income.get("currency", "USD")
            
    # --- S2.2: Balance Sheet ---  
    # bs_topK_2024 = llm_pick_balance_sheet_sections(jsonl_file_2024_path, top_k=5)
    # bs_topK_2023 = llm_pick_balance_sheet_sections(jsonl_file_2023_path, top_k=5)

    # bs_win_2024, bs_text_2024 = assemble_financial_statement_windows_from_ids(bs_topK_2024, jsonl_file_2024_path, md_file_path_2024, window_size=15, one_based_lines=True, choose_first_match_only=True)
    # bs_win_2023, bs_text_2023 = assemble_financial_statement_windows_from_ids(bs_topK_2023, jsonl_file_2023_path, md_file_path_2023, window_size=15, one_based_lines=True, choose_first_match_only=True)
    
    # balance_2024 = extract_balance_sheet(bs_text_2024, years=[2024, 2023, 2022])
    # balance_2023 = extract_balance_sheet(bs_text_2023, years=[2024, 2023, 2022])

    # print("\n=== Balance Sheet 2024 Extraction ===")
    # print_balance_sheet_table(balance_2024)
    # print("\n=== Balance Sheet 2023 Extraction ===")
    # print_balance_sheet_table(balance_2023)

    # merged_balance = merge_balance_sheet_per_year_priority(balance_2024, balance_2023, years=[2024, 2023, 2022], debug=True)

    # print("\n=== Merged Balance Sheet ===")
    # print_balance_sheet_table(merged_balance)

    # for field, values in merged_balance["fields"].items():
    #     attr_name = field.replace(" ", "_").replace("'", "").replace("-", "_").lower()
    #     field_obj = getattr(report.balance_sheet, attr_name, None)
    #     if isinstance(field_obj, FinancialData):
    #         field_obj.year_2024 = values.get("2024", "N/A")
    #         field_obj.year_2023 = values.get("2023", "N/A")
    #         field_obj.year_2022 = values.get("2022", "N/A")
    #         field_obj.multiplier = merged_balance.get("multiplier", "Units")
    #         field_obj.currency = merged_balance.get("currency", "USD")


    # --- S2.3: Cash Flow Statement ---  
    # cf_topK_2024 = llm_pick_cash_flow_sections(jsonl_file_2024_path, top_k=25)
    # cf_topK_2023 = llm_pick_cash_flow_sections(jsonl_file_2023_path, top_k=25)

    # # 2) assemble text windows from IDs
    # cf_win_2024, cf_text_2024 = assemble_financial_statement_windows_from_ids(
    #     cf_topK_2024, jsonl_file_2024_path, md_file_path_2024,
    #     window_size=15, one_based_lines=True, choose_first_match_only=True
    # )
    # cf_win_2023, cf_text_2023 = assemble_financial_statement_windows_from_ids(
    #     cf_topK_2023, jsonl_file_2023_path, md_file_path_2023,
    #     window_size=15, one_based_lines=True, choose_first_match_only=True
    # )

    # # 3) extract per-year
    # cashflow_2024 = extract_cash_flow_statement(cf_text_2024, years=[2024, 2023, 2022])
    # cashflow_2023 = extract_cash_flow_statement(cf_text_2023, years=[2024, 2023, 2022])

    # print("\n=== Cash Flow 2024 Extraction ===")
    # print_cash_flow_table(cashflow_2024)
    # print("\n=== Cash Flow 2023 Extraction ===")
    # print_cash_flow_table(cashflow_2023)

    # # 4) merge with 2024-first policy
    # merged_cashflow = merge_cash_flow_per_year_priority(cashflow_2024, cashflow_2023, years=[2024, 2023, 2022], debug=True)

    # print("\n=== Merged Cash Flow Statement ===")
    # print_cash_flow_table(merged_cashflow)

    # # 5) bind into CompanyReport so DDRGenerator renders S2.3
    # # Map fields -> dataclass attributes:
    # mapping = {
    #     "Net Cash Flow from Operations": "net_cash_from_operations",
    #     "Net Cash Flow from Investing": "net_cash_from_investing",
    #     "Net Cash Flow from Financing": "net_cash_from_financing",
    #     "Net Increase/Decrease in Cash": "net_increase_decrease_cash",
    #     "Dividends": "dividends",
    # }

    # for field_name, attr in mapping.items():
    #     values = merged_cashflow["fields"].get(field_name, {})
    #     field_obj = getattr(report.cash_flow_statement, attr, None)
    #     if isinstance(field_obj, FinancialData):
    #         field_obj.year_2024 = values.get("2024", "N/A")
    #         field_obj.year_2023 = values.get("2023", "N/A")
    #         field_obj.year_2022 = values.get("2022", "N/A")
    #         field_obj.multiplier = merged_cashflow.get("multiplier", "Units")
    #         field_obj.currency = merged_cashflow.get("currency", "USD")

    # --- S2.4: Key Financial Metrics ---  
    
    # example table to test compute_key_metrics_from_tables()
    merged_income = {
        "years": [2024, 2023, 2022],
        "multiplier": "Millions",
        "currency": "GBP",
        "fields": {
            "Revenue": {"2024": 510.4, "2023": 472.6, "2022": 442.8},
            "Cost of Goods Sold": {"2024": "N/A", "2023": "N/A", "2022": "N/A"},
            "Gross Profit": {"2024": "N/A", "2023": "N/A", "2022": "N/A"},
            "Operating Expense": {"2024": 452.3, "2023": 427.2, "2022": 451.6},
            "Operating Income": {"2024": 58.1, "2023": 45.4, "2022": 49.4},
            "Net Profit": {"2024": 39.5, "2023": 5.4, "2022": 47.4},
            "Income before income taxes": {"2024": 53.3, "2023": 44.1, "2022": 47.9},
            "Income tax expense(benefit)": {"2024": 10.6, "2023": 6.4, "2022": 3.5},
            "Interest Expense": {"2024": 4.8, "2023": 1.3, "2022": 1.5},
        },
    }

    merged_balance = {
        "years": [2024, 2023, 2022],
        "multiplier": "Millions",
        "currency": "GBP",
        "fields": {
            "Total Assets": {"2024": 692.1, "2023": 596.4, "2022": 620.1},
            "Current Assets": {"2024": 264.0, "2023": 183.7, "2022": 181.2},
            "Non-Current Assets": {"2024": 422.3, "2023": 412.7, "2022": 438.9},
            "Total Liabilities": {"2024": "(335.8)", "2023": "(217.9)", "2022": "(202.0)"},
            "Current Liabilities": {"2024": "(221.9)", "2023": "(142.1)", "2022": "(113.7)"},
            "Non-Current Liabilities": {"2024": "(113.9)", "2023": "(75.8)", "2022": "(88.3)"},
            "Shareholders' Equity": {"2024": 356.3, "2023": 378.5, "2022": 418.1},
            "Retained Earnings": {"2024": 52.3, "2023": 62.9, "2022": 87.2},
            "Total Equity and Liabilities": {"2024": 692.1, "2023": 596.4, "2022": 620.1},
            "Inventories": {"2024": 127.1, "2023": 101.7, "2022": 99.6},
            "Prepaid Expenses": {"2024": "N/A", "2023": "N/A", "2022": "N/A"},
        },
    }

    merged_cashflow = {
        "years": [2024, 2023, 2022],
        "multiplier": "Millions",
        "currency": "GBP",
        "fields": {
            "Net Cash Flow from Operations": {"2024": 81.0, "2023": 65.9, "2022": 80.5},
            "Net Cash Flow from Investing": {"2024": "(47.6)", "2023": "(39.4)", "2022": "(30.5)"},
            "Net Cash Flow from Financing": {"2024": "(37.3)", "2023": "(40.2)", "2022": "(35.8)"},
            "Net Increase/Decrease in Cash": {"2024": "(3.9)", "2023": "(13.7)", "2022": 14.2},
            "Dividends": {"2024": "(19.6)", "2023": "(17.3)", "2022": "(14.4)"},
        },
    }

    # metrics = compute_key_metrics_from_tables(merged_income, merged_balance, merged_cashflow, years=[2024, 2023, 2022])

    # # Write into your dataclass
    # for field, values in metrics["fields"].items():
    #     attr = field.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    #     field_obj = getattr(report.key_financial_metrics, attr, None)
    #     if isinstance(field_obj, FinancialData):
    #         field_obj.year_2024 = values.get("2024", "N/A")
    #         field_obj.year_2023 = values.get("2023", "N/A")
    #         field_obj.year_2022 = values.get("2022", "N/A")
    #         field_obj.multiplier = metrics.get("multiplier", "Percent")
    #         field_obj.currency = metrics.get("currency", "N/A")

    # metrics = compute_key_metrics_from_tables(merged_income, merged_balance, merged_cashflow, years=[2024, 2023, 2022])
    # print_key_metrics_table(metrics)    
    
    # --- S2.5: Operating Performance ---  
    # op_topK_2024 = llm_pick_operating_performance_sections(jsonl_file_2024_path, top_k=15)
    # op_topK_2023 = llm_pick_operating_performance_sections(jsonl_file_2023_path, top_k=15)

    # win_2024, oper_text_2024 = assemble_financial_statement_windows_from_ids(
    #     op_topK_2024, jsonl_file_2024_path, md_file_path_2024,
    #     window_size=15, one_based_lines=True, choose_first_match_only=True
    # )
    # win_2023, oper_text_2023 = assemble_financial_statement_windows_from_ids(
    #     op_topK_2023, jsonl_file_2023_path, md_file_path_2023,
    #     window_size=15, one_based_lines=True, choose_first_match_only=True
    # )

    # oper_2024 = extract_operating_performance(oper_text_2024, years=[2024, 2023, 2022])
    # oper_2023 = extract_operating_performance(oper_text_2023, years=[2024, 2023, 2022])

    # print("\n=== Operating Performance 2024 Extraction ===")
    # print_operating_performance_table(oper_2024)
    # print("\n=== Operating Performance 2023 Extraction ===")
    # print_operating_performance_table(oper_2023)

    # merged_operating_performance = merge_operating_performance_per_year_priority(oper_2024, oper_2023, years=[2024, 2023, 2022], debug=True)

    # print("\n=== S2.5 Merged Operating Performance ===")
    # print_operating_performance_table(merged_operating_performance)

    # --- S3.1: Profitability Analysis ---  
    # profitability_analysis = llm_build_profitability_analysis(report, merged_income=merged_income, merged_balance=merged_balance, merged_cashflow=merged_cashflow,  
    #     derived_metrics=metrics, s25_operating=merged_operating_performance, years=[2024, 2023, 2022])
    # for k, v in profitability_analysis.items():
    #     print(f"{k}:\n{v}\n")

    # --- S3.2: Financial Performance Summary ---  
    # s32 = llm_build_financial_performance_summary(
    #     report,
    #     merged_income=merged_income,       
    #     merged_balance=merged_balance,   
    #     merged_cashflow=merged_cashflow, 
    #     derived_metrics=metrics,
    #     s25_operating=merged_operating_performance,
    #     years=[2024, 2023, 2022],
    #     company_name=company_name,
    #     establishment_date=establishment_date,
    #     company_hq=company_hq,
    # )
    # print("=== S3.2 ===")
    # for k, v in s32.items():
    #     print(k)
    #     print("  2024:", v["2024 Report"])
    #     print("  2023:", v["2023 Report"])


    # --- S3.3: Business Competitiveness ---  
    
    # bc_chunks_2024 = llm_pick_competitiveness_sections(jsonl_file_2024_path, top_k=10, model="gpt-4o-mini")
    # _, text_2024 = assemble_financial_statement_windows_from_ids(bc_chunks_2024, jsonl_file_2024_path, md_file_path_2024, window_size=5, one_based_lines=True, debug=False)
    # s33_2024 = llm_build_business_competitiveness_for_year(report, 2024, text_2024, model="gpt-4o-mini")

    # bc_chunks_2023 = llm_pick_competitiveness_sections(jsonl_file_2023_path, top_k=10, model="gpt-4o-mini")
    # _, text_2023 = assemble_financial_statement_windows_from_ids(bc_chunks_2023, jsonl_file_2023_path, md_file_path_2023, window_size=5, one_based_lines=True, debug=False)
    # s33_2023 = llm_build_business_competitiveness_for_year(report, 2023,text_2023,model="gpt-4o-mini")

    # report.business_competitiveness.business_model_2024 = s33_2024.get("Business Model", "N/A")
    # report.business_competitiveness.market_position_2024 = s33_2024.get("Market Position", "N/A")
    # report.business_competitiveness.business_model_2023 = s33_2023.get("Business Model", "N/A")
    # report.business_competitiveness.market_position_2023 = s33_2023.get("Market Position", "N/A")
        
    # --- S4.1: Risk Factors ---

    rf_chunks_2024 = llm_pick_risk_sections(jsonl_file_2024_path, top_k=10)
    rf_chunks_2023 = llm_pick_risk_sections(jsonl_file_2023_path, top_k=10)

    # 2) Windows -> combined text
    _, text_2024 = assemble_financial_statement_windows_from_ids(rf_chunks_2024, jsonl_file_2024_path, md_file_path_2024, window_size=10, one_based_lines=True, choose_first_match_only=False)
    _, text_2023 = assemble_financial_statement_windows_from_ids(rf_chunks_2023, jsonl_file_2023_path, md_file_path_2023, window_size=10, one_based_lines=True, choose_first_match_only=False)

    risks_2024_only = extract_risk_factor(text_2024, 2024)
    risks_2023_only = extract_risk_factor(text_2023, 2023)

    f24 = risks_2024_only.get("factors", {})
    s41_2024 = {}
    s41_2024["Market Risks"]      = f24.get("Market Risks", {}).get("2024", "N/A")
    s41_2024["Operational Risks"] = f24.get("Operational Risks", {}).get("2024", "N/A")
    s41_2024["Financial Risks"]   = f24.get("Financial Risks", {}).get("2024", "N/A")
    s41_2024["Compliance Risks"]  = f24.get("Compliance Risks", {}).get("2024", "N/A")

    f23 = risks_2023_only.get("factors", {})
    s41_2023 = {}
    s41_2023["Market Risks"]      = f23.get("Market Risks", {}).get("2023", "N/A")
    s41_2023["Operational Risks"] = f23.get("Operational Risks", {}).get("2023", "N/A")
    s41_2023["Financial Risks"]   = f23.get("Financial Risks", {}).get("2023", "N/A")
    s41_2023["Compliance Risks"]  = f23.get("Compliance Risks", {}).get("2023", "N/A")

    report.risk_factors.market_risks.report_2024      = s41_2024.get("Market Risks", "N/A")
    report.risk_factors.operational_risks.report_2024 = s41_2024.get("Operational Risks", "N/A")
    report.risk_factors.financial_risks.report_2024   = s41_2024.get("Financial Risks", "N/A")
    report.risk_factors.compliance_risks.report_2024  = s41_2024.get("Compliance Risks", "N/A")

    report.risk_factors.market_risks.report_2023      = s41_2023.get("Market Risks", "N/A")
    report.risk_factors.operational_risks.report_2023 = s41_2023.get("Operational Risks", "N/A")
    report.risk_factors.financial_risks.report_2023   = s41_2023.get("Financial Risks", "N/A")
    report.risk_factors.compliance_risks.report_2023  = s41_2023.get("Compliance Risks", "N/A")
    
    
    
    
    
    
    
    
    
    
    return report 

# lastly need  function that calls generator on extracted info and saves the report  
    # generator = DDRGenerator(report_info)
    # output_file = "artifacts/test_ddr_report.md"
    # generator.save_report(output_file)
    # print(f"\n✅ Saved generated DDR report to: {output_file}")

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Generate DDR report using two annual reports (2024 and 2023)."
    )

    parser.add_argument(
        "--md2024",
        required=True,
        help="Path to the 2024 markdown file (newest annual report)",
    )

    parser.add_argument(
        "--md2023",
        required=True,
        help="Path to the 2023 markdown file (previous year's report)",
    )

    args = parser.parse_args()

    # Run main extraction pipeline
    report_info = extract(args.md2024, args.md2023)

    # Ensure output directory exists
    Path("artifacts").mkdir(parents=True, exist_ok=True)

    generator = DDRGenerator(report_info)
    output_file = "artifacts/finddr_report.md"
    generator.save_report(output_file)
    print(f"\n✅ Saved generated DDR report to: {output_file}")