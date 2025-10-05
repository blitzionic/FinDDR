import re
import json
import os
from openai import OpenAI
from dotenv import load_dotenv 

from pathlib import Path
from typing import List, Dict
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
      
      #print(f"{result['rank']}. {result['title']} (distance: {result['distance']:.3f})")
      #print(f"   Section ID: {result['section_id']}")
      #print(f"   Lines: {result['lines'][0]}-{result['lines'][1]}")
      #print(f"   Chars: {result['char_count']:,}")
      #print(f"   Original text:\n {get_text_from_lines(md_file, result['lines'][0], result['lines'][1])}")
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

# ---------------------------------------------------------------------------------------------








# ----------------- test extraction -------------------
def extract(md_file: str):
    md_path = Path(md_file)
    md_file = md_path.stem 
    """Test the extraction functions with sample markdown"""
    try: 
        with open(md_path, "r", encoding="utf-8") as f:
            sample_md = f.read() 
            print(f"Successfully loaded {len(sample_md)} characters from {md_file}")
    except FileNotFoundError: 
        print(f"File not found: {md_file}")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print("\n=== Testing Advanced Segmentation ===")
    # create jsonl file 
    advanced_sections = normalize_and_segment_markdown(sample_md, Path(md_file).stem)
    jsonl_file = f"data/sections_report/{md_file}.jsonl"
    # build index 
    build_section_embeddings(f"data/sections_report/{md_file}.jsonl", f"data/parsed/{md_file}.md")
    # --- S1.1: Basic Information --- 
    company_name = query_company_name(jsonl_file, md_file),
    establishment_date = query_establishment_date(jsonl_file, md_file),
    company_hq = query_company_hq(jsonl_file, md_file)
    
    
    
    
    exit(-1) 




if __name__ == "__main__":
  
    parser = argparse.ArgumentParser()
    parser.add_argument(
      'md_file', nargs='?'
    )
    args = parser.parse_args()
    md_file = args.md_file
    extract(md_file)