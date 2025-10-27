from pathlib import Path
import json
import re
from typing import List, Dict
from mistralai import Mistral, DocumentURLChunk
from mistralai.models import OCRResponse

from dotenv import load_dotenv
import os

load_dotenv(override=True)
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    """Replace image placeholders with base64-encoded image data."""
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(
            f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})"
        )
    return markdown_str

def get_combined_markdown(ocr_response: OCRResponse) -> str:
    """Combine OCR text and images into a single markdown document."""
    markdowns = []
    for page in ocr_response.pages:
        image_data = {img.id: img.image_base64 for img in page.images}
        markdowns.append(replace_images_in_markdown(page.markdown, image_data))
    return "\n\n".join(markdowns)

def process_pdf(pdf_path: Path, output_dir: Path, client: Mistral):
    """Process one PDF file with Mistral OCR and save Markdown output."""
    print(f"🔹 Processing: {pdf_path.name}")

    uploaded_file = client.files.upload(
        file={"file_name": pdf_path.name, "content": pdf_path.read_bytes()},
        purpose="ocr",
    )

    signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
    pdf_response = client.ocr.process(
        document=DocumentURLChunk(document_url=signed_url.url),
        model="mistral-ocr-latest",
        include_image_base64=False
    )

    md_content = get_combined_markdown(pdf_response)
    output_file = output_dir / f"{pdf_path.stem}.md"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"✅ Saved Markdown to {output_file}")


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


def find_markdown_files(folder_path: str) -> list[Path]:
    folder = Path(folder_path)
    if not folder.exists():
        return []
    return sorted(folder.rglob("*_mistral.md"))

def process_single_markdown(md_file_path: str, force_reparse: bool = False) -> bool:
    md_file = Path(md_file_path)
    if not md_file.exists():
        print(f"❌ File not found: {md_file}")
        return False

    # Keep base name (include _mistral suffix) to align with embeddings expectations
    base_name = md_file.stem

    sections_dir = Path("data/sections_report")
    sections_dir.mkdir(parents=True, exist_ok=True)
    jsonl_file = sections_dir / f"{base_name}.jsonl"

    if jsonl_file.exists() and not force_reparse:
        print(f"✅ JSONL already exists: {jsonl_file} (use --force to reprocess)")
        return True

    try:
        with md_file.open("r", encoding="utf-8") as f:
            markdown_text = f.read()
        print(f"📖 Loaded {len(markdown_text):,} characters from {md_file.name}")

        try:
            sections = normalize_and_segment_markdown(markdown_text, base_name)
            print(f"✅ Sections processed and saved to: {jsonl_file} ({len(sections)} sections)")
            return True
        except ImportError:
            print("❌ Error: normalize_and_segment module not available")
            return False
        except Exception as e:
            print(f"❌ Section processing failed: {e}")
            return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False


def process_markdown_folder(folder_path: str, force_reparse: bool = False) -> list[str]:
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    md_files = find_markdown_files(folder)
    if not md_files:
        print(f"⚠️  No *_mistral.md files found in {folder}")
        return []

    print(f"📁 Found {len(md_files)} markdown files in {folder}")
    processed = []

    for i, md_file in enumerate(md_files, 1):
        print(f"\n📋 Processing {i}/{len(md_files)}: {md_file.name}")
        ok = process_single_markdown(str(md_file), force_reparse=force_reparse)
        if ok:
            processed.append(str(Path("data/sections_report") / f"{md_file.stem}.jsonl"))

    print(f"\n🎉 Completed! Processed {len(processed)}/{len(md_files)} files")
    print(f"� JSONL files saved to: data/sections_report")
    return processed


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Mistral OCR and segmentation: Convert PDFs to Markdown, or segment Markdown into JSONL sections."
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to input PDF/Markdown or a folder. For OCR, provide a PDF file or a folder of PDFs. For segmentation, provide a Markdown file or folder."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data/parsed_md_val_mistral",
        help="Output directory for generated Markdown files when doing OCR (default: data/parsed_md_val_mistral)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if JSONL exists (segmentation mode)"
    )
    parser.add_argument(
        "--only-jsonl",
        action="store_true",
        help="Skip OCR and only segment a single existing Markdown file to JSONL"
    )
    parser.add_argument(
        "--recursive-jsonl",
        action="store_true",
        help="Recursively segment all *_mistral.md files under a folder into JSONL"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)

    # Segmentation modes
    if args.recursive_jsonl:
        return process_markdown_folder(str(input_path), force_reparse=args.force)

    if args.only_jsonl:
        if not input_path.exists() or input_path.suffix.lower() != ".md":
            print("❌ Please provide a Markdown (.md) file when using --only-jsonl")
            return
        return process_single_markdown(str(input_path), force_reparse=args.force)

    # OCR modes (PDF -> Markdown)
    outdir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"❌ Input not found: {input_path}")
        return

    if input_path.is_dir():
        pdfs = sorted(input_path.glob("*.pdf"))
        if not pdfs:
            print(f"⚠️  No PDFs found in {input_path}")
            return
        for pdf_path in pdfs:
            try:
                process_pdf(pdf_path, outdir, client)
            except Exception as e:
                print(f"❌ Error with {pdf_path.name}: {e}")
    else:
        if input_path.suffix.lower() != ".pdf":
            print(f"❌ File must be a PDF for OCR mode: {input_path}")
            return
        try:
            process_pdf(input_path, outdir, client)
        except Exception as e:
            print(f"❌ Error processing {input_path.name}: {e}")

if __name__ == "__main__":
    main()
