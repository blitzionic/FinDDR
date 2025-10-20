from pydoc import doc
from pathlib import Path

import pandas as pd
import json
import os
import sys
import argparse
from typing import List, Optional

class DocProcessor:

    def __init__(self):
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions, TableStructureOptions, TableFormerMode, EasyOcrOptions
        )
        from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
        from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice

        pipe = PdfPipelineOptions()
        # OCR (toggle if you want auto OCR for scanned bits)
        pipe.do_ocr = True
        pipe.ocr_options = EasyOcrOptions(lang=["en"])  # add more langs if needed

        # Table structure (TablFormer)
        pipe.do_table_structure = True
        pipe.table_structure_options = TableStructureOptions(
            mode=TableFormerMode.ACCURATE,
            do_cell_matching=True
        )
        
        pdf_option = PdfFormatOption(
            # backend=DoclingParseV4DocumentBackend,  # or omit to use default
            pipeline_options=pipe
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: pdf_option
            }
        )

    def parse_once(self, pdf_path):
        result = self.converter.convert(pdf_path).document
        return result

    def process_to_markdown(self, parsed_doc):
        return parsed_doc.export_to_markdown()

    def process_to_tables(self, parsed_doc):
        tables_markdown = []
        for table_ix, table in enumerate(parsed_doc.tables, start=0):
            table_df: pd.DataFrame = table.export_to_dataframe()
            md = f"## Table {table_ix}\n\n{table_df.to_markdown(index=False)}\n\n"
            tables_markdown.append(md)
            
        return "".join(tables_markdown)

    def process_document_to_markdown(self, pdf_path):
        """
        Process a PDF document and return the markdown content.
        This method combines parsing and markdown conversion for convenient use.
        """
        # Parse the document
        parsed_doc = self.parse_once(pdf_path)
        
        # Convert to markdown
        markdown_content = self.process_to_markdown(parsed_doc)
        
        return markdown_content
  
  
def process_markdown_folder(self, folder_path: str, force_reparse: bool = False) -> List[str]:
        """
        Process all markdown files in a folder and convert them to sections using normalize_and_segment_markdown.
        
        Args:
            folder_path: Path to folder containing markdown files
            force_reparse: If True, reprocess even if JSONL file already exists
            
        Returns:
            List of paths to generated JSONL files
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Find all markdown files with _raw_parsed.md pattern
        md_files = list(folder_path.glob("*_raw_parsed.md"))
        
        if not md_files:
            print(f"⚠️  No *_raw_parsed.md files found in {folder_path}")
            return []
        
        print(f"📁 Found {len(md_files)} markdown files in {folder_path}")
        processed_files = []
        
        # Ensure output directories exist
        sections_dir = Path("data/sections_report")
        sections_dir.mkdir(parents=True, exist_ok=True)
        
        for i, md_file in enumerate(md_files, 1):
            print(f"\n📋 Processing {i}/{len(md_files)}: {md_file.name}")
            
            # Get base name (remove _raw_parsed.md suffix)
            base_name = md_file.stem.replace("_raw_parsed", "")
            
            # Check if JSONL already exists
            jsonl_file = sections_dir / f"{base_name}.jsonl"
            
            if jsonl_file.exists() and not force_reparse:
                print(f"✅ JSONL already exists: {jsonl_file.name} (skipping)")
                processed_files.append(str(jsonl_file))
                continue
            
            try:
                # Read markdown content
                with md_file.open("r", encoding="utf-8") as f:
                    markdown_content = f.read()
                
                print(f"📖 Loaded {len(markdown_content):,} characters from {md_file.name}")
                
                # Process sections using normalize_and_segment_markdown
                try:
                    from extraction import normalize_and_segment_markdown
                    normalize_and_segment_markdown(markdown_content, base_name)
                    print(f"✅ Sections processed for {base_name}")
                    processed_files.append(str(jsonl_file))
                    
                except ImportError:
                    print("❌ Error: extraction module not available")
                    print("   Make sure extraction.py is in the same directory")
                    continue
                except Exception as e:
                    print(f"❌ Section processing failed for {base_name}: {e}")
                    continue
                
            except Exception as e:
                print(f"❌ Error reading {md_file.name}: {e}")
                continue
        
        print(f"\n🎉 Completed! Processed {len(processed_files)}/{len(md_files)} files")
        print(f"📂 JSONL files saved to: {sections_dir}")
        return processed_files

def find_markdown_files(folder_path: str) -> List[Path]:
    """Find all *_raw_parsed.md files in a folder"""
    folder = Path(folder_path)
    if not folder.exists():
        return []
    
    md_files = list(folder.glob("*_raw_parsed.md"))
    return sorted(md_files)

def process_single_markdown(md_file_path: str, force_reparse: bool = False) -> bool:
    """Process a single markdown file to sections"""
    md_file = Path(md_file_path)
    
    if not md_file.exists():
        print(f"❌ File not found: {md_file}")
        return False
    
    if not md_file.name.endswith("_raw_parsed.md"):
        print(f"⚠️  Warning: File doesn't match expected pattern (*_raw_parsed.md): {md_file.name}")
    
    # Get base name
    base_name = md_file.stem.replace("_raw_parsed", "")
    
    # Check if JSONL already exists
    sections_dir = Path("data/sections_report")
    sections_dir.mkdir(parents=True, exist_ok=True)
    jsonl_file = sections_dir / f"{base_name}.jsonl"
    
    if jsonl_file.exists() and not force_reparse:
        print(f"✅ JSONL already exists: {jsonl_file} (use --force to reprocess)")
        return True
    
    try:
        # Read markdown content
        with md_file.open("r", encoding="utf-8") as f:
            markdown_content = f.read()
        
        print(f"📖 Loaded {len(markdown_content):,} characters from {md_file.name}")
        
        # Process sections
        try:
            from extraction import normalize_and_segment_markdown
            normalize_and_segment_markdown(markdown_content, base_name)
            print(f"✅ Sections processed and saved to: {jsonl_file}")
            return True
            
        except ImportError:
            print("❌ Error: extraction module not available")
            return False
        except Exception as e:
            print(f"❌ Section processing failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a PDF to Markdown using Docling, or directly segment existing Markdown files into structured JSONL sections."
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input file or folder. Example: data/raw/nvidia_2023.pdf or data/parsed/"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="data/parsed/",
        help="Output directory for the generated Markdown file (default: data/parsed/)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if JSONL files already exist"
    )

    parser.add_argument(
        "--only-jsonl",
        action="store_true",
        help="Skip Docling parsing and only segment a single existing Markdown file to JSONL"
    )

    parser.add_argument(
        "--recursive-jsonl",
        action="store_true",
        help="Recursively parse all *_raw_parsed.md files in a folder into JSONL"
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Case 1: recursive segmentation ---
    if args.recursive_jsonl:
        if not input_path.exists() or not input_path.is_dir():
            print(f"❌ Folder not found: {input_path}")
            sys.exit(1)
        print(f"🧩 Recursively segmenting all *_raw_parsed.md files under {input_path} ...")

        from extraction import normalize_and_segment_markdown

        md_files = list(input_path.rglob("*_raw_parsed.md"))
        if not md_files:
            print(f"⚠️ No *_raw_parsed.md files found in {input_path}")
            sys.exit(0)

        print(f"📁 Found {len(md_files)} markdown files to segment.")
        for md_path in md_files:
            base_name = md_path.stem
            jsonl_file = Path("data/sections_report") / f"{base_name}.jsonl"
            if jsonl_file.exists() and not args.force:
                print(f"✅ Skipping {md_path.name} (already has JSONL).")
                continue

            try:
                with open(md_path, "r", encoding="utf-8") as f:
                    markdown_text = f.read()
                sections = normalize_and_segment_markdown(markdown_text, base_name)
                print(f"✅ Segmented {md_path.name} → {jsonl_file.name} ({len(sections)} sections)")
            except Exception as e:
                print(f"❌ Error processing {md_path.name}: {e}")
        sys.exit(0)

    # --- Case 2: only segment Markdown into JSONL ---
    if args.only_jsonl:
        if not input_path.exists():
            print(f"❌ Markdown file not found: {input_path}")
            sys.exit(1)
        if input_path.suffix.lower() != ".md":
            print(f"⚠️ The file {input_path} is not a Markdown file. Please provide a .md file when using --only-jsonl.")
            sys.exit(1)

        print(f"🧩 Segmenting {input_path.name} directly into JSONL ...")
        try:
            from extraction import normalize_and_segment_markdown
            with open(input_path, "r", encoding="utf-8") as f:
                markdown_text = f.read()
            sections = normalize_and_segment_markdown(markdown_text, input_path.stem)
            print(f"✅ Extracted {len(sections)} sections from {input_path.name}")
        except ImportError:
            print("⚠️ Note: extraction.py not found — skipping segmentation.")
        except Exception as e:
            print(f"❌ Error during segmentation: {e}")
        sys.exit(0)

    # --- Case 3: full pipeline (PDF → Markdown → JSONL) ---
    if not input_path.exists():
        print(f"❌ PDF file not found: {input_path}")
        sys.exit(1)

    if input_path.suffix.lower() != ".pdf":
        print(f"⚠️ The file {input_path} is not a PDF. To convert Markdown only, use --only-jsonl or --recursive-jsonl.")
        sys.exit(1)

    # Step 1: Parse PDF → Markdown
    print(f"📘 Parsing {input_path.name} ...")
    proc = DocProcessor()
    markdown = proc.process_document_to_markdown(str(input_path))

    # Step 2: Save Markdown
    md_file = outdir / f"{input_path.stem}_raw_parsed.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"✅ Saved Markdown to {md_file}")

    # Step 3: Segment Markdown into structured sections
    try:
        from extraction import normalize_and_segment_markdown
        print(f"🧩 Segmenting {md_file.name} ...")
        sections = normalize_and_segment_markdown(markdown, md_file.stem)
        print(f"✅ Extracted {len(sections)} sections from markdown.")
    except ImportError:
        print("⚠️ Note: extraction.py not found — skipping section segmentation.")
    except Exception as e:
        print(f"❌ Error during segmentation: {e}")
