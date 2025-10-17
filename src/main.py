import os
import re
import json
import sys
from pathlib import Path
from parser import DocProcessor
from extraction import extract
from report_generator import CompanyReport, DDRGenerator
import argparse


def check_existing_files(pdf_path):
    """Check if markdown and related files already exist for a PDF"""
    base_name = Path(pdf_path).stem
    
    # Check for markdown file
    md_file = Path("data/parsed") / f"{base_name}_raw_parsed.md"
    
    # Check for JSONL file (used by extraction)
    jsonl_file = Path("data/sections_report") / f"{base_name}.jsonl"
    
    # Check for embeddings
    embeddings_file = Path("data/embeddings") / f"{base_name}_raw_parsed.faiss"
    
    return {
        'markdown': md_file.exists(),
        'jsonl': jsonl_file.exists(),
        'embeddings': embeddings_file.exists(),
        'md_path': md_file,
        'jsonl_path': jsonl_file,
        'embeddings_path': embeddings_file
    }


def process_document_to_markdown(processor, pdf_path, force_reparse=False):
    """Process a PDF document and return the markdown content and filename"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Determine output filename
    base_name = Path(pdf_path).stem
    output_dir = Path("data/parsed")
    output_dir.mkdir(parents=True, exist_ok=True)
    md_file = output_dir / f"{base_name}_raw_parsed.md"
    
    # Check if markdown file already exists
    if md_file.exists() and not force_reparse:
        print(f"✅ Found existing markdown file: {md_file}")
        print(f"   Skipping PDF parsing for {Path(pdf_path).name}")
        return str(md_file), base_name
    
    print(f" Processing {pdf_path}...")
    
    # Parse the document
    parsed_doc = processor.parse_once(pdf_path)
    
    # Convert to markdown
    markdown_content = processor.process_to_markdown(parsed_doc)
    
    # Save markdown file
    with md_file.open("w", encoding="utf-8") as fp:
        fp.write(markdown_content)
    
    print(f"✅ Markdown saved to: {md_file}")
    return str(md_file), base_name


def determine_report_years(pdf1_path, pdf2_path):
    """Determine which PDF is 2023 and which is 2024 based on filename or content"""
    pdf1_name = Path(pdf1_path).stem.lower()
    pdf2_name = Path(pdf2_path).stem.lower()
    
    if "2024" in pdf1_name and "2023" in pdf2_name:
        return pdf1_path, pdf2_path  # pdf1 is 2024, pdf2 is 2023
    elif "2023" in pdf1_name and "2024" in pdf2_name:
        return pdf2_path, pdf1_path  # pdf2 is 2024, pdf1 is 2023
    elif "2024" in pdf1_name:
        return pdf1_path, pdf2_path  # pdf1 is 2024, assume pdf2 is 2023
    elif "2024" in pdf2_name:
        return pdf2_path, pdf1_path  # pdf2 is 2024, assume pdf1 is 2023
    else:
        # Default assumption: first file is 2024, second is 2023
        print("⚠️ Warning: Could not determine years from filenames. Assuming first file is 2024, second is 2023.")
        return pdf1_path, pdf2_path


def main():
    # Example usage:
    # python main.py company_2024_report.pdf company_2023_report.pdf
    # python main.py report1.pdf report2.pdf --force-reparse
    # python main.py report1.pdf report2.pdf --case-id val001 --output submissions/val001.md
    
    arg_parser = argparse.ArgumentParser(
        description="FinDDR 2025 Challenge: Generate comprehensive financial research reports from annual reports")
    
    arg_parser.add_argument('pdf_file1', 
                           help='Path to first annual report PDF (preferably 2024 or more recent)')
    arg_parser.add_argument('pdf_file2', 
                           help='Path to second annual report PDF (preferably 2023 or older)')
    arg_parser.add_argument('--output', '-o', 
                           default='artifacts/findr_report.md',
                           help='Output path for the generated report (default: artifacts/findr_report.md)')
    arg_parser.add_argument('--case-id', 
                           help='Case ID for the report (used in FinDDR challenge submissions)')
    arg_parser.add_argument('--force-reparse', 
                           action='store_true',
                           help='Force reparsing of PDF files even if markdown files already exist')
    
    args = arg_parser.parse_args()
    
    pdf1 = args.pdf_file1
    pdf2 = args.pdf_file2
    output_path = args.output
    case_id = args.case_id
    force_reparse = args.force_reparse

    # Validate input files
    if not pdf1 or not pdf2:
        print("❌ Error: Both PDF files are required")
        arg_parser.print_help()
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"FinDDR 2025 Financial Document Deep Research Challenge")
    print(f"{'='*60}")
    print(f"Processing annual reports:")
    print(f"  File 1: {pdf1}")
    print(f"  File 2: {pdf2}")
    if case_id:
        print(f"  Case ID: {case_id}")
    print(f"  Output: {output_path}")
    if force_reparse:
        print(f"  Mode: Force reparse enabled - will reprocess existing markdown files")
    else:
        print(f"  Mode: Smart parsing - will reuse existing markdown files")
    print(f"{'='*60}\n")

    try:
        # Initialize document processor
        processor = DocProcessor()
        
        # Determine which file is which year
        pdf_2024, pdf_2023 = determine_report_years(pdf1, pdf2)
        print(f"📅 Year assignment: {Path(pdf_2024).name} → 2024, {Path(pdf_2023).name} → 2023")
        
        # Check existing files
        files_2024 = check_existing_files(pdf_2024)
        files_2023 = check_existing_files(pdf_2023)
        
        print(f"\n📁 Existing files check:")
        print(f"   2024 Report ({Path(pdf_2024).name}):")
        print(f"     • Markdown: {'✅' if files_2024['markdown'] else '❌'}")
        print(f"     • JSONL: {'✅' if files_2024['jsonl'] else '❌'}")
        print(f"     • Embeddings: {'✅' if files_2024['embeddings'] else '❌'}")
        print(f"   2023 Report ({Path(pdf_2023).name}):")
        print(f"     • Markdown: {'✅' if files_2023['markdown'] else '❌'}")
        print(f"     • JSONL: {'✅' if files_2023['jsonl'] else '❌'}")
        print(f"     • Embeddings: {'✅' if files_2023['embeddings'] else '❌'}")
        print()
        
        # Process both documents to markdown
        print(" Step 1: Converting PDF documents to markdown...")
        md_file_2024, base_name_2024 = process_document_to_markdown(processor, pdf_2024, force_reparse)
        md_file_2023, base_name_2023 = process_document_to_markdown(processor, pdf_2023, force_reparse)
        
        print(f"\n Step 2: Extracting structured data from reports...")
        
        # Extract structured information from 2024 report (primary)
        print("📊 Analyzing 2024 report...")
        try:
            report_2024 = extract(md_file_2024)
        except Exception as e:
            print(f"⚠️ Warning: Could not fully extract data from 2024 report: {e}")
            report_2024 = None
        
        # Extract structured information from 2023 report (for comparison)
        print("📊 Analyzing 2023 report...")
        try:
            report_2023 = extract(md_file_2023)
        except Exception as e:
            print(f"⚠️ Warning: Could not fully extract data from 2023 report: {e}")
            report_2023 = None
        
        print(f"\n Step 3: Generating comprehensive research report...")
        
        # For now, use the 2024 report as the primary structure
        # In a full implementation, you would merge data from both years
        company_report = report_2024 if report_2024 else CompanyReport()
        
        # generate FinDDR report 
        generator = DDRGenerator(company_report)
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # save report 
        generator.save_report(str(output_file))
        
        print(f"\n✅ Successfully generated FinDDR research report!")
        print(f"📄 Report saved to: {output_file}")
        print(f"\n📋 Report Structure Generated:")
        print(f"   • Section 1: Company Overview")
        print(f"     - S1.1: Basic Information")
        print(f"     - S1.2: Core Competencies") 
        print(f"     - S1.3: Mission & Vision")
        print(f"   • Section 2: Financial Performance")
        print(f"     - S2.1: Income Statement")
        print(f"     - S2.2: Balance Sheet")
        print(f"     - S2.3: Cash Flow Statement")
        print(f"     - S2.4: Key Financial Ratios")
        print(f"     - S2.5: Operating Performance")
        print(f"   • Section 3: Business Analysis")
        print(f"     - S3.1: Profitability Analysis")
        print(f"     - S3.2: Financial Performance Summary")
        print(f"     - S3.3: Business Competitiveness")
        print(f"   • Section 4: Risk Factors")
        print(f"   • Section 5: Corporate Governance")
        print(f"   • Section 6: Future Outlook")
        
        if case_id:
            print(f"\n🏷️  Case ID: {case_id}")
            print(f"📧 For FinDDR 2025 submission, email this report to: finddr2025@gmail.com")
            print(f"📧 Subject: FinDDR2025-{case_id.split('_')[0]}-YourModelName-YourTeamName")

    except FileNotFoundError as e:
        print(f"❌ File Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error processing {pdf1} and {pdf2}: {str(e)}")
        print(f"💡 Hint: Make sure both PDF files exist and are readable")
        sys.exit(1)


if __name__ == "__main__":
    main()