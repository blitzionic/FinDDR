import os
import sys
import argparse
from pathlib import Path
from mistral_parse import process_pdf
from dotenv import load_dotenv
from mistralai import Mistral
from extraction import extract, Lang
from report_generator import DDRGenerator
from embeddings import build_section_embeddings
from normalize_and_segment import normalize_and_segment_markdown

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.clean_markdown import normalize_file, clean_file
from utils.check_files import check_existing_files, determine_report_years


def main():
    # Example usage:
    # python main.py company_2024_report.pdf company_2023_report.pdf

    arg_parser = argparse.ArgumentParser(
        description="FinDDR 2025 Challenge: Generate comprehensive financial research reports from annual reports")
    arg_parser.add_argument('--pdf_2024', required=True, help='Path to first annual report PDF (preferably 2024 or more recent)')
    arg_parser.add_argument('--pdf_2023', required=True, help='Path to second annual report PDF (preferably 2023 or older)')
    arg_parser.add_argument('--lang', choices=['EN', 'ZH_SIM', 'ZH_TR', 'IN'], required=True, help='Target language for extraction and prompts (default: EN)')
    
    args = arg_parser.parse_args()
    pdf1 = args.pdf_2024
    pdf2 = args.pdf_2023
    target_lang = Lang[args.lang]

    base_2024 = None
    base_2023 = None

    print(f"\n{'='*60}")
    print(f"FinDDR 2025 Financial Document Deep Research Challenge")
    print(f"{'='*60}")
    print(f"Processing annual reports:")
    print(f"  File 1: {pdf1}")
    print(f"  File 2: {pdf2}")
    print(f"  Target language: {target_lang.value}")
    
    print(f"{'='*60}\n")

    try:
        pdf_2024, pdf_2023 = determine_report_years(pdf1, pdf2)
        
        base_2024 = Path(pdf_2024).stem
        base_2023 = Path(pdf_2023).stem
        
        load_dotenv()
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY is not set in environment")
        client = Mistral(api_key=api_key)
        output_dir = Path("data/parsed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check existing files
        files_2024 = check_existing_files(pdf_2024)
        files_2023 = check_existing_files(pdf_2023)
        
        print(f"Checking existing files:")
        print(f"   2024 Report ({Path(pdf_2024).name}):")
        print(f"     • Markdown: {'✅' if files_2024['markdown'] else '❌'}")
        print(f"     • JSONL: {'✅' if files_2024['jsonl'] else '❌'}")
        print(f"     • Embeddings: {'✅' if files_2024['embeddings'] else '❌'}")
        print(f"   2023 Report ({Path(pdf_2023).name}):")
        print(f"     • Markdown: {'✅' if files_2023['markdown'] else '❌'}")
        print(f"     • JSONL: {'✅' if files_2023['jsonl'] else '❌'}")
        print(f"     • Embeddings: {'✅' if files_2023['embeddings'] else '❌'}")
        print()
        
        # Process both pdfs to markdown and saves to data 
        print(" Step 1: Converting PDF documents to markdown...")
        if not files_2024['markdown']:
            process_pdf(Path(pdf_2024), output_dir, client)
        else:
            print(f"   ✅ Reusing existing markdown for 2024: {files_2024['md_path'].name}")
        if not files_2023['markdown']:
            process_pdf(Path(pdf_2023), output_dir, client)
        else:
            print(f"   ✅ Reusing existing markdown for 2023: {files_2023['md_path'].name}")
            
        print(f"\n Step 2: Processing markdown files...")
        md_file_2024 = files_2024['md_path'] if files_2024['markdown'] else Path("data/parsed") / f"{Path(pdf_2024).stem}.md"
        md_file_2023 = files_2023['md_path'] if files_2023['markdown'] else Path("data/parsed") / f"{Path(pdf_2023).stem}.md"
        
        try: 
            normalize_file(md_file_2024)
            normalize_file(md_file_2023)
            clean_file(md_file_2024)
            clean_file(md_file_2023)
            
        except Exception as e:
            print(f"Failed to normalize and clean markdown files: {e}")
        
        # Create jsonl files 
        print("\n Step 2a: Segmenting markdown files into structured sections...")
        try:
            jsonl_2024 = Path("data/sections_report") / f"{base_2024}.jsonl"
            jsonl_2023 = Path("data/sections_report") / f"{base_2023}.jsonl"
            
            # Segment 2024 markdown to jsonl
            if not jsonl_2024.exists():
                print(f"   🔹 Segmenting {md_file_2024.name}...")
                with open(md_file_2024, "r", encoding="utf-8") as f:
                    md_content_2024 = f.read()
                sections_2024 = normalize_and_segment_markdown(md_content_2024, base_2024)
                print(f"   ✅ Created {len(sections_2024)} sections for 2024 report")
            else:
                print(f"   ✅ JSONL already exists for 2024 report: {jsonl_2024.name}")
            
            # Segment 2023 markdown to jsonl
            if not jsonl_2023.exists():
                print(f"   🔹 Segmenting {md_file_2023.name}...")
                with open(md_file_2023, "r", encoding="utf-8") as f:
                    md_content_2023 = f.read()
                sections_2023 = normalize_and_segment_markdown(md_content_2023, base_2023)
                print(f"   ✅ Created {len(sections_2023)} sections for 2023 report")
            else:
                print(f"   ✅ JSONL already exists for 2023 report: {jsonl_2023.name}")
                
        except Exception as e:
            print(f"❌ Failed to segment markdown files: {e}")
            print(f"   Continuing anyway - embeddings may be skipped if JSONL is missing")
        
        
        # create embeddings for markdown files 
        print("\n Step 2b: Building embeddings for semantic search...")
        try:
            
            jsonl_2024 = Path("data/sections_report") / f"{base_2024}.jsonl"
            jsonl_2023 = Path("data/sections_report") / f"{base_2023}.jsonl"

            # Build embeddings for 2024 report
            if jsonl_2024.exists() and md_file_2024.exists():
                print(f"   🔹 Building embeddings for 2024 report...")
                build_section_embeddings(str(jsonl_2024), str(md_file_2024))
            else:
                missing = []
                if not jsonl_2024.exists():
                    missing.append(str(jsonl_2024))
                if not md_file_2024.exists():
                    missing.append(str(md_file_2024))
                print(f"   ⚠️  Skipping 2024 embeddings — missing: {', '.join(missing)}")

            # Build embeddings for 2023 report
            if jsonl_2023.exists() and md_file_2023.exists():
                print(f"   🔹 Building embeddings for 2023 report...")
                build_section_embeddings(str(jsonl_2023), str(md_file_2023))
            else:
                missing = []
                if not jsonl_2023.exists():
                    missing.append(str(jsonl_2023))
                if not md_file_2023.exists():
                    missing.append(str(md_file_2023))
                print(f"   ⚠️  Skipping 2023 embeddings — missing: {', '.join(missing)}")
        except Exception as e:
            print(f"Failed to build embeddings: {e}")

        print(f"\n Step 3: Generating comprehensive research report...")
        try:
            report = extract(str(md_file_2024), str(md_file_2023), currency_code="USD", target_lang=target_lang)
        except Exception as e:
            print(f"Failed to extract from reports: {e}")
        
        company_name = base_2024.split('_')[0] if '_' in base_2024 else base_2024
        generator = DDRGenerator(report, currency_code="USD")
        output_dir = Path("artifacts")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"finddr_report_{company_name}.md"
        generator.save_report(str(output_file))
        print(f"✅ Report saved to: {output_file}")   
        
    except FileNotFoundError as e:
        print(f"❌ File Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error processing {pdf1} and {pdf2}: {str(e)}")
        print(f"💡 Hint: Make sure both PDF files exist and are readable")
        sys.exit(1)


if __name__ == "__main__":
    main()