"""
Report Integration Module

This module integrates the extraction functions from extraction.py
with the report generator to create complete financial reports.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import extraction functions
from extraction import (
    normalize_and_segment_markdown, build_section_embeddings,
    query_company_name, query_company_hq, query_establishment_date,
    extract_mission_vision_values, extract_core_competencies,
    extract_income_statement, get_text_from_lines
)

# Import report generator classes
from report_generator import (
    CompanyReport, BasicInfo, CoreCompetencies, CoreCompetency,
    MissionVision, IncomeStatement, FinancialData, ReportGenerator
)


def find_financial_performance_sections(jsonl_file: str, top_k: int = 10) -> List[Dict]:
    """
    Search through JSONL file to identify the most likely section_ids that contain
    financial performance information based on section_id and title analysis.
    
    Args:
        jsonl_file: Path to the JSONL file (e.g., "data/sections_report/chemming_raw_parsed.jsonl")
        top_k: Number of top sections to return (default: 10)
    
    Returns:
        List of dictionaries with section info, ranked by financial relevance
    """
    
    # Keywords that indicate financial performance content (weighted by importance)
    financial_keywords = {
        # High priority financial terms
        "financial": 10, "performance": 8, "revenue": 10, "income": 9, "profit": 9,
        "earnings": 9, "cash": 8, "ebitda": 9, "statement": 8, "results": 7,
        
        # Medium priority terms
        "highlights": 6, "summary": 5, "analysis": 5, "overview": 4, "operating": 6,
        "margin": 7, "growth": 5, "conversion": 6, "diluted": 7, "underlying": 6,
        
        # Specific financial statement terms
        "balance": 8, "sheet": 7, "consolidated": 7, "comprehensive": 6,
        "flows": 7, "position": 6, "equity": 6, "assets": 6, "liabilities": 6,
        
        # Performance indicators
        "turnover": 7, "sales": 6, "costs": 6, "expenses": 6, "ratios": 7,
        "metrics": 6, "kpi": 7, "key": 5, "indicators": 6,
        
        # Time-based financial sections
        "annual": 5, "quarterly": 5, "year": 4, "2024": 6, "2023": 6, "2022": 6,
        
        # Management commentary
        "discussion": 5, "management": 4, "commentary": 5, "outlook": 5,
        "chairman": 4, "ceo": 4, "review": 5
    }
    
    sections = []
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    section = json.loads(line.strip())
                    sections.append(section)
    except FileNotFoundError:
        print(f"JSONL file not found: {jsonl_file}")
        return []
    except Exception as e:
        print(f"Error reading JSONL file: {e}")
        return []
    
    # Score each section based on financial keywords in section_id and title
    scored_sections = []
    
    for section in sections:
        section_id = section.get("section_id", "").lower()
        title = section.get("title", "").lower()
        char_count = section.get("char_count", 0)
        tables = section.get("tables", [])
        
        # Combine section_id and title for keyword matching
        combined_text = f"{section_id} {title}"
        
        # Calculate base score from keyword matches
        score = 0
        matched_keywords = []
        
        for keyword, weight in financial_keywords.items():
            if keyword in combined_text:
                score += weight
                matched_keywords.append(keyword)
        
        # Boost score for sections with financial tables
        if tables:
            score += len(tables) * 3  # Tables often contain financial data
        
        # Boost score for substantial content (likely detailed financial sections)
        if char_count > 5000:
            score += 5
        elif char_count > 2000:
            score += 3
        elif char_count > 500:
            score += 1
        
        # Penalty for very short sections (likely headers only)
        if char_count < 50:
            score -= 2
        
        # Special boosts for highly relevant patterns
        if any(pattern in combined_text for pattern in [
            "financial statement", "income statement", "cash flow", 
            "balance sheet", "profit loss", "comprehensive income",
            "financial performance", "financial highlights"
        ]):
            score += 15
        
        if any(pattern in combined_text for pattern in [
            "2024 performance", "2023 performance", "annual results",
            "financial results", "operating results"
        ]):
            score += 10
        
        # Add to scored list only if score > 0
        if score > 0:
            scored_sections.append({
                "section_id": section.get("section_id"),
                "title": section.get("title"),
                "section_number": section.get("section_number"),
                "lines": section.get("lines"),
                "char_count": char_count,
                "tables_count": len(tables),
                "score": score,
                "matched_keywords": matched_keywords
            })
    
    # Sort by score (descending) and return top_k
    scored_sections.sort(key=lambda x: x["score"], reverse=True)
    
    return scored_sections[:top_k]


def extract_complete_report(md_file_path: str) -> CompanyReport:
    """
    Extract complete company report from markdown file
    
    Args:
        md_file_path: Path to the markdown file to process
    
    Returns:
        CompanyReport object with extracted data
    """
    # Initialize report
    report = CompanyReport()
    
    # Get file paths
    md_path = Path(md_file_path)
    md_file = md_path.stem
    
    print(f"Processing: {md_file_path}")
    
    try:
        # Read markdown content
        with open(md_file_path, "r", encoding="utf-8") as f:
            sample_md = f.read()
            print(f"Loaded {len(sample_md):,} characters from {md_file}")
    except FileNotFoundError:
        print(f"File not found: {md_file_path}")
        return report
    except Exception as e:
        print(f"Error reading file: {e}")
        return report

    # Create JSONL file and build embeddings
    print("Creating sections and building embeddings...")
    advanced_sections = normalize_and_segment_markdown(sample_md, md_file)
    jsonl_file = f"data/sections_report/{md_file}.jsonl"
    build_section_embeddings(jsonl_file, md_file_path)
    
    # Extract S1.1: Basic Information
    print("Extracting basic information...")
    try:
        report.basic_info.company_name = query_company_name(jsonl_file, md_file)
        report.basic_info.establishment_date = query_establishment_date(jsonl_file, md_file)
        report.basic_info.headquarters_location = query_company_hq(jsonl_file, md_file)
    except Exception as e:
        print(f"Error extracting basic info: {e}")
    
    # Extract S1.3: Mission & Vision
    print("Extracting mission and vision...")
    try:
        mv = extract_mission_vision_values(jsonl_file, md_file)
        report.mission_vision.mission_statement = mv.get('mission', 'N/A')
        report.mission_vision.vision_statement = mv.get('vision', 'N/A')
        report.mission_vision.core_values = mv.get('core_values', 'N/A')
    except Exception as e:
        print(f"Error extracting mission/vision: {e}")
    
    # Extract S2.1: Income Statement
    print("Extracting income statement...")
    try:
        inc = extract_income_statement(md_file, years=[2024, 2023, 2022])
        
        # Map income statement fields
        for field_name, field_data in inc["fields"].items():
            # Convert field name to attribute name
            attr_name = field_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            
            if hasattr(report.income_statement, attr_name):
                financial_data = getattr(report.income_statement, attr_name)
                financial_data.year_2024 = field_data.get("2024", "N/A")
                financial_data.year_2023 = field_data.get("2023", "N/A")
                financial_data.year_2022 = field_data.get("2022", "N/A")
                financial_data.multiplier = inc.get("multiplier", "Units")
                financial_data.currency = inc.get("currency", "USD")
    except Exception as e:
        print(f"Error extracting income statement: {e}")
    
    # Find top financial sections for reference
    print("Finding financial performance sections...")
    try:
        financial_sections = find_financial_performance_sections(jsonl_file, top_k=10)
        section_ids = [section['section_id'] for section in financial_sections]
        print(f"Found {len(section_ids)} relevant financial sections:")
        for i, section_id in enumerate(section_ids[:5], 1):  # Show top 5
            print(f"  {i}. {section_id}")
    except Exception as e:
        print(f"Error finding financial sections: {e}")
    
    return report


def generate_report_from_markdown(md_file_path: str, output_path: str = None) -> str:
    """
    Complete pipeline: extract data from markdown and generate formatted report
    
    Args:
        md_file_path: Path to input markdown file
        output_path: Optional output path for the report (defaults to artifacts/)
    
    Returns:
        Path to the generated report file
    """
    
    # Extract data
    print("=== Starting Report Generation ===")
    report = extract_complete_report(md_file_path)
    
    # Generate report
    print("Generating markdown report...")
    generator = ReportGenerator(report)
    
    # Set default output path if not provided
    if output_path is None:
        md_filename = Path(md_file_path).stem
        output_path = f"artifacts/{md_filename}_financial_report.md"
    
    # Save report
    generator.save_report(output_path)
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate financial report from markdown file")
    parser.add_argument("md_file", help="Path to markdown file to process")
    parser.add_argument("--output", "-o", help="Output path for the report")
    parser.add_argument("--find-financial", action="store_true", 
                       help="Only find and display financial sections")
    
    args = parser.parse_args()
    
    if args.find_financial:
        # Just find financial sections
        md_path = Path(args.md_file)
        md_file = md_path.stem
        jsonl_file = f"data/sections_report/{md_file}.jsonl"
        
        if not Path(jsonl_file).exists():
            print(f"JSONL file not found: {jsonl_file}")
            print("Run full extraction first to create the JSONL file.")
            exit(1)
        
        print(f"Finding financial sections in: {jsonl_file}")
        sections = find_financial_performance_sections(jsonl_file)
        
        print("=" * 80)
        print("TOP 10 FINANCIAL PERFORMANCE SECTIONS")
        print("=" * 80)
        
        for i, section in enumerate(sections, 1):
            print(f"\n{i}. SECTION ID: {section['section_id']}")
            print(f"   Title: {section['title']}")
            print(f"   Score: {section['score']}")
            print(f"   Keywords: {', '.join(section['matched_keywords'])}")
            print("-" * 40)
    else:
        # Generate full report
        output_file = generate_report_from_markdown(args.md_file, args.output)
        print(f"\nReport generation complete!")
        print(f"Output file: {output_file}")