import os, json
import re
from pathlib import Path
import shutil
import tempfile
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from tqdm import tqdm
import time
from functools import lru_cache
from pathlib import Path
import yaml
import sys

from embeddings import build_section_embeddings, search_sections, append_next_sections
from report_generator import BalanceSheet, CashFlowStatement, CompanyReport, DDRGenerator, FinancialData, IncomeStatement, KeyFinancialMetrics, OperatingPerformance
sys.path.insert(0, str(Path(__file__).parent.parent))
from prompts.prompts import build_s1_1_prompt, build_s1_2_prompt, build_s1_3_prompt, build_s2_1_prompt, build_s2_2_prompt, build_s2_3_prompt, build_s2_5_prompt, build_s3_1_prompt
from prompts.prompts import build_s3_2_prompt, build_s3_3_prompt, build_s4_1_prompt, build_s5_1_prompt, build_s5_2_prompt, build_s6_1_prompt, build_s6_2_prompt, build_s6_3_prompt

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from enum import Enum

class Lang(str, Enum):
    EN = "EN" # Standard English
    ZH_SIM = "ZH_SIM"   # Simplified Chinese (CN)
    ZH_TR = "ZH_TR"   # Traditional Chinese (HK)
    IN = "IN"   # Indonesian

TARGET_LANGUAGE = Lang.EN

def display_lang(lang: Lang) -> str:
    return {"EN": "English", "ZH_SIM": "简体中文", "ZH_TR": "繁體中文"}[lang]
  
def is_chinese(lang: Lang) -> bool:
    return lang in (Lang.ZH_SIM, Lang.ZH_TR)

def set_target_language(lang: Lang):
    global TARGET_LANGUAGE
    TARGET_LANGUAGE = lang

def set_company_name(name: str):
    global COMPANY_NAME
    COMPANY_NAME = name
    
def set_currency_code(code: str):
    global CURRENCY_CODE
    CURRENCY_CODE = code
    
def set_multiplier(label: str):
    global MULTIPLIER
    MULTIPLIER = label

# ============================= Helper Functions ===========================

_YAML_PATH = Path(__file__).parents[1] / "prompts" / "keywords.yaml"
@lru_cache(maxsize=1)
def _load() -> dict:
    with open(_YAML_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_queries(section: str, lang: str):
    data = _load()
    return data.get(section, {}).get("search_queries", {}).get(lang.upper(), [])

def _safe_json_from_llm(s: str) -> dict:
    if s is None:
        return {}
    # strip code fences
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE)

    # try plain JSON first
    try:
        return json.loads(s)
    except Exception:
        pass

    # extract first {...} block if there’s extra text
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return {}

    js = m.group(0)

    # fix common issues: trailing commas before } or ]
    js = re.sub(r",\s*([}\]])", r"\1", js)

    # normalize smart quotes
    js = js.replace("“", '"').replace("”", '"').replace("’", "'")

    try:
        return json.loads(js)
    except Exception:
        return {}

def _normalize_na(v) -> str:
    import re
    
    if v is None:
        return "N/A"
    s = str(v).strip()
    if not s or s.upper() == "N/A":
        return "N/A"

    try:
        # Handle parentheses-style negatives like (10.6)
        if s.startswith("(") and s.endswith(")"):
            cleaned = re.sub(r"[£$€¥,]", "", s)
            inner = cleaned.strip("()")
            # validate it’s numeric but preserve original decimal precision
            float(inner)
            return f"({inner})"

        # Otherwise, clean currency symbols but do NOT reformat decimals
        cleaned = re.sub(r"[£$€¥,()]", "", s)

        # Validate numeric — if valid, return as-is (don’t format)
        float(cleaned)
        return cleaned

    except (ValueError, TypeError):
        # Return as-is for text, "N/A", or other non-numeric strings
        return s

def retrieve_relevant_text(search_queries: List[str], top_k: int, md_file: str) -> str:
    """
    Search for sections using multiple queries and return the complete combined text.
    Handles duplicate section names by using composite keys (section_id + line_range).
    
    Args:
        search_queries: List of search query strings
        top_k: Number of top results to return per query
        md_file: Name of the markdown file (without .md extension)
    
    Returns:
        Combined text from all relevant sections
    """
    all_results = []
    for query in search_queries:
        # returns list of dict, where each item is section identifier  
        results = search_sections(query, top_k=top_k, md_file=md_file)
        all_results.extend(results)
    
    # Sort by relevance (distance score) first
    all_results.sort(key=lambda x: x.get('distance', float('inf')))
    
    # De-duplicate using composite key: section_id + line_range + section_number
    seen_sections = set()
    unique_results = []
    
    for result in all_results:
        # Create composite key to handle duplicate section names
        start_line, end_line = result["lines"]
        composite_key = f"{result['section_id']}_{start_line}_{end_line}_{result.get('section_number', 0)}"
        
        if composite_key not in seen_sections:
            seen_sections.add(composite_key)
            unique_results.append(result)
  
    print(f"Final unique results: {len(unique_results)} sections")
    
    # Get the actual text for selected sections
    with open(f"data/parsed/{md_file}.md", "r", encoding="utf-8") as f:
        markdown_text = f.read()
    
    lines = markdown_text.split('\n')
    context = ""

    for h in unique_results:
        s, e = h["lines"]
        section_text = '\n'.join(lines[s - 1:e + 1])
        # Include section number for clarity when there are duplicate titles
        section_identifier = f"{h.get('title')}"
        context += f"\n--- {section_identifier} ---\n{section_text}\n\n"
    
    return context.strip()

def to_zh_currency(code: str, trad: bool = False) -> str:
    """
    Map ISO currency codes to Chinese names.
    Fallbacks to the original code if unknown.
    """
    if code is None:
        return "未知" if not trad else "未知"
    code = str(code).upper().strip()
    table_sim = {
        "USD": "美元",
        "EUR": "欧元",
        "GBP": "英镑",
        "JPY": "日元",
        "CNY": "人民币",
        "HKD": "港元",
        "TWD": "新台币",
        "SGD": "新元",
        "AUD": "澳元",
        "CAD": "加元",
        "CHF": "瑞士法郎",
        "INR": "卢比",
        "IDR": "印尼盾",
        "MYR": "令吉",
    }
    table_tr = {
        "USD": "美元",
        "EUR": "歐元",
        "GBP": "英鎊",
        "JPY": "日圓",
        "CNY": "人民幣",
        "HKD": "港元",
        "TWD": "新台幣",
        "SGD": "新幣",
        "AUD": "澳元",
        "CAD": "加元",
        "CHF": "瑞士法郎",
        "INR": "盧比",
        "IDR": "印尼盾",
        "MYR": "令吉",
    }
    table = table_tr if trad else table_sim
    return table.get(code, code)

def to_zh_multiplier(mult: str, trad: bool = False) -> str:
    """
    Normalize English scale labels → Chinese labels.
    Accepts common variants (Units/Unit, Thousands/K, Millions/M, Billions/B).
    """
    if mult is None:
        return "未知" if not trad else "未知"
    s = str(mult).strip().lower()
    # normalize
    if s in ("unit", "units"):
        key = "Units"
    elif s in ("thousand", "thousands", "k"):
        key = "Thousands"
    elif s in ("million", "millions", "m", "mil"):
        key = "Millions"
    elif s in ("billion", "billions", "bn", "b"):
        key = "Billions"
    elif s in ("trillion", "trillions", "tn", "t"):
        key = "Trillions"
    else:
        return mult

    if trad:
            mapping = {
                "Units": "單位",
                "Thousands": "千",
                "Millions": "百萬",
                "Billions": "十億",
                "Trillions": "兆",
            }
    else:
        mapping = {
            "Units": "单位",
            "Thousands": "千",
            "Millions": "百万",
            "Billions": "十亿",
            "Trillions": "万亿",
        }
    return mapping[key]

def to_zh_labels(currency_code: str, multiplier_label: str, trad: bool = False) -> Tuple[str, str]:
    return to_zh_currency(currency_code, trad), to_zh_multiplier(multiplier_label, trad)

def to_float(value):
    if value in (None, "", "N/A", "-", "--", "—"):
        return None
    try:
        s = str(value).strip()
        # strip currency & spaces
        if s and s[0] in "$€£¥":
            s = s[1:].strip()
        s = s.replace(",", "").replace("\xa0", "").strip()
        # parentheses -> negative
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]
        return float(s)
    except Exception:
        return None

def fill_income_data(income_data):
    def fmt(v):
        if v is None:
            return "N/A"
        out = f"{abs(v):,.1f}".rstrip("0").rstrip(".")
        return f"({out})" if v < 0 else out


    def safe_set(d, key, value):
        """
        Only update d[key] if it's truly empty.
        Prevents overwriting fallback or pre-filled values.
        """
        if d.get(key) in (None, "", "N/A", "-", "--", "—"):
            d[key] = value


    def fill_year(d):
        # read parsed values
        rev    = to_float(d.get("revenue"))
        cogs   = to_float(d.get("cost_of_goods_sold"))   # expense (may be negative)
        gp     = to_float(d.get("gross_profit"))
        opex   = to_float(d.get("operating_expense"))    # expense (may be negative)
        oi     = to_float(d.get("operating_income"))
        net    = to_float(d.get("net_profit"))
        pretax = to_float(d.get("income_before_taxes"))
        tax    = to_float(d.get("tax_expense"))          # expense (may be negative)

        # --- Revenue, COGS, Gross Profit ---
        if gp is None and rev is not None and cogs is not None:
            gp = rev - abs(cogs)
            safe_set(d, "gross_profit", fmt(gp))

        if cogs is None and rev is not None and gp is not None:
            cogs = -(rev - gp)  # keep as negative
            safe_set(d, "cost_of_goods_sold", fmt(cogs))

        if rev is None and gp is not None and cogs is not None:
            rev = gp + abs(cogs)
            safe_set(d, "revenue", fmt(rev))

        # --- Operating Income, Operating Expense, Gross Profit ---
        if oi is None and gp is not None and opex is not None:
            oi = gp - abs(opex)
            safe_set(d, "operating_income", fmt(oi))

        if opex is None and gp is not None and oi is not None:
            opex = -(gp - oi)
            safe_set(d, "operating_expense", fmt(opex))

        if gp is None and oi is not None and opex is not None:
            gp = oi + abs(opex)
            safe_set(d, "gross_profit", fmt(gp))

        # --- Net, Pretax, Tax (tax is an expense) ---
        if net is None and pretax is not None and tax is not None:
            net = pretax - abs(tax)
            safe_set(d, "net_profit", fmt(net))

        if pretax is None and net is not None and tax is not None:
            pretax = net + abs(tax)
            safe_set(d, "income_before_taxes", fmt(pretax))

        if tax is None and pretax is not None and net is not None:
            tax = -(pretax - net)
            safe_set(d, "tax_expense", fmt(tax))

    years = ("2024", "2023", "2022")
    
    for _ in range(2):
        for y in years:
            if y in income_data:
                fill_year(income_data[y])
    return income_data


def fill_missing_balance_sheet_values(balance_data):
    """
    Fill missing balance sheet values using standard identities without overwriting
    existing non-missing values.

    Works per year independently. Equations used:
      1) total_assets = current_assets + non_current_assets
         ↔ current_assets = total_assets - non_current_assets
         ↔ non_current_assets = total_assets - current_assets
      2) total_liabilities = current_liabilities + non_current_liabilities
         ↔ current_liabilities = total_liabilities - non_current_liabilities
         ↔ non_current_liabilities = total_liabilities - current_liabilities
      3) accounting equation: total_assets = total_liabilities + shareholders_equity
         ↔ shareholders_equity = total_assets - total_liabilities
         ↔ total_liabilities = total_assets - shareholders_equity
      4) total_equity_and_liabilities = total_assets (identity)

    Only fills values that are missing in balance_data[year][field].
    Does NOT overwrite existing non-missing strings.
    """

    def is_missing(v):
        return v in (None, "", "N/A", "-", "--")

    def to_float(v):
        if is_missing(v):
            return None
        s = str(v).strip().replace("\xa0", "").replace(" ", "")
        # Strip leading currency symbol
        if s and s[0] in ("£", "$", "€", "¥"):
            s = s[1:]
        # Handle parentheses for negatives
        neg = s.startswith("(") and s.endswith(")")
        if neg:
            s = s[1:-1]
        # Normalize thousands/decimal (supports "17.102.428", "36,072.95", "1.234.567,89")
        has_dot, has_comma = "." in s, "," in s
        try:
            if has_dot and has_comma:
                # Decide decimal by last separator
                if s.rfind(".") > s.rfind(","):
                    s = s.replace(",", "")
                else:
                    s = s.replace(".", "").replace(",", ".")
            elif has_dot and s.count(".") > 1:
                s = s.replace(".", "")
            elif has_comma and s.count(",") > 1:
                s = s.replace(",", "")
            else:
                s = s.replace(",", ".")
            val = float(s)
            return -val if neg else val
        except Exception:
            return None

    def fmt(v):
        if v is None:
            return "N/A"
        # Balance sheet lines are usually non-negative, but keep parentheses if negative appears.
        return f"({abs(v):,.1f})" if v < 0 else f"{v:,.1f}"

    # Fields we might compute
    F = {
        "TA": "total_assets",
        "CA": "current_assets",
        "NCA": "non_current_assets",
        "TL": "total_liabilities",
        "CL": "current_liabilities",
        "NCL": "non_current_liabilities",
        "SE": "shareholders_equity",
        "TEL": "total_equity_and_liabilities",
    }

    years = [y for y in ("2024", "2023", "2022") if y in balance_data]

    for year in years:
        # We’ll iterate until a full pass makes no changes
        changed = True
        while changed:
            changed = False

            row = balance_data.get(year, {})
            # Parse current numeric values
            vals = {k: to_float(row.get(v)) for k, v in F.items()}

            # 1) Assets split
            # TA = CA + NCA
            if vals["TA"] is None and vals["CA"] is not None and vals["NCA"] is not None:
                row[F["TA"]] = fmt(vals["CA"] + vals["NCA"]); changed = True
            if vals["CA"] is None and vals["TA"] is not None and vals["NCA"] is not None:
                row[F["CA"]] = fmt(vals["TA"] - vals["NCA"]); changed = True
            if vals["NCA"] is None and vals["TA"] is not None and vals["CA"] is not None:
                row[F["NCA"]] = fmt(vals["TA"] - vals["CA"]); changed = True

            # Refresh after possible writes
            vals = {k: to_float(row.get(v)) for k, v in F.items()}

            # 2) Liabilities split
            # TL = CL + NCL
            if vals["TL"] is None and vals["CL"] is not None and vals["NCL"] is not None:
                row[F["TL"]] = fmt(vals["CL"] + vals["NCL"]); changed = True
            if vals["CL"] is None and vals["TL"] is not None and vals["NCL"] is not None:
                row[F["CL"]] = fmt(vals["TL"] - vals["NCL"]); changed = True
            if vals["NCL"] is None and vals["TL"] is not None and vals["CL"] is not None:
                row[F["NCL"]] = fmt(vals["TL"] - vals["CL"]); changed = True

            # Refresh again
            vals = {k: to_float(row.get(v)) for k, v in F.items()}

            # 3) Accounting equation: TA = TL + SE
            if vals["TA"] is None and vals["TL"] is not None and vals["SE"] is not None:
                row[F["TA"]] = fmt(vals["TL"] + vals["SE"]); changed = True
            if vals["SE"] is None and vals["TA"] is not None and vals["TL"] is not None:
                row[F["SE"]] = fmt(vals["TA"] - vals["TL"]); changed = True
            if vals["TL"] is None and vals["TA"] is not None and vals["SE"] is not None:
                row[F["TL"]] = fmt(vals["TA"] - vals["SE"]); changed = True

            # Refresh again
            vals = {k: to_float(row.get(v)) for k, v in F.items()}

            # 4) Total Equity & Liabilities = Total Assets
            if is_missing(row.get(F["TEL"])) and vals["TA"] is not None:
                row[F["TEL"]] = fmt(vals["TA"]); changed = True
            if is_missing(row.get(F["TA"])) and vals["TEL"] is not None:
                row[F["TA"]] = fmt(vals["TEL"]); changed = True

        # Done with this year; write back
        balance_data[year] = row

    return balance_data

# ===================== S1.1: Basic Information with FAISS Search =====================

def extract_s1_1(md_file_2024: str, top_k: int = 10, model: str = "gpt-4.1-mini"):

    lang_key = TARGET_LANGUAGE.value
    search_queries = get_queries("s1_1", lang_key)
    print(search_queries)
    context = retrieve_relevant_text(search_queries, top_k, md_file_2024)
    prompt = build_s1_1_prompt(context, TARGET_LANGUAGE)
    
    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a precise information extractor. Extract only what's explicitly stated in the text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=600
        )
        result = _safe_json_from_llm(response.choices[0].message.content)
        return (
            _normalize_na(result.get("company_name", "N/A")),
            _normalize_na(result.get("establishment_date", "N/A")),
            _normalize_na(result.get("headquarters", "N/A"))
        )
    except:
        return ("N/A", "N/A", "N/A")


# ===================== S1.2: Core Competencies with FAISS Search =====================

def extract_s1_2(md_file: str, top_k: int, year: int, model: str = "gpt-4.1-mini"):

    lang_key = TARGET_LANGUAGE.value
    search_queries = get_queries("s1_2", lang_key)
    context = retrieve_relevant_text(search_queries, top_k, md_file)
    prompt = build_s1_2_prompt(context, year, TARGET_LANGUAGE)    
    
    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": f"Extract core competencies for {year}. Be specific and factual."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1500
        )
        content = response.choices[0].message.content
        
        return _safe_json_from_llm(content)
    
    except Exception as e:
        return {
            "Innovation Advantages": "N/A",
            "Product Advantages": "N/A",
            "Brand Recognition": "N/A",
            "Reputation Ratings": "N/A"
        }

def merge_core_competencies(comp_2024: dict, comp_2023: dict) -> dict:
    """
    Merge core competencies from both years into the required format.
    """
    merged = {}
    for key in ["Innovation Advantages", "Product Advantages", "Brand Recognition", "Reputation Ratings"]:
        merged[key] = {
            "2024": comp_2024.get(key, "N/A"),
            "2023": comp_2023.get(key, "N/A")
        }
    return merged

def extract_s1_3(md_file_2024: str, top_k: int, model: str = "gpt-4.1-mini"):
        
    lang_key = TARGET_LANGUAGE.value
    search_queries = get_queries("s1_3", lang_key)
    context = retrieve_relevant_text(search_queries, top_k, md_file_2024)
    prompt = build_s1_3_prompt(context, TARGET_LANGUAGE)

    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Extract exactly what is requested. Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=800
        )
        
        result = _safe_json_from_llm(response.choices[0].message.content)
        
        return {
            "mission": _normalize_na(result.get("mission", "N/A")),
            "vision": _normalize_na(result.get("vision", "N/A")),
            "core_values": _normalize_na(result.get("core_values", "N/A"))
        }
        
    except Exception as e:
        print(f"Error in extract_mission_vision_with_rag: {e}")
        return {
            "mission": "N/A",
            "vision": "N/A", 
            "core_values": "N/A"
        }

# ===================== Section 2: Financial Statements with FAISS Search =====================
        
def extract_s2_1(md_file_2024: str, md_file_2023: str, top_k: int, model: str = "gpt-4.1-mini"):
        
    lang_key = TARGET_LANGUAGE.value
    search_queries = get_queries("s2_1", lang_key)
    
    context_2024 = retrieve_relevant_text(search_queries, top_k, md_file_2024)
    prompt_2024 = build_s2_1_prompt(context_2024, 2024, TARGET_LANGUAGE)
    
    context_2023 = retrieve_relevant_text(search_queries, top_k, md_file_2023)
    prompt_2023 = build_s2_1_prompt(context_2023, 2023, TARGET_LANGUAGE)

    result_2024 = None
    result_2023 = None

    try:
        response_2024 = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a financial data extraction expert. Extract exact values from financial statements. Return valid JSON only."},
                {"role": "user", "content": prompt_2024}
            ],
            temperature=0,
            max_tokens=2000
        )
        result_2024 = _safe_json_from_llm(response_2024.choices[0].message.content)
        print(f"✓ Successfully extracted from 2024 report")
    except Exception as e:
        print(f"✗ Error extracting from 2024 report: {e}")
        result_2024 = {}

    try:
        response_2023 = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a financial data extraction expert. Extract exact values from financial statements. Return valid JSON only."},
                {"role": "user", "content": prompt_2023}
            ],
            temperature=0,
            max_tokens=2000
        )
        result_2023 = _safe_json_from_llm(response_2023.choices[0].message.content)
        print(f"✓ Successfully extracted from 2023 report")
    except Exception as e:
        print(f"✗ Error extracting from 2023 report: {e}")
        result_2023 = {}

        
    def _merge_year_data(primary_data, fallback_data):
        """
        Merge year data field-by-field.
        Use primary if value is not N/A, otherwise use fallback.
        """
        if not primary_data:
            return fallback_data or {}
        if not fallback_data:
            return primary_data or {}
        
        merged = {}
        all_keys = set(primary_data.keys()) | set(fallback_data.keys())
        
        for key in all_keys:
            primary_value = primary_data.get(key, "N/A")
            fallback_value = fallback_data.get(key, "N/A")
            
            # Use primary unless it's N/A, then use fallback
            if primary_value not in ["N/A", None, ""]:
                merged[key] = primary_value
            else:
                merged[key] = fallback_value
        
        return merged

    data_2024 = result_2024.get("2024", {}) if result_2024 else {}

    # For 2023: merge 2024 report and 2023 report field-by-field
    data_2023_from_2024 = result_2024.get("2023", {}) if result_2024 else {}
    data_2023_from_2023 = result_2023.get("2023", {}) if result_2023 else {}
    data_2023 = _merge_year_data(data_2023_from_2024, data_2023_from_2023)

    # For 2022: merge 2024 report and 2023 report field-by-field  
    data_2022_from_2024 = result_2024.get("2022", {}) if result_2024 else {}
    data_2022_from_2023 = result_2023.get("2022", {}) if result_2023 else {}
    data_2022 = _merge_year_data(data_2022_from_2024, data_2022_from_2023)    

    multiplier = result_2024.get("multiplier", result_2023.get("multiplier", "Units") if result_2023 else "Units") if result_2024 else "Units"
    currency = result_2024.get("currency", result_2023.get("currency", "USD") if result_2023 else "USD") if result_2024 else "USD"
    
    return {
        "2024": {
            "revenue": _normalize_na(data_2024.get("revenue", "N/A")),
            "cost_of_goods_sold": _normalize_na(data_2024.get("cost_of_goods_sold", "N/A")),
            "gross_profit": _normalize_na(data_2024.get("gross_profit", "N/A")),
            "operating_expense": _normalize_na(data_2024.get("operating_expense", "N/A")),
            "operating_income": _normalize_na(data_2024.get("operating_income", "N/A")),
            "net_profit": _normalize_na(data_2024.get("net_profit", "N/A")),
            "income_before_taxes": _normalize_na(data_2024.get("income_before_taxes", "N/A")),
            "tax_expense": _normalize_na(data_2024.get("tax_expense", "N/A")),
            "interest_expense": _normalize_na(data_2024.get("interest_expense", "N/A"))
        },
        "2023": {
            "revenue": _normalize_na(data_2023.get("revenue", "N/A")),
            "cost_of_goods_sold": _normalize_na(data_2023.get("cost_of_goods_sold", "N/A")),
            "gross_profit": _normalize_na(data_2023.get("gross_profit", "N/A")),
            "operating_expense": _normalize_na(data_2023.get("operating_expense", "N/A")),
            "operating_income": _normalize_na(data_2023.get("operating_income", "N/A")),
            "net_profit": _normalize_na(data_2023.get("net_profit", "N/A")),
            "income_before_taxes": _normalize_na(data_2023.get("income_before_taxes", "N/A")),
            "tax_expense": _normalize_na(data_2023.get("tax_expense", "N/A")),
            "interest_expense": _normalize_na(data_2023.get("interest_expense", "N/A"))
        },
        "2022": {
            "revenue": _normalize_na(data_2022.get("revenue", "N/A")),
            "cost_of_goods_sold": _normalize_na(data_2022.get("cost_of_goods_sold", "N/A")),
            "gross_profit": _normalize_na(data_2022.get("gross_profit", "N/A")),
            "operating_expense": _normalize_na(data_2022.get("operating_expense", "N/A")),
            "operating_income": _normalize_na(data_2022.get("operating_income", "N/A")),
            "net_profit": _normalize_na(data_2022.get("net_profit", "N/A")),
            "income_before_taxes": _normalize_na(data_2022.get("income_before_taxes", "N/A")),
            "tax_expense": _normalize_na(data_2022.get("tax_expense", "N/A")),
            "interest_expense": _normalize_na(data_2022.get("interest_expense", "N/A"))
        },
        "multiplier": _normalize_na(multiplier),
        "currency": _normalize_na(currency)
    }
    
    
def extract_s2_2(md_file_2024: str, md_file_2023: str, top_k: int, model: str):

    lang_key = TARGET_LANGUAGE.value
    search_queries = get_queries("s2_2", lang_key)
    context_2024 = retrieve_relevant_text(search_queries, top_k, md_file_2024)
    context_2023 = retrieve_relevant_text(search_queries, top_k, md_file_2023)
    
    prompt_2024 = build_s2_2_prompt(context_2024, 2024, CURRENCY_CODE, MULTIPLIER, TARGET_LANGUAGE)
    prompt_2023 = build_s2_2_prompt(context_2023, 2023, CURRENCY_CODE, MULTIPLIER, TARGET_LANGUAGE)
    
    result_2024 = None
    result_2023 = None
    
    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a financial data extraction expert. Extract exact values from balance sheet statements. Return valid JSON only."},
                {"role": "user", "content": prompt_2024}
            ],
            temperature=0,
            max_tokens=2000
        )
        result_2024 = _safe_json_from_llm(response.choices[0].message.content)
        print(f"✓ Successfully extracted from 2024 report")
    except Exception as e:
        print(f"✗ Error extracting from 2024 report: {e}")
        result_2024 = {}

    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a financial data extraction expert. Extract exact values from balance sheet statements. Return valid JSON only."},
                {"role": "user", "content": prompt_2023}
            ],
            temperature=0,
            max_tokens=2000
        )
        result_2023 = _safe_json_from_llm(response.choices[0].message.content)
        print(f"✓ Successfully extracted from 2023 report")
    except Exception as e:
        print(f"✗ Error extracting from 2023 report: {e}")
        result_2023 = {}

    def _merge_year_data(primary_data, fallback_data):
        if not primary_data:
            return fallback_data or {}
        if not fallback_data:
            return primary_data or {}
        
        merged = {}
        all_keys = set(primary_data.keys()) | set(fallback_data.keys())
        
        for key in all_keys:
            primary_value = primary_data.get(key, "N/A")
            fallback_value = fallback_data.get(key, "N/A")
            
            if primary_value not in ["N/A", None, ""]:
                merged[key] = primary_value
            else:
                merged[key] = fallback_value
        
        return merged
    
    data_2024 = result_2024.get("2024", {}) if result_2024 else {}

    data_2023_from_2024 = result_2024.get("2023", {}) if result_2024 else {}
    data_2023_from_2023 = result_2023.get("2023", {}) if result_2023 else {}
    data_2023 = _merge_year_data(data_2023_from_2024, data_2023_from_2023)
 
    data_2022_from_2024 = result_2024.get("2022", {}) if result_2024 else {}
    data_2022_from_2023 = result_2023.get("2022", {}) if result_2023 else {}
    data_2022 = _merge_year_data(data_2022_from_2024, data_2022_from_2023)    

    multiplier = result_2024.get("multiplier", result_2023.get("multiplier", "Units") if result_2023 else "Units") if result_2024 else "Units"
    currency = result_2024.get("currency", result_2023.get("currency", "USD") if result_2023 else "USD") if result_2024 else "USD"
        
    return {
        "2024": {
            "total_assets": _normalize_na(data_2024.get("total_assets", "N/A")),
            "current_assets": _normalize_na(data_2024.get("current_assets", "N/A")),
            "non_current_assets": _normalize_na(data_2024.get("non_current_assets", "N/A")),
            "total_liabilities": _normalize_na(data_2024.get("total_liabilities", "N/A")),
            "current_liabilities": _normalize_na(data_2024.get("current_liabilities", "N/A")),
            "non_current_liabilities": _normalize_na(data_2024.get("non_current_liabilities", "N/A")),
            "shareholders_equity": _normalize_na(data_2024.get("shareholders_equity", "N/A")),
            "retained_earnings": _normalize_na(data_2024.get("retained_earnings", "N/A")),
            "total_equity_and_liabilities": _normalize_na(data_2024.get("total_equity_and_liabilities", "N/A")),
            "inventories": _normalize_na(data_2024.get("inventories", "N/A")),
            "prepaid_expenses": _normalize_na(data_2024.get("prepaid_expenses", "N/A")),
        },
        "2023": {
            "total_assets": _normalize_na(data_2023.get("total_assets", "N/A")),
            "current_assets": _normalize_na(data_2023.get("current_assets", "N/A")),
            "non_current_assets": _normalize_na(data_2023.get("non_current_assets", "N/A")),
            "total_liabilities": _normalize_na(data_2023.get("total_liabilities", "N/A")),
            "current_liabilities": _normalize_na(data_2023.get("current_liabilities", "N/A")),
            "non_current_liabilities": _normalize_na(data_2023.get("non_current_liabilities", "N/A")),
            "shareholders_equity": _normalize_na(data_2023.get("shareholders_equity", "N/A")),
            "retained_earnings": _normalize_na(data_2023.get("retained_earnings", "N/A")),
            "total_equity_and_liabilities": _normalize_na(data_2023.get("total_equity_and_liabilities", "N/A")),
            "inventories": _normalize_na(data_2023.get("inventories", "N/A")),
            "prepaid_expenses": _normalize_na(data_2023.get("prepaid_expenses", "N/A")),
        },
        "2022": {
            "total_assets": _normalize_na(data_2022.get("total_assets", "N/A")),
            "current_assets": _normalize_na(data_2022.get("current_assets", "N/A")),
            "non_current_assets": _normalize_na(data_2022.get("non_current_assets", "N/A")),
            "total_liabilities": _normalize_na(data_2022.get("total_liabilities", "N/A")),
            "current_liabilities": _normalize_na(data_2022.get("current_liabilities", "N/A")),
            "non_current_liabilities": _normalize_na(data_2022.get("non_current_liabilities", "N/A")),
            "shareholders_equity": _normalize_na(data_2022.get("shareholders_equity", "N/A")),
            "retained_earnings": _normalize_na(data_2022.get("retained_earnings", "N/A")),
            "total_equity_and_liabilities": _normalize_na(data_2022.get("total_equity_and_liabilities", "N/A")),
            "inventories": _normalize_na(data_2022.get("inventories", "N/A")),
            "prepaid_expenses": _normalize_na(data_2022.get("prepaid_expenses", "N/A")),
        },
        "multiplier": _normalize_na(multiplier),
        "currency": _normalize_na(currency),
    }
    

def extract_s2_3(md_file_2024: str, md_file_2023: str, top_k: int, model: str = "gpt-4.1-mini"):

    lang_key = TARGET_LANGUAGE.value
    search_queries = get_queries("s2_3", lang_key)
    
    context_2024 = retrieve_relevant_text(search_queries, top_k, md_file_2024)
    context_2023 = retrieve_relevant_text(search_queries, top_k, md_file_2023)
    
    prompt_2024 = build_s2_3_prompt(context_2024, 2024, CURRENCY_CODE, MULTIPLIER, TARGET_LANGUAGE=lang_key)
    prompt_2023 = build_s2_3_prompt(context_2023, 2023, CURRENCY_CODE, MULTIPLIER, TARGET_LANGUAGE=lang_key)

    result_2024 = None
    result_2023 = None
    
    try:
        response_2024 = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a financial data extraction expert. Extract exact values from financial statements. Return valid JSON only."},
                {"role": "user", "content": prompt_2024}
            ],
            temperature=0,
            max_tokens=2000
        )
        result_2024 = _safe_json_from_llm(response_2024.choices[0].message.content)
        print(f"✓ Successfully extracted from 2024 report")
    except Exception as e:
        print(f"✗ Error extracting from 2024 report: {e}")
        result_2024 = {}

    try:
        response_2023 = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a financial data extraction expert. Extract exact values from financial statements. Return valid JSON only."},
                {"role": "user", "content": prompt_2023}
            ],
            temperature=0,
            max_tokens=2000
        )
        result_2023 = _safe_json_from_llm(response_2023.choices[0].message.content)
        print(f"✓ Successfully extracted from 2023 report")
    except Exception as e:
        print(f"✗ Error extracting from 2023 report: {e}")
        result_2023 = {}
        
    def _merge_year_data(primary_data, fallback_data):
        """
        Merge year data field-by-field.
        Use primary if value is not N/A, otherwise use fallback.
        """
        if not primary_data:
            return fallback_data or {}
        if not fallback_data:
            return primary_data or {}
        
        merged = {}
        all_keys = set(primary_data.keys()) | set(fallback_data.keys())
        
        for key in all_keys:
            primary_value = primary_data.get(key, "N/A")
            fallback_value = fallback_data.get(key, "N/A")
            
            # Use primary unless it's N/A, then use fallback
            if primary_value not in ["N/A", None, ""]:
                merged[key] = primary_value
            else:
                merged[key] = fallback_value
        
        return merged

    data_2024 = result_2024.get("2024", {}) if result_2024 else {}

    # For 2023: merge 2024 report and 2023 report field-by-field
    data_2023_from_2024 = result_2024.get("2023", {}) if result_2024 else {}
    data_2023_from_2023 = result_2023.get("2023", {}) if result_2023 else {}
    data_2023 = _merge_year_data(data_2023_from_2024, data_2023_from_2023)

    # For 2022: merge 2024 report and 2023 report field-by-field  
    data_2022_from_2024 = result_2024.get("2022", {}) if result_2024 else {}
    data_2022_from_2023 = result_2023.get("2022", {}) if result_2023 else {}
    data_2022 = _merge_year_data(data_2022_from_2024, data_2022_from_2023)    

    multiplier = result_2024.get("multiplier", result_2023.get("multiplier", "Units") if result_2023 else "Units") if result_2024 else "Units"
    currency = result_2024.get("currency", result_2023.get("currency", "USD") if result_2023 else "USD") if result_2024 else "USD"
    
    return {
        "2024": {
            "net_cash_from_operations": _normalize_na(data_2024.get("net_cash_from_operations", "N/A")),
            "net_cash_from_investing": _normalize_na(data_2024.get("net_cash_from_investing", "N/A")),
            "net_cash_from_financing": _normalize_na(data_2024.get("net_cash_from_financing", "N/A")),
            "net_increase_decrease_in_cash": _normalize_na(data_2024.get("net_increase_decrease_in_cash", "N/A")),
            "dividends": _normalize_na(data_2024.get("dividends", "N/A"))
        },
        "2023": {
            "net_cash_from_operations": _normalize_na(data_2023.get("net_cash_from_operations", "N/A")),
            "net_cash_from_investing": _normalize_na(data_2023.get("net_cash_from_investing", "N/A")),
            "net_cash_from_financing": _normalize_na(data_2023.get("net_cash_from_financing", "N/A")),
            "net_increase_decrease_in_cash": _normalize_na(data_2023.get("net_increase_decrease_in_cash", "N/A")),
            "dividends": _normalize_na(data_2023.get("dividends", "N/A"))
        },
        "2022": {
            "net_cash_from_operations": _normalize_na(data_2022.get("net_cash_from_operations", "N/A")),
            "net_cash_from_investing": _normalize_na(data_2022.get("net_cash_from_investing", "N/A")),
            "net_cash_from_financing": _normalize_na(data_2022.get("net_cash_from_financing", "N/A")),
            "net_increase_decrease_in_cash": _normalize_na(data_2022.get("net_increase_decrease_in_cash", "N/A")),
            "dividends": _normalize_na(data_2022.get("dividends", "N/A"))
        },
        "multiplier": _normalize_na(multiplier),
        "currency": _normalize_na(currency),
    }

def extract_s2_4(report):
    """
    Compute Key Financial Metrics (S2.4) using extracted data from Income Statement,
    Balance Sheet, and Cash Flow Statement. Stores results into report.key_financial_metrics.
    """

    inc = report.income_statement
    bal = report.balance_sheet
    cf = report.cash_flow_statement
    km = report.key_financial_metrics

    def to_float(x):
        """
        Parse numbers with:
        - parentheses for negatives
        - currency symbols
        - 'k' suffix (thousands)
        - European or US thousands/decimal separators (e.g., '17.102.428', '36,072.95', '1.234.567,89')
        Returns None if not parseable.
        """
        if x in (None, "N/A", "", "-", "--"):
            return None
        s = str(x).strip()

        # Strip currency symbols & spaces
        if s and s[0] in ("£", "$", "€", "¥"):
            s = s[1:].strip()
        s = s.replace("\xa0", "").replace(" ", "")

        # Handle parentheses negatives
        neg = s.startswith("(") and s.endswith(")")
        if neg:
            s = s[1:-1].strip()

        # Handle 'k' suffix (thousands)
        mult = 1_000 if s.lower().endswith("k") else 1
        if mult != 1:
            s = s[:-1].strip()

        # Detect separators
        has_dot = "." in s
        has_comma = "," in s

        try:
            if has_dot and has_comma:
                # Decide decimal sep by the rightmost sep
                last_dot = s.rfind(".")
                last_comma = s.rfind(",")
                if last_dot > last_comma:
                    # '.' is decimal; remove commas
                    s = s.replace(",", "")
                else:
                    # ',' is decimal; remove dots, then make ',' -> '.'
                    s = s.replace(".", "").replace(",", ".")
            elif has_dot and s.count(".") > 1:
                # Multi-dot grouping -> remove all dots
                s = s.replace(".", "")
            elif has_comma and s.count(",") > 1:
                # Multi-comma grouping -> remove all commas
                s = s.replace(",", "")
            else:
                # Single separator (either '.' or ','), assume decimal
                s = s.replace(",", ".")
            val = float(s) * mult
            return -val if neg else val
        except Exception:
            return None

    def safe_div(a, b):
        """Safely divide two numbers; return None if a or b is None or b == 0."""
        if a is None or b is None or b == 0:
            return None
        try:
            return a / b
        except Exception:
            return None

    def pct(v):
        """Convert ratio to percentage string, using parentheses for negatives."""
        if v is None:
            return "N/A"
        if v < 0:
            return f"({abs(v * 100):.2f}%)"
        return f"{v * 100:.2f}%"

    def avg(a, b):
        """Average only if both are present."""
        if a is None or b is None:
            return None
        return (a + b) / 2

    for year in ["2024", "2023", "2022"]:
        # Income
        rev  = to_float(getattr(inc.revenue, f"year_{year}", None))
        cogs = to_float(getattr(inc.cost_of_goods_sold, f"year_{year}", None))
        if cogs is not None:
            cogs = abs(cogs)

        op_inc = to_float(getattr(inc.operating_income, f"year_{year}", None))
        net_inc = to_float(getattr(inc.net_profit, f"year_{year}", None))
        interest = to_float(getattr(inc.interest_expense, f"year_{year}", None))
        if interest is not None:
            interest = abs(interest)

        tax_exp = to_float(getattr(inc.income_tax_expense, f"year_{year}", None))  # keep sign
        inc_before_tax = to_float(getattr(inc.income_before_income_taxes, f"year_{year}", None))

        # Balance
        curr_assets = to_float(getattr(bal.current_assets, f"year_{year}", None))
        curr_liab   = to_float(getattr(bal.current_liabilities, f"year_{year}", None))
        if curr_liab is not None:
            curr_liab = abs(curr_liab)

        invent  = to_float(getattr(bal.inventories, f"year_{year}", None))
        prepaid = to_float(getattr(bal.prepaid_expenses, f"year_{year}", None))

        if invent is None or prepaid is None or curr_assets is None:
            quick_ratio = None
        else:
            quick_ratio = safe_div(curr_assets - invent - prepaid, curr_liab)

        total_assets = to_float(getattr(bal.total_assets, f"year_{year}", None))
        total_liab   = to_float(getattr(bal.total_liabilities, f"year_{year}", None))
        if total_liab is not None:
            total_liab = abs(total_liab)

        equity = to_float(getattr(bal.shareholders_equity, f"year_{year}", None))

        # Cash flow
        divs = to_float(getattr(cf.dividends, f"year_{year}", None))
        if divs is not None:
            divs = abs(divs)

        # Averages (use previous year)
        prev_year = str(int(year) - 1)
        total_assets_prev = getattr(bal.total_assets, f"year_{prev_year}", None)
        equity_prev       = getattr(bal.shareholders_equity, f"year_{prev_year}", None)

        total_assets_prev = to_float(total_assets_prev) if total_assets_prev is not None else None
        equity_prev       = to_float(equity_prev)       if equity_prev is not None       else None

        avg_assets = avg(total_assets, total_assets_prev)   # -> None if prev missing
        avg_equity = avg(equity, equity_prev)

        # --- Metrics (raw ratios) ---
        gross_margin     = safe_div((rev - cogs) if (rev is not None and cogs is not None) else None, rev)
        op_margin        = safe_div(op_inc, rev)
        net_margin       = safe_div(net_inc, rev)
        curr_ratio       = safe_div(curr_assets, curr_liab)
        # quick_ratio computed above (may be None)
        interest_coverage= safe_div(op_inc, interest)
        asset_turnover   = safe_div(rev, avg_assets)
        debt_to_equity   = safe_div(total_liab, equity)
        roe              = safe_div(net_inc, avg_equity)
        roa              = safe_div(net_inc, avg_assets)
        eff_tax_rate     = safe_div(tax_exp, inc_before_tax)
        payout_ratio     = safe_div(divs, net_inc)

        # --- Assign as PERCENT STRINGS (parentheses for negatives) ---
        setattr(km.gross_margin,          f"year_{year}", pct(gross_margin))
        setattr(km.operating_margin,      f"year_{year}", pct(op_margin))
        setattr(km.net_profit_margin,     f"year_{year}", pct(net_margin))
        setattr(km.current_ratio,         f"year_{year}", pct(curr_ratio))
        setattr(km.quick_ratio,           f"year_{year}", pct(quick_ratio))
        setattr(km.interest_coverage,     f"year_{year}", pct(interest_coverage))
        setattr(km.asset_turnover,        f"year_{year}", pct(asset_turnover))
        setattr(km.debt_to_equity,        f"year_{year}", pct(debt_to_equity))
        setattr(km.return_on_equity,      f"year_{year}", pct(roe))
        setattr(km.return_on_assets,      f"year_{year}", pct(roa))
        setattr(km.effective_tax_rate,    f"year_{year}", pct(eff_tax_rate))
        setattr(km.dividend_payout_ratio, f"year_{year}", pct(payout_ratio))



def merge_revenue_dicts(data_2024: dict, data_2023: dict) -> dict:
    """
    Merge 2024 and 2023 outputs so:
    - 2024 values have precedence
    - If a 2024 field is "N/A", use 2023's corresponding field
    """
    merged = {}
    for year in ["2024", "2023", "2022"]:
        merged.setdefault(year, {})
        for field in ["revenue_by_product_service", "revenue_by_region"]:
            v2024 = data_2024.get(year, {}).get(field, "N/A")
            v2023 = data_2023.get(year, {}).get(field, "N/A")
            if not v2024 or v2024.strip().upper() in {"N/A", "NA", ""}:
                merged[year][field] = v2023
            else:
                merged[year][field] = v2024
    return merged

def extract_s2_5(md_file_2024: str, md_file_2023: str, top_k: int = 15, model: str = "gpt-4.1-mini"):

    lang_key = TARGET_LANGUAGE.value
    search_queries = get_queries("s2_5", lang_key)
    context_2024 = retrieve_relevant_text(search_queries, top_k, md_file_2024)
    context_2023 = retrieve_relevant_text(search_queries, top_k, md_file_2023)
    
    prompt_2024 = build_s2_5_prompt(context_2024, COMPANY_NAME, TARGET_LANGUAGE)
    prompt_2023 = build_s2_5_prompt(context_2023, COMPANY_NAME, TARGET_LANGUAGE)
    
    response_2024 = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a precise financial data extractor. Return only JSON."},
            {"role": "user", "content": prompt_2024}
        ],
        temperature=0
    )

    response_2023 = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a precise financial data extractor. Return only JSON."},
            {"role": "user", "content": prompt_2023}
        ],
        temperature=0
    )

    data_2024 = _safe_json_from_llm(response_2024.choices[0].message.content)
    data_2023 = _safe_json_from_llm(response_2023.choices[0].message.content)
    # print(f"DEBUG - S2.5 2024 extraction result: {data_2024}")
    merged = merge_revenue_dicts(data_2024, data_2023)
    return merged

# ===================== Section 3: Business Analysis =====================  
          
def extract_s3_1(report, model: str = "gpt-4.1-mini"):
    
    inc = report.income_statement
    bal = report.balance_sheet
    cf = report.cash_flow_statement
    perf = report.operating_performance
    metrics = report.key_financial_metrics
    
    multiplier = getattr(report.income_statement, "primary_multiplier", "Millions")
    currency = getattr(report.income_statement, "primary_currency", "USD")
        
    # sufficient?
    financial_context = f"""
    
        
        Company: {COMPANY_NAME}
        INCOME STATEMENT:
        Revenue: 2024: {inc.revenue.year_2024}, 2023: {inc.revenue.year_2023}, 2022: {inc.revenue.year_2022}
        Cost of Goods Sold: 2024: {inc.cost_of_goods_sold.year_2024}, 2023: {inc.cost_of_goods_sold.year_2023}, 2022: {inc.cost_of_goods_sold.year_2022}
        Gross Profit: 2024: {inc.gross_profit.year_2024}, 2023: {inc.gross_profit.year_2023}, 2022: {inc.gross_profit.year_2022}
        Operating Income: 2024: {inc.operating_income.year_2024}, 2023: {inc.operating_income.year_2023}, 2022: {inc.operating_income.year_2022}
        Operating Expense: 2024: {inc.operating_expense.year_2024}, 2023: {inc.operating_expense.year_2023}, 2022: {inc.operating_expense.year_2022}
        Income Tax Expense: 2024: {inc.income_tax_expense.year_2024}, 2023: {inc.income_tax_expense.year_2023}, 2022: {inc.income_tax_expense.year_2022}
        Net Profit: 2024: {inc.net_profit.year_2024}, 2023: {inc.net_profit.year_2023}, 2022: {inc.net_profit.year_2022}

        KEY METRICS:
        Gross Margin: 2024: {metrics.gross_margin.year_2024}, 2023: {metrics.gross_margin.year_2023}, 2022: {metrics.gross_margin.year_2022}
        Operating Margin: 2024: {metrics.operating_margin.year_2024}, 2023: {metrics.operating_margin.year_2023}, 2022: {metrics.operating_margin.year_2022}
        Net Profit Margin: 2024: {metrics.net_profit_margin.year_2024}, 2023: {metrics.net_profit_margin.year_2023}, 2022: {metrics.net_profit_margin.year_2022}
        Effective Tax Rate: 2024: {metrics.effective_tax_rate.year_2024}, 2023: {metrics.effective_tax_rate.year_2023}, 2022: {metrics.effective_tax_rate.year_2022}

        OPERATING PERFORMANCE:
        Revenue by Product/Service:
        2024: {perf.revenue_by_product_service.year_2024}
        2023: {perf.revenue_by_product_service.year_2023}
        2022: {perf.revenue_by_product_service.year_2022}

        Revenue by Geographic Region:
        2024: {perf.revenue_by_geographic_region.year_2024}
        2023: {perf.revenue_by_geographic_region.year_2023}
        2022: {perf.revenue_by_geographic_region.year_2022}
        
        """
    
    prompt = build_s3_1_prompt(financial_context, COMPANY_NAME, multiplier, currency, TARGET_LANGUAGE)
    
    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an expert financial analyst. Analyze only the provided data and provide insightful business interpretations. Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=2000
        )
        
        result = _safe_json_from_llm(response.choices[0].message.content)
        
        return {
            "revenue_direct_cost_dynamics": _normalize_na(result.get("revenue_direct_cost_dynamics", "N/A")),
            "operating_efficiency": _normalize_na(result.get("operating_efficiency", "N/A")),
            "external_oneoff_impact": _normalize_na(result.get("external_oneoff_impact", "N/A"))
        }
        
    except Exception as e:
        print(f"Error in extract_s3_1: {e}")
        

def build_financial_context_s3_2(report, year: int):
    """
    Build financial context for Section 3.2 Financial Performance Summary.
    For each target year, include its own data and the immediately previous year's data
    to enable direct year-over-year comparisons.
    """

    inc = report.income_statement
    bal = report.balance_sheet
    cf = report.cash_flow_statement
    perf = report.operating_performance
    metrics = report.key_financial_metrics

    previous_year = year - 1 if hasattr(inc.revenue, f"year_{year - 1}") else None
    years_to_include = [y for y in [previous_year, year] if y is not None]

    context = f"""
    COMPANY: {COMPANY_NAME}
    CURRENCY: {getattr(inc, 'primary_currency', 'N/A')}
    MULTIPLIER: {getattr(inc, 'primary_multiplier', 'N/A')}
    TARGET REPORTING PERIOD: {year}
    COMPARISON YEARS: {', '.join(map(str, years_to_include))}
    """

    def line(label, obj):
        # Include both target and previous year values
        values = []
        for y in years_to_include:
            v = getattr(obj, f"year_{y}", "N/A")
            values.append(f"{y}: {v}")
        return f"{label}: {' | '.join(values)}"

    context += f"""
    INCOME STATEMENT DATA:
    {line("Revenue", inc.revenue)}
    {line("Cost of Goods Sold", inc.cost_of_goods_sold)}
    {line("Gross Profit", inc.gross_profit)}
    {line("Operating Expense", inc.operating_expense)}
    {line("Operating Income", inc.operating_income)}
    {line("Net Profit", inc.net_profit)}
    {line("Income Before Taxes", inc.income_before_income_taxes)}
    {line("Income Tax Expense", inc.income_tax_expense)}
    {line("Interest Expense", inc.interest_expense)}

    BALANCE SHEET DATA:
    {line("Total Assets", bal.total_assets)}
    {line("Current Assets", bal.current_assets)}
    {line("Non-Current Assets", bal.non_current_assets)}
    {line("Total Liabilities", bal.total_liabilities)}
    {line("Current Liabilities", bal.current_liabilities)}
    {line("Non-Current Liabilities", bal.non_current_liabilities)}
    {line("Shareholders' Equity", bal.shareholders_equity)}
    {line("Retained Earnings", bal.retained_earnings)}
    {line("Inventories", bal.inventories)}

    CASH FLOW DATA:
    {line("Net Cash from Operations", cf.net_cash_from_operations)}
    {line("Net Cash from Investing", cf.net_cash_from_investing)}
    {line("Net Cash from Financing", cf.net_cash_from_financing)}
    {line("Net Increase/Decrease in Cash", cf.net_increase_decrease_cash)}
    {line("Dividends", cf.dividends)}

    KEY FINANCIAL METRICS:
    {line("Gross Margin", metrics.gross_margin)}
    {line("Operating Margin", metrics.operating_margin)}
    {line("Net Profit Margin", metrics.net_profit_margin)}
    {line("Current Ratio", metrics.current_ratio)}
    {line("Debt to Equity", metrics.debt_to_equity)}
    {line("Return on Equity", metrics.return_on_equity)}
    {line("Return on Assets", metrics.return_on_assets)}
    {line("Effective Tax Rate", metrics.effective_tax_rate)}
    {line("Interest Coverage", metrics.interest_coverage)}
    {line("Asset Turnover", metrics.asset_turnover)}

    OPERATING PERFORMANCE:
    Revenue by Product/Service:
    """ + " | ".join([f"{y}: {getattr(perf.revenue_by_product_service, f'year_{y}', 'N/A')}" for y in years_to_include]) + """

    Revenue by Geographic Region:
    """ + " | ".join([f"{y}: {getattr(perf.revenue_by_geographic_region, f'year_{y}', 'N/A')}" for y in years_to_include]) + "\n"

    return context
    
def extract_s3_2(report, model = "gpt-4.1-mini"):

    financial_context_2024 = build_financial_context_s3_2(report, 2024)
    financial_context_2023 = build_financial_context_s3_2(report, 2023)
    
    company_name = getattr(report, 'company_name', 'N/A')
    
    prompt_2024 = build_s3_2_prompt(financial_context_2024, 2024, company_name, TARGET_LANGUAGE.value)
    prompt_2023 = build_s3_2_prompt(financial_context_2023, 2023, company_name, TARGET_LANGUAGE.value)
    
    result = {}
    
    try:
        response_2024 = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a financial analyst providing comprehensive performance analysis."},
                {"role": "user", "content": prompt_2024}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        data_2024 = _safe_json_from_llm(response_2024.choices[0].message.content)
        result.update(data_2024)
    except Exception as e:
        print(f"Error extracting S3.2 for 2024: {e}")
    
    try:
        response_2023 = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a financial analyst providing comprehensive performance analysis."},
                {"role": "user", "content": prompt_2023}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        data_2023 = _safe_json_from_llm(response_2023.choices[0].message.content)
        result.update(data_2023)
    except Exception as e:
        print(f"Error extracting S3.2 for 2023: {e}")
    return result
              
def extract_s3_3(md_file_2024: str, md_file_2023: str, top_k: int = 15, model: str = "gpt-4.1-mini"):
        
    lang_key = TARGET_LANGUAGE.value
    search_queries = get_queries("s3_3", lang_key)
    context_2024 = retrieve_relevant_text(search_queries, top_k, md_file_2024)
    context_2023 = retrieve_relevant_text(search_queries, top_k, md_file_2023)
    prompt_2024 = build_s3_3_prompt(context_2024, 2024, TARGET_LANGUAGE)
    prompt_2023 = build_s3_3_prompt(context_2023, 2023, TARGET_LANGUAGE)

    def _call_llm(prompt: str) -> dict:
        try:
            resp = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are an expert business analyst. Use only the provided context. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1200
            )
            return _safe_json_from_llm(resp.choices[0].message.content)
        except Exception as e:
            print(f"[S3.3] LLM error: {e}")
            return {}

    result_2024 = _call_llm(prompt_2024)
    result_2023 = _call_llm(prompt_2023)

    return {
        "business_model_2024": _normalize_na(result_2024.get("business_model", "N/A")),
        "business_model_2023": _normalize_na(result_2023.get("business_model", "N/A")),
        "market_position_2024": _normalize_na(result_2024.get("market_position", "N/A")),
        "market_position_2023": _normalize_na(result_2023.get("market_position", "N/A"))
    }

# ===================== Section 4: Risk Factors =====================  
        
def extract_s4_1(md_file_2024: str, md_file_2023: str, top_k: int = 15, model: str = "gpt-4.1-mini"):

    lang_key = TARGET_LANGUAGE.value
    search_queries = get_queries("s4_1", lang_key)
    context_2024 = retrieve_relevant_text(search_queries, top_k, md_file_2024)
    context_2023 = retrieve_relevant_text(search_queries, top_k, md_file_2023)
    
    prompt_2024 = build_s4_1_prompt(context_2024, 2024, COMPANY_NAME, TARGET_LANGUAGE)
    prompt_2023 = build_s4_1_prompt(context_2023, 2023, COMPANY_NAME, TARGET_LANGUAGE)
    
    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an expert risk analyst. Extract risk factor information from annual reports. Return valid JSON only."},
                {"role": "user", "content": prompt_2024}
            ],
            temperature=0,
            max_tokens=1500
    )
    except Exception as e:
        print(f"Error in _extract_risk_factors_single_year for {2024}: {e}")
    result_2024 = _safe_json_from_llm(response.choices[0].message.content)
    
    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an expert risk analyst. Extract risk factor information from annual reports. Return valid JSON only."},
                {"role": "user", "content": prompt_2023}
            ],
            temperature=0,
            max_tokens=1500
    )
    except Exception as e:
        print(f"Error in _extract_risk_factors_single_year for {2023}: {e}")
    result_2023 = _safe_json_from_llm(response.choices[0].message.content)
    
    return {
        "market_risks_2024": _normalize_na(result_2024.get("market_risks", "N/A")),
        "market_risks_2023": _normalize_na(result_2023.get("market_risks", "N/A")),
        "operational_risks_2024": _normalize_na(result_2024.get("operational_risks", "N/A")),
        "operational_risks_2023": _normalize_na(result_2023.get("operational_risks", "N/A")),
        "financial_risks_2024": _normalize_na(result_2024.get("financial_risks", "N/A")),
        "financial_risks_2023": _normalize_na(result_2023.get("financial_risks", "N/A")),
        "compliance_risks_2024": _normalize_na(result_2024.get("compliance_risks", "N/A")),
        "compliance_risks_2023": _normalize_na(result_2023.get("compliance_risks", "N/A"))
    }
            

# ===================== Section 5: Corporate Governance =====================
        
def extract_s5_1(md_file_2024: str, top_k: int = 15, model: str = "gpt-4.1-mini"):
    
    lang_key = TARGET_LANGUAGE.value
    search_queries = get_queries("s5_1", lang_key)
    
    context_2024 = retrieve_relevant_text(search_queries, top_k, md_file_2024)
    
    prompt_2024 = build_s5_1_prompt(context_2024, TARGET_LANGUAGE)

    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an expert corporate governance analyst. Extract board composition and executive compensation information from annual reports. Return valid JSON only."},
                {"role": "user", "content": prompt_2024}
            ],
            temperature=0,
            max_tokens=2000
        )
        
        result = _safe_json_from_llm(response.choices[0].message.content)
    
        board_members = []
        if "board_members" in result and isinstance(result["board_members"], list):
            for member in result["board_members"]:
                if isinstance(member, dict):
                    board_members.append({
                        "name": _normalize_na(member.get("name", "N/A")),
                        "position": _normalize_na(member.get("position", "N/A")),
                        "total_income": _normalize_na(member.get("total_income", "N/A"))
                    })
        
        return {"board_members": board_members}
        
    except Exception as e:
        print(f"Error in _extract_board_composition: {e}")

def extract_s5_2(md_file_2024: str, md_file_2023: str, top_k: int = 15, model: str = "gpt-4.1-mini"):

    lang_key = TARGET_LANGUAGE.value
    search_queries = get_queries("s5_2", lang_key)
    
    context_2024 = retrieve_relevant_text(search_queries, top_k, md_file_2024)
    context_2023 = retrieve_relevant_text(search_queries, top_k, md_file_2023)
    
    prompt_2024 = build_s5_2_prompt(context_2024, 2024, COMPANY_NAME, TARGET_LANGUAGE)
    prompt_2023 = build_s5_2_prompt(context_2023, 2023, COMPANY_NAME, TARGET_LANGUAGE)
    
    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an expert corporate governance analyst. Extract internal control information from annual reports. Return valid JSON only."},
                {"role": "user", "content": prompt_2024}
            ],
            temperature=0,
            max_tokens=2000
        )
        result_2024 = _safe_json_from_llm(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in _extract_internal_controls_single_year for {2024}: {e}")
         
    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an expert corporate governance analyst. Extract internal control information from annual reports. Return valid JSON only."},
                {"role": "user", "content": prompt_2023}
            ],
            temperature=0,
            max_tokens=2000
        )
        result_2023 = _safe_json_from_llm(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in _extract_internal_controls_single_year for {2023}: {e}")
         
    return {
        "risk_assessment_procedures_2024": _normalize_na(result_2024.get("risk_assessment_procedures", "N/A")),
        "risk_assessment_procedures_2023": _normalize_na(result_2023.get("risk_assessment_procedures", "N/A")),
        "control_activities_2024": _normalize_na(result_2024.get("control_activities", "N/A")),
        "control_activities_2023": _normalize_na(result_2023.get("control_activities", "N/A")),
        "monitoring_mechanisms_2024": _normalize_na(result_2024.get("monitoring_mechanisms", "N/A")),
        "monitoring_mechanisms_2023": _normalize_na(result_2023.get("monitoring_mechanisms", "N/A")),
        "identified_material_weaknesses_2024": _normalize_na(result_2024.get("identified_material_weaknesses", "N/A")),
        "identified_material_weaknesses_2023": _normalize_na(result_2023.get("identified_material_weaknesses", "N/A")),
        "effectiveness_2024": _normalize_na(result_2024.get("effectiveness", "N/A")),
        "effectiveness_2023": _normalize_na(result_2023.get("effectiveness", "N/A"))
    }
    
# ===================== Section 6: Future Outlook =====================

def extract_s6_1(md_file_2024: str, md_file_2023: str, top_k: int = 15, model: str = "gpt-4.1-mini"):
        
    lang_key = TARGET_LANGUAGE.value
    search_queries = get_queries("s6_1", lang_key)
    context_2024 = retrieve_relevant_text(search_queries, top_k, md_file_2024)
    context_2023 = retrieve_relevant_text(search_queries, top_k, md_file_2023)

    prompt_2024 = build_s6_1_prompt(context_2024, 2024, COMPANY_NAME, TARGET_LANGUAGE)
    prompt_2023 = build_s6_1_prompt(context_2023, 2023, COMPANY_NAME, TARGET_LANGUAGE)
    
    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an expert strategic analyst. Extract strategic direction information from annual reports. Return valid JSON only."},
                {"role": "user", "content": prompt_2024}
            ],
            temperature=0,
            max_tokens=2000
        )
        result_2024 = _safe_json_from_llm(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in _extract_strategic_direction_single_year for 2024: {e}")
        
    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an expert strategic analyst. Extract strategic direction information from annual reports. Return valid JSON only."},
                {"role": "user", "content": prompt_2023}
            ],
            temperature=0,
            max_tokens=2000
    )
        result_2023 = _safe_json_from_llm(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in _extract_strategic_direction_single_year for 2023: {e}")
        
    return {
        "mergers_acquisition_2024": _normalize_na(result_2024.get("mergers_acquisition", "N/A")),
        "mergers_acquisition_2023": _normalize_na(result_2023.get("mergers_acquisition", "N/A")),
        "new_technologies_2024": _normalize_na(result_2024.get("new_technologies", "N/A")),
        "new_technologies_2023": _normalize_na(result_2023.get("new_technologies", "N/A")),
        "organisational_restructuring_2024": _normalize_na(result_2024.get("organisational_restructuring", "N/A")),
        "organisational_restructuring_2023": _normalize_na(result_2023.get("organisational_restructuring", "N/A"))
    }

def extract_s6_2(md_file_2024: str, md_file_2023: str, top_k: int = 15, model: str = "gpt-4.1-mini"):

    lang_key = TARGET_LANGUAGE.value
    search_queries = get_queries("s6_2", lang_key)
    context_2024 = retrieve_relevant_text(search_queries, top_k, md_file_2024)
    context_2023 = retrieve_relevant_text(search_queries, top_k, md_file_2023)

    prompt_2024 = build_s6_2_prompt(context_2024, 2024, COMPANY_NAME, TARGET_LANGUAGE)
    prompt_2023 = build_s6_2_prompt(context_2023, 2023, COMPANY_NAME, TARGET_LANGUAGE)
        
    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an expert business analyst. Extract challenges and uncertainties information from annual reports. Return valid JSON only."},
                {"role": "user", "content": prompt_2024}
            ],
            temperature=0,
            max_tokens=1600
        )
        result_2024 = _safe_json_from_llm(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in _extract_challenges_uncertainties_single_year for {2024}: {e}")

    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an expert business analyst. Extract challenges and uncertainties information from annual reports. Return valid JSON only."},
                {"role": "user", "content": prompt_2023}
            ],
            temperature=0,
            max_tokens=1600
        )
        result_2023 = _safe_json_from_llm(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in _extract_challenges_uncertainties_single_year for {2023}: {e}")
    
    return {
        "economic_challenges_2024": _normalize_na(result_2024.get("economic_challenges", "N/A")),
        "economic_challenges_2023": _normalize_na(result_2023.get("economic_challenges", "N/A")),
        "competitive_pressures_2024": _normalize_na(result_2024.get("competitive_pressures", "N/A")),
        "competitive_pressures_2023": _normalize_na(result_2023.get("competitive_pressures", "N/A"))
    }

def extract_s6_3(md_file_2024: str, md_file_2023: str, top_k: int = 15, model: str = "gpt-4.1-mini"):

    lang_key = TARGET_LANGUAGE.value
    search_queries = get_queries("s6_3", lang_key)
    context_2024 = retrieve_relevant_text(search_queries, top_k, md_file_2024)
    context_2023 = retrieve_relevant_text(search_queries, top_k, md_file_2023)
    
    prompt_2024 = build_s6_3_prompt(context_2024, 2024, COMPANY_NAME, TARGET_LANGUAGE)
    prompt_2023 = build_s6_3_prompt(context_2023, 2023, COMPANY_NAME, TARGET_LANGUAGE)
    
    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an expert innovation analyst. Extract innovation and development information from annual reports. Return valid JSON only."},
                {"role": "user", "content": prompt_2024}
            ],
            temperature=0,
            max_tokens=1200
        )
        result_2024 = _safe_json_from_llm(response.choices[0].message.content)    
    except Exception as e:
        print(f"Error in _extract_innovation_development_single_year for {2024}: {e}")

    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an expert innovation analyst. Extract innovation and development information from annual reports. Return valid JSON only."},
                {"role": "user", "content": prompt_2023}
            ],
            temperature=0,
            max_tokens=1200
        )
        result_2023 = _safe_json_from_llm(response.choices[0].message.content)    
    except Exception as e:
        print(f"Error in _extract_innovation_development_single_year for {2023}: {e}")

    return {
        "rd_investments_2024": _normalize_na(result_2024.get("rd_investments", "N/A")),
        "rd_investments_2023": _normalize_na(result_2023.get("rd_investments", "N/A")),
        "new_product_launches_2024": _normalize_na(result_2024.get("new_product_launches", "N/A")),
        "new_product_launches_2023": _normalize_na(result_2023.get("new_product_launches", "N/A"))
    }
    
# ===================== TEST EXTRACT =====================

def _slugify(name: str) -> str:
    name = (name or "").strip() or "report"
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "report"

def _atomic_write(text: str, out_path: str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
        tmp.write(text)
        tmp_path = tmp.name
    shutil.move(tmp_path, out_path) 

def save_partial_report(report, output_path: str, currency_code: str = "USD"):
    """
    Safely render and save the current report state to disk.
    Uses the DDRGenerator so formatting is identical to the final output.
    """
    try:
        gen = DDRGenerator(report, currency_code=currency_code)
        markdown = gen.generate_full_report()
        _atomic_write(markdown, output_path)
        print(f"[partial-save] Wrote snapshot to: {output_path}")
    except Exception as e:
        print(f"[partial-save][WARN] Failed to write snapshot: {e}")


def extract(md_file1: str, md_file2: str, *, currency_code: str = "USD", target_lang: Lang = Lang.EN):
    """
    Modified extract function using FAISS-based RAG search instead of section ranking.
    """

    start_time = time.time()
    print(f"Starting RAG extraction pipeline at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    report = CompanyReport()

    set_target_language(target_lang)
    print(f"Target language set to: {target_lang.name}")
    report.meta_output_lang = str(target_lang)
    
    md_path_2024 = Path(md_file1)
    md_path_2023 = Path(md_file2)
    md_file_2024 = md_path_2024.stem
    md_file_2023 = md_path_2023.stem
    
    company_from_filename = md_file_2024.split('_')[0] if '_' in md_file_2024 else md_file_2024
    slug = _slugify(company_from_filename)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    company_folder = f"artifacts/{slug}"
    partial_path = f"{company_folder}/partial/{timestamp}_report.md"
    print(f"Report will be saved to company folder: {company_folder}/")
    
    def checkpoint(section_label: str):
        print(f"Saving partial after: {section_label}")
        save_partial_report(report, partial_path, currency_code=currency_code)
        
    
    print("\n" + "="*60)
    print("PROCESSING: Building FAISS Embeddings")
    print("="*60)
    
    # Build embeddings if not exists
    jsonl_file_2024_path = f"data/sections_report/{md_file_2024}.jsonl"
    jsonl_file_2023_path = f"data/sections_report/{md_file_2023}.jsonl"
    
    build_section_embeddings(jsonl_file_2024_path, f"data/parsed/{md_file_2024}.md")
    build_section_embeddings(jsonl_file_2023_path, f"data/parsed/{md_file_2023}.md")
    
    print("\n" + "="*60)
    print("📋 PROCESSING: S1.1 - Basic Information (2024 with RAG)")
    print("="*60)
    
    company_name, establishment_date, headquarters = extract_s1_1(md_file_2024, top_k=25, model="gpt-4.1-mini")

    report.basic_info.company_name = company_name
    report.basic_info.establishment_date = establishment_date
    report.basic_info.headquarters_location = headquarters
    set_company_name(company_name)

    print("✅ COMPLETED: S1.1 - Basic Information")
    
    print("\n" + "="*60)
    print("PROCESSING: S1.2 - Core Competencies (2024 + 2023 with RAG)")
    print("="*60)
    
    core_comp_2024 = extract_s1_2(md_file=md_file_2024, top_k=12, year=2024, model="gpt-4.1-mini")
    core_comp_2023 = extract_s1_2(md_file=md_file_2023, top_k=12, year=2023, model="gpt-4.1-mini")
    core_comp = merge_core_competencies(core_comp_2024, core_comp_2023)
    
    # Save to report
    report.core_competencies.innovation_advantages.report_2024 = str(core_comp.get("Innovation Advantages", {}).get("2024", "N/A"))
    report.core_competencies.innovation_advantages.report_2023 = str(core_comp.get("Innovation Advantages", {}).get("2023", "N/A"))
    report.core_competencies.product_advantages.report_2024 = str(core_comp.get("Product Advantages", {}).get("2024", "N/A"))
    report.core_competencies.product_advantages.report_2023 = str(core_comp.get("Product Advantages", {}).get("2023", "N/A"))
    report.core_competencies.brand_recognition.report_2024 = str(core_comp.get("Brand Recognition", {}).get("2024", "N/A"))
    report.core_competencies.brand_recognition.report_2023 = str(core_comp.get("Brand Recognition", {}).get("2023", "N/A"))
    report.core_competencies.reputation_ratings.report_2024 = str(core_comp.get("Reputation Ratings", {}).get("2024", "N/A"))
    report.core_competencies.reputation_ratings.report_2023 = str(core_comp.get("Reputation Ratings", {}).get("2023", "N/A"))
    
    print("✅ COMPLETED: S1.2 - Core Competencies")
    
    print("\n" + "="*60)
    print("PROCESSING: S1.3 - Mission & Vision (2024 with RAG)")
    print("="*60)
    
    # only use 2024's report for mission & vision
    mv = extract_s1_3(md_file_2024, top_k=25, model="gpt-4.1-mini")
    
     # Save to report
    
    report.mission_vision.mission_statement = mv['mission']
    report.mission_vision.vision_statement = mv['vision']
    report.mission_vision.core_values = mv['core_values']
    
    print("✅ COMPLETED: S1.3 - Mission & Vision")
    # checkpoint("Section 1 - Company Overview (S1.1-S1.3)")
    
    print("\n" + "="*60)
    print("PROCESSING: S2.1 - Income Statement (with RAG)")
    print("="*60)
    
    # Use FAISS search for income statement
    income_data = extract_s2_1(md_file_2024, md_file_2023, top_k=12, model="gpt-4.1-mini")

    income_data = fill_income_data(income_data)

    def getv(y, k, default="N/A"):
        return income_data.get(y, {}).get(k, default)

    # Assign to report (no KeyErrors if a year/key is missing)
    report.income_statement.revenue.year_2024 = getv("2024", "revenue")
    report.income_statement.revenue.year_2023 = getv("2023", "revenue")
    report.income_statement.revenue.year_2022 = getv("2022", "revenue")

    report.income_statement.cost_of_goods_sold.year_2024 = getv("2024", "cost_of_goods_sold")
    report.income_statement.cost_of_goods_sold.year_2023 = getv("2023", "cost_of_goods_sold")
    report.income_statement.cost_of_goods_sold.year_2022 = getv("2022", "cost_of_goods_sold")

    report.income_statement.gross_profit.year_2024 = getv("2024", "gross_profit")
    report.income_statement.gross_profit.year_2023 = getv("2023", "gross_profit")
    report.income_statement.gross_profit.year_2022 = getv("2022", "gross_profit")

    report.income_statement.operating_expense.year_2024 = getv("2024", "operating_expense")
    report.income_statement.operating_expense.year_2023 = getv("2023", "operating_expense")
    report.income_statement.operating_expense.year_2022 = getv("2022", "operating_expense")

    report.income_statement.operating_income.year_2024 = getv("2024", "operating_income")
    report.income_statement.operating_income.year_2023 = getv("2023", "operating_income")
    report.income_statement.operating_income.year_2022 = getv("2022", "operating_income")

    report.income_statement.net_profit.year_2024 = getv("2024", "net_profit")
    report.income_statement.net_profit.year_2023 = getv("2023", "net_profit")
    report.income_statement.net_profit.year_2022 = getv("2022", "net_profit")

    report.income_statement.income_before_income_taxes.year_2024 = getv("2024", "income_before_taxes")
    report.income_statement.income_before_income_taxes.year_2023 = getv("2023", "income_before_taxes")
    report.income_statement.income_before_income_taxes.year_2022 = getv("2022", "income_before_taxes")

    report.income_statement.income_tax_expense.year_2024 = getv("2024", "tax_expense")
    report.income_statement.income_tax_expense.year_2023 = getv("2023", "tax_expense")
    report.income_statement.income_tax_expense.year_2022 = getv("2022", "tax_expense")

    report.income_statement.interest_expense.year_2024 = getv("2024", "interest_expense")
    report.income_statement.interest_expense.year_2023 = getv("2023", "interest_expense")
    report.income_statement.interest_expense.year_2022 = getv("2022", "interest_expense")

    report.income_statement.primary_currency = income_data.get("currency", "N/A")
    report.income_statement.primary_multiplier = income_data.get("multiplier", "N/A")
    
    set_currency_code(report.income_statement.primary_currency)
    set_multiplier(report.income_statement.primary_multiplier)
    
    print("✅ S2.1 Income Statement completed")
    print(f"   Revenue 2024: {income_data['2024']['revenue']} | 2023: {income_data['2023']['revenue']}")
    print(f"   Net Profit 2024: {income_data['2024']['net_profit']} | 2023: {income_data['2023']['net_profit']}")
    
    print("✅ COMPLETED: S2.1 - Income Statement")
    
    # checkpoint("Section 2 - Financial Performance (S2.1 Income Statement)")

    print("\n" + "="*60)
    print("PROCESSING: S2.2 - Balance Sheet (with RAG)")
    print("="*60)
    
    # Use FAISS search for balance sheet
    balance_data = extract_s2_2(md_file_2024, md_file_2023, top_k=20, model="gpt-4.1-mini")
    balance_data = fill_missing_balance_sheet_values(balance_data)

    # Define all expected balance sheet fields
    fields = [
        "total_assets",
        "current_assets",
        "non_current_assets",
        "total_liabilities",
        "current_liabilities",
        "non_current_liabilities",
        "shareholders_equity",
        "retained_earnings",
        "cash_and_equivalents",
        "total_equity_and_liabilities",
        "inventories",
        "prepaid_expenses",
    ]
    
    MISSING = {None, "", "N/A", "-", "--"}
    for field in fields:
        bs_item = getattr(report.balance_sheet, field, None)
        if bs_item is not None:
            for year in ["2024", "2023", "2022"]:
                value = balance_data.get(year, {}).get(field, "N/A")
                cur_val = getattr(bs_item, f"year_{year}", None)
                if (cur_val in MISSING) and (value not in MISSING):
                    setattr(bs_item, f"year_{year}", value)
        else:
            print(f"[WARN] report.balance_sheet missing attribute: {field}")

    report.balance_sheet.primary_currency = balance_data.get("currency", "N/A")
    report.balance_sheet.primary_multiplier = balance_data.get("multiplier", "N/A")
    
    print("✅ COMPLETED: S2.2 - Balance Sheet")
    
    print("\n" + "="*60)
    print("PROCESSING: S2.3 - Cash Flow Statement (with RAG)")
    print("="*60)
    
    # Use FAISS search for cash flow
    cashflow_data = extract_s2_3(md_file_2024, md_file_2023, top_k=12, model="gpt-4.1-mini")
    
    fields = [
        ("net_cash_from_operations", "net_cash_from_operations"),
        ("net_cash_from_investing", "net_cash_from_investing"),
        ("net_cash_from_financing", "net_cash_from_financing"),
        ("net_increase_decrease_cash", "net_increase_decrease_in_cash"),
        ("dividends", "dividends"),
    ]

    for field_name, key_name in fields:
        cf_item = getattr(report.cash_flow_statement, field_name, None)
        if cf_item is None:
            print(f"[WARN] Missing attribute in cash_flow_statement: {field_name}")
            continue
        for year in ["2024", "2023", "2022"]:
            value = cashflow_data.get(year, {}).get(key_name, "N/A")
            setattr(cf_item, f"year_{year}", value)

    report.cash_flow_statement.primary_currency = cashflow_data.get("currency", "N/A")
    report.cash_flow_statement.primary_multiplier = cashflow_data.get("multiplier", "N/A")

    print("✅ COMPLETED: S2.3 - Cash Flow Statement")
    
    print("=" * 60)
    print("PROCESSING: S2.4 - Key Financial Metrics")
    print("=" * 60)

    extract_s2_4(report)

    # save_partial_report(report, output_path="outputs/s2_partial.md")
    print("✅ COMPLETED: S2.4 - Key Financial Metrics")
        
    print("=" * 60)
    print("PROCESSING: S2.5 - Operating Performance")
    print("=" * 60)

    operating_perf = extract_s2_5(md_file_2024, md_file_2023, top_k=20, model="gpt-4.1-mini")

    report.operating_performance.revenue_by_product_service.year_2024 = operating_perf["2024"]["revenue_by_product_service"]
    report.operating_performance.revenue_by_product_service.year_2023 = operating_perf["2023"]["revenue_by_product_service"]
    report.operating_performance.revenue_by_product_service.year_2022 = operating_perf["2022"]["revenue_by_product_service"]
    
    report.operating_performance.revenue_by_geographic_region.year_2024 = operating_perf["2024"]["revenue_by_region"]
    report.operating_performance.revenue_by_geographic_region.year_2023 = operating_perf["2023"]["revenue_by_region"]
    report.operating_performance.revenue_by_geographic_region.year_2022 = operating_perf["2022"]["revenue_by_region"]
    
    print("✅ COMPLETED: S2.5 - Operating Performance")
    
    end_time = time.time()
    total_duration = end_time - start_time
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)
    
    print(f"\n   TOTAL EXECUTION TIME: {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    # =============== TESTER FOR SECTION 3 WITH SYNTHETIC DATA ==========================
    # --- quick helpers for mocks ---
    # def fd(y24, y23, y22, multiplier="Millions", currency="USD"):
    #     return FinancialData(year_2024=y24, year_2023=y23, year_2022=y22,
    #                         multiplier=multiplier, currency=currency)

    # def make_mock_report_for_s3():
    #     # Income Statement
    #     inc = IncomeStatement(
    #         revenue=fd(54000, 27000, 26800),
    #         cost_of_goods_sold=fd(14700, 11650, 11800),
    #         gross_profit=fd(39300, 15350, 15000),
    #         operating_expense=fd(11329, 11132, 11100),
    #         operating_income=fd(32972, 4224, 3900),
    #         net_profit=fd(29760, 4368, 3000),
    #         income_before_income_taxes=fd(33818, 4181, 3200),
    #         income_tax_expense=fd(4058, -187, 200),
    #         interest_expense=fd(10, 26, 25),
    #     )
    #     # annotate primary labels (your code reads these if present)
    #     inc.primary_currency = "USD"
    #     inc.primary_multiplier = "Millions"

    #     # Balance Sheet (only needed for S3.2 context; rough sample values)
    #     bal = BalanceSheet(
    #         total_assets=fd(65728, 41182, 38000),
    #         current_assets=fd(34000, 24000, 22000),
    #         non_current_assets=fd(31728, 17182, 16000),
    #         total_liabilities=fd(22750, 19081, 20300),
    #         current_liabilities=fd(8150, 6830, 6700),
    #         non_current_liabilities=fd(14600, 12251, 13600),
    #         shareholders_equity=fd(42978, 22101, 17700),
    #         retained_earnings=fd(12000, 9000, 7000),
    #         total_equity_and_liabilities=fd(65728, 41182, 38000),
    #         inventories=fd(2500, 2100, 2000),
    #         prepaid_expenses=fd(300, 280, 260),
    #     )
    #     bal.primary_currency = "USD"
    #     bal.primary_multiplier = "Millions"

    #     # Cash Flow (S3.2 context)
    #     cf = CashFlowStatement(
    #         net_cash_from_operations=fd(28090, 5641, 5200),
    #         net_cash_from_investing=fd(-12000, -4000, -3500),
    #         net_cash_from_financing=fd(-5000, -1200, -900),
    #         net_increase_decrease_cash=fd(11090, 441, 800),
    #         dividends=fd(395, 398, 380),
    #     )
    #     cf.primary_currency = "USD"
    #     cf.primary_multiplier = "Millions"

    #     # Key Metrics (S3.1 & S3.2)
    #     kfm = KeyFinancialMetrics(
    #         gross_margin=fd("72.72%", "56.92%", "56.0%"),
    #         operating_margin=fd("54.13%", "15.66%", "14.6%"),
    #         net_profit_margin=fd("48.85%", "16.19%", "11.2%"),
    #         current_ratio=fd("417.13%", "351.47%", "328.0%"),
    #         quick_ratio=fd("380.00%", "320.00%", "300.0%"),
    #         debt_to_equity=fd("52.94%", "86.31%", "114.7%"),
    #         interest_coverage=fd("12828.40%", "1612.21%", "1500.0%"),
    #         asset_turnover=fd("116.92%", "74.92%", "70.0%"),
    #         return_on_equity=fd("97.68%", "22.92%", "17.0%"),
    #         return_on_assets=fd("45.27%", "11.12%", "8.0%"),
    #         effective_tax_rate=fd("12.00%", "-4.47%", "6.3%"),
    #         dividend_payout_ratio=fd("1.33%", "9.11%", "12.0%"),
    #     )

    #     # Operating Performance (free-text buckets, put anything representative)
    #     perf = OperatingPerformance(
    #         revenue_by_product_service=fd(
    #             {"Compute & Networking": 47405, "Graphics": 6500, "Other": 95},
    #             {"Compute & Networking": 12000, "Graphics": 14000, "Other": 974},
    #             {"Compute & Networking": 11800, "Graphics": 14000, "Other": 1000},
    #         ),
    #         revenue_by_geographic_region=fd(
    #             {"US": 44, "China": 18, "EMEA": 20, "APAC ex-China": 18},
    #             {"US": 35, "China": 22, "EMEA": 22, "APAC ex-China": 21},
    #             {"US": 34, "China": 23, "EMEA": 22, "APAC ex-China": 21},
    #         ),
    #     )

    #     # Assemble the report (only S2 + S3 outputs needed)
    #     report = CompanyReport()
    #     report.income_statement = inc
    #     report.balance_sheet = bal
    #     report.cash_flow_statement = cf
    #     report.key_financial_metrics = kfm
    #     report.operating_performance = perf
    #     report.meta_output_lang = "en"  # or "zh-Hans"/"zh-Hant"
    #     return report

    # # --- example usage ---
    # # TARGET_LANGUAGE should be set somewhere in your codebase; for testing:
    # class Lang:
    #     EN = "en"
    #     ZH_SIM = "zh-Hans"
    #     ZH_TR = "zh-Hant"

    # TARGET_LANGUAGE = Lang.EN  # pick your flavor

    # report = make_mock_report_for_s3()


# ================== end tester =======================================
    
    
    print("\n" + "="*60)
    print("PROCESSING: S3.1 - Profitability Analysis")
    print("="*60)

    # Extract profitability analysis based on Section 2 data
    profitability_analysis = extract_s3_1(report, model="gpt-4.1-mini")

    # Save to report structure
    report.profitability_analysis.revenue_direct_cost_dynamics = profitability_analysis["revenue_direct_cost_dynamics"]
    report.profitability_analysis.operating_efficiency = profitability_analysis["operating_efficiency"] 
    report.profitability_analysis.external_oneoff_impact = profitability_analysis["external_oneoff_impact"]
    
    print("✅ COMPLETED: S3.1 - Profitability Analysis")
    print(f"   Revenue & Direct-Cost Analysis: {len(profitability_analysis['revenue_direct_cost_dynamics'])} chars")
    print(f"   Operating Efficiency Analysis: {len(profitability_analysis['operating_efficiency'])} chars")
    print(f"   External & One-Off Impact Analysis: {len(profitability_analysis['external_oneoff_impact'])} chars")

    # checkpoint("Section 3.1 - Profitability Analysis")
    
    print("\n" + "="*60)
    print("PROCESSING: S3.2 - Financial Performance Summary")
    print("="*60)

    # Extract financial performance summary based on Section 2 data
    financial_performance_summary = extract_s3_2(report, model="gpt-4.1-mini")

    # Save to report structure
    fps = report.financial_performance_summary
    fps.comprehensive_financial_health.report_2024 = financial_performance_summary["comprehensive_financial_health_2024"]
    fps.comprehensive_financial_health.report_2023 = financial_performance_summary["comprehensive_financial_health_2023"]
    fps.profitability_earnings_quality.report_2024 = financial_performance_summary["profitability_earnings_quality_2024"]
    fps.profitability_earnings_quality.report_2023 = financial_performance_summary["profitability_earnings_quality_2023"]
    fps.operational_efficiency.report_2024 = financial_performance_summary["operational_efficiency_2024"]
    fps.operational_efficiency.report_2023 = financial_performance_summary["operational_efficiency_2023"]
    fps.financial_risk_identification.report_2024 = financial_performance_summary["financial_risk_identification_2024"]
    fps.financial_risk_identification.report_2023 = financial_performance_summary["financial_risk_identification_2023"]
    fps.future_financial_performance_projection.report_2024 = financial_performance_summary["future_financial_performance_projection_2024"]
    fps.future_financial_performance_projection.report_2023 = financial_performance_summary["future_financial_performance_projection_2023"]

    print("✅ COMPLETED: S3.2 - Financial Performance Summary")
    print(f"   Comprehensive Financial Health: 2024: {len(financial_performance_summary['comprehensive_financial_health_2024'])} chars, 2023: {len(financial_performance_summary['comprehensive_financial_health_2023'])} chars")
    print(f"   Profitability Analysis: 2024: {len(financial_performance_summary['profitability_earnings_quality_2024'])} chars, 2023: {len(financial_performance_summary['profitability_earnings_quality_2023'])} chars")
    print(f"   Operational Efficiency: 2024: {len(financial_performance_summary['operational_efficiency_2024'])} chars, 2023: {len(financial_performance_summary['operational_efficiency_2023'])} chars")

    # checkpoint("Section 3.2 - Financial Performance Summary")
    
    print("\n" + "="*60)
    print("PROCESSING: S3.3 - Business Competitiveness")
    print("="*60)

    # Extract business competitiveness using RAG search
    business_competitiveness = extract_s3_3(md_file_2024, md_file_2023, top_k=15, model="gpt-4.1-mini")

    # Save to report structure
    comp = report.business_competitiveness
    comp.business_model_2024 = business_competitiveness["business_model_2024"]
    comp.business_model_2023 = business_competitiveness["business_model_2023"]
    comp.market_position_2024 = business_competitiveness["market_position_2024"]
    comp.market_position_2023 = business_competitiveness["market_position_2023"]

    print("✅ COMPLETED: S3.3 - Business Competitiveness")
    print(f"   Business Model 2024: {len(business_competitiveness['business_model_2024'])} chars")
    print(f"   Market Position 2024: {len(business_competitiveness['market_position_2024'])} chars")
    print(f"   Business Model 2023: {len(business_competitiveness['business_model_2023'])} chars")
    print(f"   Market Position 2023: {len(business_competitiveness['market_position_2023'])} chars")

    # checkpoint("Section 3.3 - Business Competitiveness")
        
    print("\n" + "="*60)
    print("PROCESSING: S4.1 - Risk Factors")
    print("="*60)

    # Extract risk factors using RAG search
    risk_factors = extract_s4_1(md_file_2024, md_file_2023, top_k=15, model="gpt-4.1-mini")
    
    rf = report.risk_factors  # You'll need to add this to CompanyReport
    rf.market_risks_2024 = risk_factors["market_risks_2024"]
    rf.market_risks_2023 = risk_factors["market_risks_2023"]
    rf.operational_risks_2024 = risk_factors["operational_risks_2024"]
    rf.operational_risks_2023 = risk_factors["operational_risks_2023"]
    rf.financial_risks_2024 = risk_factors["financial_risks_2024"]
    rf.financial_risks_2023 = risk_factors["financial_risks_2023"]
    rf.compliance_risks_2024 = risk_factors["compliance_risks_2024"]
    rf.compliance_risks_2023 = risk_factors["compliance_risks_2023"]

    print("✅ COMPLETED: S4.1 - Risk Factors")
    print(f"   Market Risks 2024: {len(risk_factors['market_risks_2024'])} chars")
    print(f"   Operational Risks 2024: {len(risk_factors['operational_risks_2024'])} chars")
    print(f"   Financial Risks 2024: {len(risk_factors['financial_risks_2024'])} chars")
    print(f"   Compliance Risks 2024: {len(risk_factors['compliance_risks_2024'])} chars")

    # checkpoint("Section 4.1 - Risk Factors")
    
    # Add this after S4.1 in your main extract function:

    print("\n" + "="*60)
    print("PROCESSING: S5.1 - Board Composition")
    print("="*60)

    # Extract board composition using RAG search (2024 only)
    board_composition = extract_s5_1(md_file_2024, top_k=15, model="gpt-4.1-mini")

    # Save to report structure (you'll need to add these fields to your CompanyReport dataclass)
    report.board_composition.members = board_composition["board_members"]

    print("✅ COMPLETED: S5.1 - Board Composition")
    print(f"   Found {len(board_composition['board_members'])} board members/executives")
    for i, member in enumerate(board_composition['board_members'][:3]):  # Show first 3
        print(f"   {i+1}. {member['name']} - {member['position']} - {member['total_income']}")

    # checkpoint("Section 5.1 - Board Composition")
            
    print("\n" + "="*60)
    print("PROCESSING: S5.2 - Internal Controls")
    print("="*60)

    # Extract internal controls using RAG search
    internal_controls = extract_s5_2(md_file_2024, md_file_2023, top_k=15, model="gpt-4.1-mini")

    # Save to report structure
    ic = report.internal_controls
    ic.risk_assessment_procedures.report_2024 = internal_controls["risk_assessment_procedures_2024"]
    ic.risk_assessment_procedures.report_2023 = internal_controls["risk_assessment_procedures_2023"]
    ic.control_activities.report_2024 = internal_controls["control_activities_2024"]
    ic.control_activities.report_2023 = internal_controls["control_activities_2023"]
    ic.monitoring_mechanisms.report_2024 = internal_controls["monitoring_mechanisms_2024"]
    ic.monitoring_mechanisms.report_2023 = internal_controls["monitoring_mechanisms_2023"]
    ic.identified_material_weaknesses.report_2024 = internal_controls["identified_material_weaknesses_2024"]
    ic.identified_material_weaknesses.report_2023 = internal_controls["identified_material_weaknesses_2023"]
    ic.effectiveness.report_2024 = internal_controls["effectiveness_2024"]
    ic.effectiveness.report_2023 = internal_controls["effectiveness_2023"]

    print("✅ COMPLETED: S5.2 - Internal Controls")
    print(f"   Risk Assessment 2024: {len(internal_controls['risk_assessment_procedures_2024'])} chars")
    print(f"   Control Activities 2024: {len(internal_controls['control_activities_2024'])} chars")
    print(f"   Monitoring Mechanisms 2024: {len(internal_controls['monitoring_mechanisms_2024'])} chars")
    print(f"   Effectiveness 2024: {len(internal_controls['effectiveness_2024'])} chars")

    # checkpoint("Section 5.2 - Internal Controls")
    
    print("\n" + "="*60)
    print("PROCESSING: S6.1 - Strategic Direction")
    print("="*60)

    # Extract strategic direction using RAG search
    strategic_direction = extract_s6_1(md_file_2024, md_file_2023, top_k=15, model="gpt-4.1-mini")

    # Save to report structure
    sd = report.strategic_direction
    sd.mergers_acquisition.report_2024 = strategic_direction["mergers_acquisition_2024"]
    sd.mergers_acquisition.report_2023 = strategic_direction["mergers_acquisition_2023"]
    sd.new_technologies.report_2024 = strategic_direction["new_technologies_2024"]
    sd.new_technologies.report_2023 = strategic_direction["new_technologies_2023"]
    sd.organisational_restructuring.report_2024 = strategic_direction["organisational_restructuring_2024"]
    sd.organisational_restructuring.report_2023 = strategic_direction["organisational_restructuring_2023"]

    print("✅ COMPLETED: S6.1 - Strategic Direction")
    print(f"   Mergers & Acquisition 2024: {len(strategic_direction['mergers_acquisition_2024'])} chars")
    print(f"   New Technologies 2024: {len(strategic_direction['new_technologies_2024'])} chars")
    print(f"   Organisational Restructuring 2024: {len(strategic_direction['organisational_restructuring_2024'])} chars")

    # checkpoint("Section 6.1 - Strategic Direction")

    print("\n" + "="*60)
    print("PROCESSING: S6.2 - Challenges and Uncertainties")
    print("="*60)

    # Extract challenges and uncertainties using RAG search
    challenges_uncertainties = extract_s6_2(md_file_2024, md_file_2023, top_k=15, model="gpt-4.1-mini")

    # Save to report structure
    cu = report.challenges_uncertainties
    cu.economic_challenges.report_2024 = challenges_uncertainties["economic_challenges_2024"]
    cu.economic_challenges.report_2023 = challenges_uncertainties["economic_challenges_2023"]
    cu.competitive_pressures.report_2024 = challenges_uncertainties["competitive_pressures_2024"]
    cu.competitive_pressures.report_2023 = challenges_uncertainties["competitive_pressures_2023"]

    print("✅ COMPLETED: S6.2 - Challenges and Uncertainties")
    print(f"   Economic Challenges 2024: {len(challenges_uncertainties['economic_challenges_2024'])} chars")
    print(f"   Competitive Pressures 2024: {len(challenges_uncertainties['competitive_pressures_2024'])} chars")
    print(f"   Economic Challenges 2023: {len(challenges_uncertainties['economic_challenges_2023'])} chars")
    print(f"   Competitive Pressures 2023: {len(challenges_uncertainties['competitive_pressures_2023'])} chars")

    # checkpoint("Section 6.2 - Challenges and Uncertainties")

    print("\n" + "="*60)
    print("PROCESSING: S6.3 - Innovation and Development Plans")
    print("="*60)

    # Extract innovation and development plans using RAG search
    innovation_development = extract_s6_3(md_file_2024, md_file_2023, top_k=15, model="gpt-4.1-mini")

    # Save to report structure
    id = report.innovation_development
    id.rd_investments.report_2024 = innovation_development["rd_investments_2024"]
    id.rd_investments.report_2023 = innovation_development["rd_investments_2023"]
    id.new_product_launches.report_2024 = innovation_development["new_product_launches_2024"]
    id.new_product_launches.report_2023 = innovation_development["new_product_launches_2023"]

    print("✅ COMPLETED: S6.3 - Innovation and Development Plans")
    print(f"   R&D Investments 2024: {len(innovation_development['rd_investments_2024'])} chars")
    print(f"   New Product Launches 2024: {len(innovation_development['new_product_launches_2024'])} chars")
    print(f"   R&D Investments 2023: {len(innovation_development['rd_investments_2023'])} chars")
    print(f"   New Product Launches 2023: {len(innovation_development['new_product_launches_2023'])} chars")

    # checkpoint("Section 6.3 - Innovation and Development Plans")

    return report
  
  
# if __name__ == "__main__":
#     import argparse
#     from pathlib import Path

#     parser = argparse.ArgumentParser(description="Generate DDR report using two annual reports (2024 and 2023).")

#     parser.add_argument("--md2024", required=True, help="Path to the 2024 markdown file (newest annual report)")

#     parser.add_argument("--md2023", required=True, help="Path to the 2023 markdown file (previous year's report)")
    
#     parser.add_argument("--currency", default="USD", help="Currency code for the report (default: USD)")

#     parser.add_argument("--output_language", required=True)
#     args = parser.parse_args()
    
#     out_lang = args.output_language
#     if out_lang == "ZH_SIM":
#         target_lang = Lang.ZH_SIM
#     elif out_lang == "ZH_TR":
#         target_lang = Lang.ZH_TR
#     elif out_lang == "EN" or out_lang == "IN":
#         target_lang = Lang.EN
#     else: 
#         raise ValueError(f"Unsupported output language: {args.output_language}. Supported languages are: ZH_SIM, ZH_TR, EN.")
    
#     # Run main extraction pipeline
#     report_info = extract(args.md2024, args.md2023, currency_code=args.currency, target_lang=target_lang)

#     # Ensure output directory exists
#     Path("artifacts").mkdir(parents=True, exist_ok=True)

#     # Create unique filename from command line argument and timestamp
#     md_path_2024 = Path(args.md2024)
#     md_file_2024 = md_path_2024.stem
#     first_term_filename = md_file_2024.split('_')[0] if '_' in md_file_2024 else md_file_2024
#     slug = _slugify(first_term_filename)
#     timestamp = time.strftime("%Y%m%d_%H%M%S")
    
#     # Create company-specific folder for main script output
#     #output_folder = f"artifacts/{slug}"
#     #output_file = f"{output_folder}/final/{timestamp}_finddr_report.md"
    
#     output_file = f"artifacts/{slug}.md"
    
#     # Ensure directory exists
#     Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
#     # generator = DDRGenerator(report_info, currency_code=args.currency)
#     # generator.save_report(output_file)
#     # generator.save_report(f"artifacts/finddr_report.md")
#     # print(f"\n✅ Saved generated DDR report to: {output_file}")

