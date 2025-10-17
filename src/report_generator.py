""" 
Financial Report Generator 
This module generates comprehensive financial reports in markdown format following a structured template 
with sections for company overview, financial performance, business analysis, risk factors, corporate governance, 
and future outlook. 
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class BasicInfo:
    """S1.1: Basic Information"""
    company_name: str = "N/A"
    establishment_date: str = "N/A"
    headquarters_location: str = "N/A"


@dataclass
class CoreCompetency:
    """Single competency with 2024 and 2023 values"""
    report_2024: str = "N/A"
    report_2023: str = "N/A"


@dataclass
class CoreCompetencies:
    """S1.2: Core Competencies"""
    innovation_advantages: CoreCompetency = field(default_factory=CoreCompetency)
    product_advantages: CoreCompetency = field(default_factory=CoreCompetency)
    brand_recognition: CoreCompetency = field(default_factory=CoreCompetency)
    reputation_ratings: CoreCompetency = field(default_factory=CoreCompetency)


@dataclass
class MissionVision:
    """S1.3: Mission & Vision"""
    mission_statement: str = "N/A"
    vision_statement: str = "N/A"
    core_values: str = "N/A"


@dataclass
class FinancialData:
    """Base class for financial data with multi-year values"""
    year_2024: Any = "N/A"
    year_2023: Any = "N/A"
    year_2022: Any = "N/A"
    multiplier: str = "Units"
    currency: str = "USD"


@dataclass
class IncomeStatement:
    """S2.1: Income Statement"""
    revenue: FinancialData = field(default_factory=FinancialData)
    cost_of_goods_sold: FinancialData = field(default_factory=FinancialData)
    gross_profit: FinancialData = field(default_factory=FinancialData)
    operating_expense: FinancialData = field(default_factory=FinancialData)
    operating_income: FinancialData = field(default_factory=FinancialData)
    net_profit: FinancialData = field(default_factory=FinancialData)
    income_before_income_taxes: FinancialData = field(default_factory=FinancialData)
    income_tax_expense: FinancialData = field(default_factory=FinancialData)
    interest_expense: FinancialData = field(default_factory=FinancialData)


@dataclass
class BalanceSheet:
    """S2.2: Balance Sheet"""
    total_assets: FinancialData = field(default_factory=FinancialData)
    current_assets: FinancialData = field(default_factory=FinancialData)
    non_current_assets: FinancialData = field(default_factory=FinancialData)
    total_liabilities: FinancialData = field(default_factory=FinancialData)
    current_liabilities: FinancialData = field(default_factory=FinancialData)
    non_current_liabilities: FinancialData = field(default_factory=FinancialData)
    shareholders_equity: FinancialData = field(default_factory=FinancialData)
    retained_earnings: FinancialData = field(default_factory=FinancialData)
    total_equity_and_liabilities: FinancialData = field(default_factory=FinancialData)
    inventories: FinancialData = field(default_factory=FinancialData)
    prepaid_expenses: FinancialData = field(default_factory=FinancialData)


@dataclass
class CashFlowStatement:
    """S2.3: Cash Flow Statement"""
    net_cash_from_operations: FinancialData = field(default_factory=FinancialData)
    net_cash_from_investing: FinancialData = field(default_factory=FinancialData)
    net_cash_from_financing: FinancialData = field(default_factory=FinancialData)
    net_increase_decrease_cash: FinancialData = field(default_factory=FinancialData)
    dividends: FinancialData = field(default_factory=FinancialData)


@dataclass
class KeyFinancialMetrics:
    """S2.4: Key Financial Metrics (percentages)"""
    gross_margin: FinancialData = field(default_factory=FinancialData)
    operating_margin: FinancialData = field(default_factory=FinancialData)
    net_profit_margin: FinancialData = field(default_factory=FinancialData)
    current_ratio: FinancialData = field(default_factory=FinancialData)
    quick_ratio: FinancialData = field(default_factory=FinancialData)
    debt_to_equity: FinancialData = field(default_factory=FinancialData)
    interest_coverage: FinancialData = field(default_factory=FinancialData)
    asset_turnover: FinancialData = field(default_factory=FinancialData)
    return_on_equity: FinancialData = field(default_factory=FinancialData)
    return_on_assets: FinancialData = field(default_factory=FinancialData)
    effective_tax_rate: FinancialData = field(default_factory=FinancialData)
    dividend_payout_ratio: FinancialData = field(default_factory=FinancialData)


@dataclass
class OperatingPerformance:
    """S2.5: Operating Performance"""
    revenue_by_product_service: FinancialData = field(default_factory=FinancialData)
    revenue_by_geographic_region: FinancialData = field(default_factory=FinancialData)


@dataclass
class ProfitabilityAnalysis:
    """S3.1: Profitability Analysis"""
    revenue_direct_cost_dynamics: str = "N/A"
    operating_efficiency: str = "N/A"
    external_oneoff_impact: str = "N/A"


@dataclass
class FinancialPerformanceSummary:
    """S3.2: Financial Performance Summary"""
    comprehensive_financial_health: CoreCompetency = field(default_factory=CoreCompetency)
    profitability_earnings_quality: CoreCompetency = field(default_factory=CoreCompetency)
    operational_efficiency: CoreCompetency = field(default_factory=CoreCompetency)
    financial_risk_identification: CoreCompetency = field(default_factory=CoreCompetency)
    future_financial_performance_projection: CoreCompetency = field(default_factory=CoreCompetency)


@dataclass
class BusinessCompetitiveness:
    """S3.3: Business Competitiveness"""
    business_model_2024: str = "N/A"
    business_model_2023: str = "N/A"
    market_position_2024: str = "N/A"
    market_position_2023: str = "N/A"


@dataclass
class RiskFactors:
    """S4.1: Risk Factors"""
    market_risks: CoreCompetency = field(default_factory=CoreCompetency)
    operational_risks: CoreCompetency = field(default_factory=CoreCompetency)
    financial_risks: CoreCompetency = field(default_factory=CoreCompetency)
    compliance_risks: CoreCompetency = field(default_factory=CoreCompetency)


@dataclass
class BoardMember:
    """Board member information"""
    name: str = "N/A"
    position: str = "N/A"
    total_income: str = "N/A"


@dataclass
class BoardComposition:
    """S5.1: Board Composition"""
    members: List[BoardMember] = field(default_factory=list)


@dataclass
class InternalControls:
    """S5.2: Internal Controls"""
    risk_assessment_procedures: CoreCompetency = field(default_factory=CoreCompetency)
    control_activities: CoreCompetency = field(default_factory=CoreCompetency)
    monitoring_mechanisms: CoreCompetency = field(default_factory=CoreCompetency)
    identified_material_weaknesses: CoreCompetency = field(default_factory=CoreCompetency)
    effectiveness: CoreCompetency = field(default_factory=CoreCompetency)


@dataclass
class StrategicDirection:
    """S6.1: Strategic Direction"""
    mergers_acquisition: CoreCompetency = field(default_factory=CoreCompetency)
    new_technologies: CoreCompetency = field(default_factory=CoreCompetency)
    organisational_restructuring: CoreCompetency = field(default_factory=CoreCompetency)


@dataclass
class ChallengesUncertainties:
    """S6.2: Challenges and Uncertainties"""
    economic_challenges: CoreCompetency = field(default_factory=CoreCompetency)
    competitive_pressures: CoreCompetency = field(default_factory=CoreCompetency)


@dataclass
class InnovationDevelopment:
    """S6.3: Innovation and Development Plans"""
    rd_investments: CoreCompetency = field(default_factory=CoreCompetency)
    new_product_launches: CoreCompetency = field(default_factory=CoreCompetency)


@dataclass
class CompanyReport:
    """Complete company financial report structure"""
    # Section 1: Company Overview
    basic_info: BasicInfo = field(default_factory=BasicInfo)
    core_competencies: CoreCompetencies = field(default_factory=CoreCompetencies)
    mission_vision: MissionVision = field(default_factory=MissionVision)

    # Section 2: Financial Performance
    income_statement: IncomeStatement = field(default_factory=IncomeStatement)
    balance_sheet: BalanceSheet = field(default_factory=BalanceSheet)
    cash_flow_statement: CashFlowStatement = field(default_factory=CashFlowStatement)
    key_financial_metrics: KeyFinancialMetrics = field(default_factory=KeyFinancialMetrics)
    operating_performance: OperatingPerformance = field(default_factory=OperatingPerformance)

    # Section 3: Business Analysis
    profitability_analysis: ProfitabilityAnalysis = field(default_factory=ProfitabilityAnalysis)
    financial_performance_summary: FinancialPerformanceSummary = field(default_factory=FinancialPerformanceSummary)
    business_competitiveness: BusinessCompetitiveness = field(default_factory=BusinessCompetitiveness)

    # Section 4: Risk Factors
    risk_factors: RiskFactors = field(default_factory=RiskFactors)

    # Section 5: Corporate Governance
    board_composition: BoardComposition = field(default_factory=BoardComposition)
    internal_controls: InternalControls = field(default_factory=InternalControls)

    # Section 6: Future Outlook
    strategic_direction: StrategicDirection = field(default_factory=StrategicDirection)
    challenges_uncertainties: ChallengesUncertainties = field(default_factory=ChallengesUncertainties)
    innovation_development: InnovationDevelopment = field(default_factory=InnovationDevelopment)


class DDRGenerator:
    """Generate markdown reports from CompanyReport data"""

    def __init__(self, report: CompanyReport):
        self.report = report

    def format_financial_value(self, value: Any) -> str:
        """Format financial values consistently"""
        if value == "N/A" or value is None:
            return "N/A"

        # Handle negative values in parentheses
        if isinstance(value, str) and value.startswith("(") and value.endswith(")"):
            return f"({value[1:-1]})"

        # Handle numeric values
        if isinstance(value, (int, float)):
            if value < 0:
                return f"({abs(value):,})"
            return f"{value:,}"

        return str(value)

        # ------------------ Section 1 ------------------
    def generate_section_1(self) -> str:
        """Generate Section 1: Company Overview"""
        return f"""# Section 1: Company Overview
          ## S1.1: Basic Information
          | Field | Value |
          | :---- | :---- |
          | Company Name | {self.report.basic_info.company_name} |
          | Establishment Date | {self.report.basic_info.establishment_date} |
          | Headquarters Location | {self.report.basic_info.headquarters_location} |

          ## S1.2 : Core Competencies
          | Perspective | 2024 Report | 2023 Report |
          | :---- | :---- | :---- |
          | Innovation Advantages | {self.report.core_competencies.innovation_advantages.report_2024} | {self.report.core_competencies.innovation_advantages.report_2023} |
          | Product Advantages | {self.report.core_competencies.product_advantages.report_2024} | {self.report.core_competencies.product_advantages.report_2023} |
          | Brand Recognition | {self.report.core_competencies.brand_recognition.report_2024} | {self.report.core_competencies.brand_recognition.report_2023} |
          | Reputation Ratings | {self.report.core_competencies.reputation_ratings.report_2024} | {self.report.core_competencies.reputation_ratings.report_2023} |

          ## S1.3 : Mission & Vision
          | Field | Value |
          | :---- | :---- |
          | Mission Statement | {self.report.mission_vision.mission_statement} |
          | Vision Statement | {self.report.mission_vision.vision_statement} |
          | Core Values | {self.report.mission_vision.core_values} |
          """

    # ------------------ Section 2 ------------------
    def generate_section_2(self) -> str:
        """Generate Section 2: Financial Performance"""
        inc = self.report.income_statement
        bal = self.report.balance_sheet
        cf = self.report.cash_flow_statement
        metrics = self.report.key_financial_metrics
        perf = self.report.operating_performance

        return f"""# Section 2: Financial Performance

          ## S2.1: Income Statement
          | Field | 2024 | 2023 | 2022 | Multiplier | Currency |
          | :---- | :---- | :---- | :---- | :---- | :---- |
          | Revenue | {self.format_financial_value(inc.revenue.year_2024)} | {self.format_financial_value(inc.revenue.year_2023)} | {self.format_financial_value(inc.revenue.year_2022)} | {inc.revenue.multiplier} | {inc.revenue.currency} |
          | Cost of Goods Sold | {self.format_financial_value(inc.cost_of_goods_sold.year_2024)} | {self.format_financial_value(inc.cost_of_goods_sold.year_2023)} | {self.format_financial_value(inc.cost_of_goods_sold.year_2022)} | {inc.cost_of_goods_sold.multiplier} | {inc.cost_of_goods_sold.currency} |
          | Gross Profit | {self.format_financial_value(inc.gross_profit.year_2024)} | {self.format_financial_value(inc.gross_profit.year_2023)} | {self.format_financial_value(inc.gross_profit.year_2022)} | {inc.gross_profit.multiplier} | {inc.gross_profit.currency} |
          | Operating Expense | {self.format_financial_value(inc.operating_expense.year_2024)} | {self.format_financial_value(inc.operating_expense.year_2023)} | {self.format_financial_value(inc.operating_expense.year_2022)} | {inc.operating_expense.multiplier} | {inc.operating_expense.currency} |
          | Operating Income | {self.format_financial_value(inc.operating_income.year_2024)} | {self.format_financial_value(inc.operating_income.year_2023)} | {self.format_financial_value(inc.operating_income.year_2022)} | {inc.operating_income.multiplier} | {inc.operating_income.currency} |
          | Net Profit | {self.format_financial_value(inc.net_profit.year_2024)} | {self.format_financial_value(inc.net_profit.year_2023)} | {self.format_financial_value(inc.net_profit.year_2022)} | {inc.net_profit.multiplier} | {inc.net_profit.currency} |
          | Income before income taxes | {self.format_financial_value(inc.income_before_income_taxes.year_2024)} | {self.format_financial_value(inc.income_before_income_taxes.year_2023)} | {self.format_financial_value(inc.income_before_income_taxes.year_2022)} | {inc.income_before_income_taxes.multiplier} | {inc.income_before_income_taxes.currency} |
          | Income tax expense(benefit) | {self.format_financial_value(inc.income_tax_expense.year_2024)} | {self.format_financial_value(inc.income_tax_expense.year_2023)} | {self.format_financial_value(inc.income_tax_expense.year_2022)} | {inc.income_tax_expense.multiplier} | {inc.income_tax_expense.currency} |
          | Interest Expense | {self.format_financial_value(inc.interest_expense.year_2024)} | {self.format_financial_value(inc.interest_expense.year_2023)} | {self.format_financial_value(inc.interest_expense.year_2022)} | {inc.interest_expense.multiplier} | {inc.interest_expense.currency} |

          ## S2.2: Balance Sheet
          | Field | 2024 | 2023 | 2022 | Multiplier | Currency |
          | :---- | :---- | :---- | :---- | :---- | :---- |
          | Total Assets | {self.format_financial_value(bal.total_assets.year_2024)} | {self.format_financial_value(bal.total_assets.year_2023)} | {self.format_financial_value(bal.total_assets.year_2022)} | {bal.total_assets.multiplier} | {bal.total_assets.currency} |
          | Current Assets | {self.format_financial_value(bal.current_assets.year_2024)} | {self.format_financial_value(bal.current_assets.year_2023)} | {self.format_financial_value(bal.current_assets.year_2022)} | {bal.current_assets.multiplier} | {bal.current_assets.currency} |
          | Non-Current Assets | {self.format_financial_value(bal.non_current_assets.year_2024)} | {self.format_financial_value(bal.non_current_assets.year_2023)} | {self.format_financial_value(bal.non_current_assets.year_2022)} | {bal.non_current_assets.multiplier} | {bal.non_current_assets.currency} |
          | Total Liabilities | {self.format_financial_value(bal.total_liabilities.year_2024)} | {self.format_financial_value(bal.total_liabilities.year_2023)} | {self.format_financial_value(bal.total_liabilities.year_2022)} | {bal.total_liabilities.multiplier} | {bal.total_liabilities.currency} |
          | Current Liabilities | {self.format_financial_value(bal.current_liabilities.year_2024)} | {self.format_financial_value(bal.current_liabilities.year_2023)} | {self.format_financial_value(bal.current_liabilities.year_2022)} | {bal.current_liabilities.multiplier} | {bal.current_liabilities.currency} |
          | Non-Current Liabilities | {self.format_financial_value(bal.non_current_liabilities.year_2024)} | {self.format_financial_value(bal.non_current_liabilities.year_2023)} | {self.format_financial_value(bal.non_current_liabilities.year_2022)} | {bal.non_current_liabilities.multiplier} | {bal.non_current_liabilities.currency} |
          | Shareholders' Equity | {self.format_financial_value(bal.shareholders_equity.year_2024)} | {self.format_financial_value(bal.shareholders_equity.year_2023)} | {self.format_financial_value(bal.shareholders_equity.year_2022)} | {bal.shareholders_equity.multiplier} | {bal.shareholders_equity.currency} |
          | Retained Earnings | {self.format_financial_value(bal.retained_earnings.year_2024)} | {self.format_financial_value(bal.retained_earnings.year_2023)} | {self.format_financial_value(bal.retained_earnings.year_2022)} | {bal.retained_earnings.multiplier} | {bal.retained_earnings.currency} |
          | Total Equity and Liabilities | {self.format_financial_value(bal.total_equity_and_liabilities.year_2024)} | {self.format_financial_value(bal.total_equity_and_liabilities.year_2023)} | {self.format_financial_value(bal.total_equity_and_liabilities.year_2022)} | {bal.total_equity_and_liabilities.multiplier} | {bal.total_equity_and_liabilities.currency} |
          | Inventories | {self.format_financial_value(bal.inventories.year_2024)} | {self.format_financial_value(bal.inventories.year_2023)} | {self.format_financial_value(bal.inventories.year_2022)} | {bal.inventories.multiplier} | {bal.inventories.currency} |
          | Prepaid Expenses | {self.format_financial_value(bal.prepaid_expenses.year_2024)} | {self.format_financial_value(bal.prepaid_expenses.year_2023)} | {self.format_financial_value(bal.prepaid_expenses.year_2022)} | {bal.prepaid_expenses.multiplier} | {bal.prepaid_expenses.currency} |
          """

    # ------------------ Section 3 ------------------
    def generate_section_3(self) -> str:
        """Generate Section 3: Business Analysis"""
        prof = self.report.profitability_analysis
        fps = self.report.financial_performance_summary
        comp = self.report.business_competitiveness

        return f"""# Section 3: Business Analysis
        ## S3.1: Profitability Analysis
        | Field | Answer |
        | :---- | :---- |
        | Revenue & Direct-Cost Dynamics | {prof.revenue_direct_cost_dynamics} |
        | Operating Efficiency | {prof.operating_efficiency} |
        | External & One-Off Impact | {prof.external_oneoff_impact} |

        ## S3.2: Financial Performance Summary
        | Perspective | 2024 Report | 2023 Report |
        | :---- | :---- | :---- |
        | Comprehensive financial health | {fps.comprehensive_financial_health.report_2024} | {fps.comprehensive_financial_health.report_2023} |
        | Profitability and earnings quality | {fps.profitability_earnings_quality.report_2024} | {fps.profitability_earnings_quality.report_2023} |
        | Operational efficiency | {fps.operational_efficiency.report_2024} | {fps.operational_efficiency.report_2023} |
        | Financial risk identification and early warning | {fps.financial_risk_identification.report_2024} | {fps.financial_risk_identification.report_2023} |
        | Future financial performance projection | {fps.future_financial_performance_projection.report_2024} | {fps.future_financial_performance_projection.report_2023} |

        ## S3.3: Business Competitiveness
        | Field | 2024 Report | 2023 Report |
        | :---- | :---- | :---- |
        | Business Model | {comp.business_model_2024} | {comp.business_model_2023} |
        | Market Position | {comp.market_position_2024} | {comp.market_position_2023} |
        """

            # ------------------ Section 4 ------------------
    def generate_section_4(self) -> str:
        """Generate Section 4: Risk Factors"""
        rf = self.report.risk_factors
        return f"""# Section 4: Risk Factors
          ## S4.1: Risk Factors
          | Perspective | 2024 Report | 2023 Report |
          | :---- | :---- | :---- |
          | Market Risks | {rf.market_risks.report_2024} | {rf.market_risks.report_2023} |
          | Operational Risks | {rf.operational_risks.report_2024} | {rf.operational_risks.report_2023} |
          | Financial Risks | {rf.financial_risks.report_2024} | {rf.financial_risks.report_2023} |
          | Compliance Risks | {rf.compliance_risks.report_2024} | {rf.compliance_risks.report_2023} |
          """

    # ------------------ Section 5 ------------------
    def generate_section_5(self) -> str:
        """Generate Section 5: Corporate Governance"""
        board = self.report.board_composition
        controls = self.report.internal_controls
        board_table = ""

        if board.members:
            for member in board.members:
                board_table += f"| {member.name} | {member.position} | {member.total_income} |\n"
        else:
            board_table = "| N/A | N/A | N/A |\n"

        return f"""# Section 5: Corporate Governance
        ## S5.1: Board Composition
        | Name | Position | Total Income |
        | :---- | :---- | :---- |
        {board_table}
        ## S5.2: Internal Controls
        | Perspective | 2024 Report | 2023 Report |
        | :---- | :---- | :---- |
        | Risk assessment procedures | {controls.risk_assessment_procedures.report_2024} | {controls.risk_assessment_procedures.report_2023} |
        | Control activities | {controls.control_activities.report_2024} | {controls.control_activities.report_2023} |
        | Monitoring mechanisms | {controls.monitoring_mechanisms.report_2024} | {controls.monitoring_mechanisms.report_2023} |
        | Identified material weaknesses or deficiencies | {controls.identified_material_weaknesses.report_2024} | {controls.identified_material_weaknesses.report_2023} |
        | Effectiveness | {controls.effectiveness.report_2024} | {controls.effectiveness.report_2023} |
        """

        # ------------------ Section 6 ------------------
    def generate_section_6(self) -> str:
        """Generate Section 6: Future Outlook"""
        strat = self.report.strategic_direction
        challenges = self.report.challenges_uncertainties
        innovation = self.report.innovation_development

        return f"""# Section 6: Future Outlook
        ## S6.1: Strategic Direction
        | Perspective | 2024 Report | 2023 Report |
        | :---- | :---- | :---- |
        | Mergers and Acquisition | {strat.mergers_acquisition.report_2024} | {strat.mergers_acquisition.report_2023} |
        | New technologies | {strat.new_technologies.report_2024} | {strat.new_technologies.report_2023} |
        | Organisational Restructuring | {strat.organisational_restructuring.report_2024} | {strat.organisational_restructuring.report_2023} |

        ## S6.2: Challenges and Uncertainties
        | Perspective | 2024 Report | 2023 Report |
        | :---- | :---- | :---- |
        | Economic challenges such as inflation, recession risks, and shifting consumer behavior that could impact revenue and profitability. | {challenges.economic_challenges.report_2024} | {challenges.economic_challenges.report_2023} |
        | Competitive pressures from both established industry players and new, disruptive market entrants that the company faces. | {challenges.competitive_pressures.report_2024} | {challenges.competitive_pressures.report_2023} |

        ## S6.3: Innovation and Development Plans
        | Perspective | 2024 Report | 2023 Report |
        | :---- | :---- | :---- |
        | R&D investments, with a focus on advancing technology, improving products, and creating new solutions to cater to market trends | {innovation.rd_investments.report_2024} | {innovation.rd_investments.report_2023} |
        | New product launches, emphasizing the company's commitment to continuously introducing differentiated products | {innovation.new_product_launches.report_2024} | {innovation.new_product_launches.report_2023} |
        """

    # ------------------ Combine and Save ------------------
    def generate_full_report(self) -> str:
        """Generate complete markdown report"""
        sections = [
            self.generate_section_1(),
            self.generate_section_2(),
            self.generate_section_3(),
            self.generate_section_4(),
            self.generate_section_5(),
            self.generate_section_6(),
        ]
        return "\n".join(sections)

    def save_report(self, output_path: str):
        """Save the complete report to a markdown file"""
        report_content = self.generate_full_report()

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"Report saved to: {output_path}")

# ------------------ Main Entry ------------------
if __name__ == "__main__":
    sample_report = DDRGenerator.create_sample_report()
    generator = DDRGenerator(sample_report)
    output_file = "artifacts/sample_financial_report.md"
    generator.save_report(output_file)
    print("Sample report generated successfully!")
    print(f"Check the file: {output_file}")