from report_generator import CompanyReport, DDRGenerator


def run_test():
    report = CompanyReport()
    # Set multiplier and currency from your provided merge info
    multiplier = "Millions"
    currency = "GBP"

    # Populate income statement fields with provided values
    inc = report.income_statement
    inc.revenue.year_2024 = 510.4
    inc.revenue.year_2023 = 473.0
    inc.revenue.year_2022 = 442.8
    inc.revenue.multiplier = multiplier
    inc.revenue.currency = currency

    inc.cost_of_goods_sold.year_2024 = "N/A"
    inc.cost_of_goods_sold.year_2023 = "N/A"
    inc.cost_of_goods_sold.year_2022 = "N/A"
    inc.cost_of_goods_sold.multiplier = multiplier
    inc.cost_of_goods_sold.currency = currency

    inc.gross_profit.year_2024 = "N/A"
    inc.gross_profit.year_2023 = "N/A"
    inc.gross_profit.year_2022 = "N/A"
    inc.gross_profit.multiplier = multiplier
    inc.gross_profit.currency = currency

    inc.operating_expense.year_2024 = 452.3
    inc.operating_expense.year_2023 = 427.2
    inc.operating_expense.year_2022 = 451.6
    inc.operating_expense.multiplier = multiplier
    inc.operating_expense.currency = currency

    inc.operating_income.year_2024 = 58.1
    inc.operating_income.year_2023 = 45.4
    inc.operating_income.year_2022 = 49.4
    inc.operating_income.multiplier = multiplier
    inc.operating_income.currency = currency

    inc.net_profit.year_2024 = 39.5
    inc.net_profit.year_2023 = 5.4
    inc.net_profit.year_2022 = 47.4
    inc.net_profit.multiplier = multiplier
    inc.net_profit.currency = currency

    inc.income_before_taxes.year_2024 = 53.3
    inc.income_before_taxes.year_2023 = 44.1
    inc.income_before_taxes.year_2022 = 47.9
    inc.income_before_taxes.multiplier = multiplier
    inc.income_before_taxes.currency = currency

    inc.income_tax_expense.year_2024 = 10.6
    inc.income_tax_expense.year_2023 = 6.4
    inc.income_tax_expense.year_2022 = 3.5
    inc.income_tax_expense.multiplier = multiplier
    inc.income_tax_expense.currency = currency

    inc.interest_expense.year_2024 = 4.8
    inc.interest_expense.year_2023 = 1.3
    inc.interest_expense.year_2022 = 1.5
    inc.interest_expense.multiplier = multiplier
    inc.interest_expense.currency = currency

    # Make a small change to basic info so the report includes a header
    report.basic_info.company_name = "Chemming plc"
    report.basic_info.establishment_date = "N/A"
    report.basic_info.headquarters_location = "N/A"

    generator = DDRGenerator(report)
    out = "artifacts/test_income_statement.md"
    generator.save_report(out)
    print(f"Test income statement written to: {out}")


if __name__ == '__main__':
    run_test()