from functools import lru_cache


KNOWLEDGE_BASE = {
    "bitcoin": "Bitcoin is a decentralized digital currency that is not controlled by a government or bank. Its value can be highly volatile.",
    "etf": "An ETF (Exchange-Traded Fund) is a collection of investments that can be bought and sold like a stock.",
    "roth_ira": "A Roth IRA is a retirement account funded with after-tax dollars. Qualified withdrawals in retirement are tax-free.",
    "traditional_ira": "A Traditional IRA is a retirement account that may provide tax benefits today but is generally taxed upon withdrawal.",
    "cd": "A Certificate of Deposit is a savings product offered by banks and credit unions that pays a fixed interest rate in exchange for leaving money deposited for a specified period.",
    "emergency_fund": "An emergency fund is money set aside for unexpected expenses such as job loss, medical expenses, or major repairs.",
    "diversification": "Diversification is the practice of spreading investments across multiple assets to reduce risk.",
    "compound_interest": "Compound interest is interest earned on both the original principal and previously earned interest.",
    "crypto": "Crypto is a broad category of digital assets. Different crypto assets can have very different designs and risks.",
    "stocks": "Stocks represent ownership shares in companies. Their value can rise or fall over time.",
    "budgeting": "Budgeting is the practice of giving your income a plan across essentials, goals, and flexible spending.",
    "discretionary_spending": "Discretionary spending is money used for flexible wants, such as travel or entertainment, after essentials and priority goals are covered.",
    "travel_savings": "Travel savings is money set aside gradually for trips so travel spending does not compete unexpectedly with bills, debt payments, or other priorities.",
    "college_savings": "College savings is money set aside over time for education costs. A useful plan considers the timeline, current savings, and a contribution that fits alongside other priorities.",
    "debt_payoff": "Debt payoff planning starts by understanding balances, interest rates, and minimum payments, then choosing a sustainable amount to pay down over time.",
    "retirement_budgeting": "Retirement budgeting means estimating future essential and flexible spending, then comparing that with expected income, savings, and debt obligations.",
}

TOPIC_KNOWLEDGE_ALIASES = {
    "travel_spending": "travel_savings",
    "college_costs": "college_savings",
    "debt_free_retirement": "retirement_budgeting",
}


@lru_cache(maxsize=None)
def retrieve_knowledge(topic: str | None) -> dict:
    normalized_topic = (topic or "").lower().replace(" ", "_")
    normalized_topic = TOPIC_KNOWLEDGE_ALIASES.get(normalized_topic, normalized_topic)
    content = KNOWLEDGE_BASE.get(normalized_topic)
    if not content:
        return {}
    return {"topic": normalized_topic, "content": content, "source": "local_demo_knowledge_base"}
