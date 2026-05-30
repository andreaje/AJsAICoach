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
}


def retrieve_knowledge(topic: str | None) -> dict:
    normalized_topic = (topic or "").lower().replace(" ", "_")
    content = KNOWLEDGE_BASE.get(normalized_topic)
    if not content:
        return {}
    return {"topic": normalized_topic, "content": content, "source": "local_demo_knowledge_base"}
