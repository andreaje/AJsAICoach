MARKET_PRICES = {
    "bitcoin": 108250,
    "voo": 585,
    "spy": 640,
    "aapl": 215,
}

INTEREST_RATES = {
    "cd": "4.2%",
    "high_yield_savings": "3.8%",
}


def get_market_price(asset: str):
    return MARKET_PRICES.get(asset.lower())


def get_interest_rate(product: str):
    return INTEREST_RATES.get(product.lower().replace(" ", "_"))


def collect_tool_results(user_message: str, primary_topic: str) -> dict:
    lower_message = user_message.lower()
    results = {}

    if any(term in lower_message for term in ["price", "cost", "worth", "trading at"]):
        price = get_market_price(primary_topic)
        if price is not None:
            results["market_price"] = {"asset": primary_topic, "value": price, "mocked": True}

    if any(term in lower_message for term in ["rate", "interest", "yield", "apy"]):
        rate = get_interest_rate(primary_topic)
        if rate is not None:
            results["interest_rate"] = {"product": primary_topic, "value": rate, "mocked": True}

    return results
