import re


def evaluate_guardrails(user_message: str) -> dict:
    lower_message = user_message.lower()
    personalized_advice = bool(
        re.search(r"\bshould i\b", lower_message)
        or re.search(r"\bfor me\b", lower_message)
        or any(phrase in lower_message for phrase in ["all my savings", "my entire savings", "tell me what to buy"])
    )
    tax_advice = any(term in lower_message for term in ["tax", "deduct", "deduction", "irs", "taxable"])
    legal_advice = any(term in lower_message for term in ["legal", "law", "lawsuit", "attorney", "lawyer"])

    categories = []
    if personalized_advice:
        categories.append("personalized_investment_advice")
    if tax_advice:
        categories.append("tax_advice")
    if legal_advice:
        categories.append("legal_advice")

    return {
        "mode": "educational_only" if categories else "standard",
        "categories": categories,
    }


def guardrail_message(guardrail_result: dict) -> str:
    if guardrail_result["mode"] == "standard":
        return ""
    return (
        "I can explain the risks and tradeoffs in general terms, but I cannot tell you what is right for your personal "
        "situation. For guidance tailored to your circumstances, consider consulting an appropriate qualified professional. "
    )
