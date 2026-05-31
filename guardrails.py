GUARDRAIL_CATEGORIES = {
    "personalized_financial_advice_boundary",
    "out_of_domain_request",
}


def evaluate_guardrails(user_message: str, llm_categories: list[str] | None = None) -> dict:
    del user_message  # Guardrail classification comes from the LLM understanding layer.
    categories = list(
        dict.fromkeys(
            category
            for category in (llm_categories or [])
            if category in GUARDRAIL_CATEGORIES
        )
    )
    guardrail_triggered = next(
        (
            category
            for category in ["out_of_domain_request", "personalized_financial_advice_boundary"]
            if category in categories
        ),
        None,
    )
    return {
        "mode": "educational_only" if categories else "standard",
        "categories": categories,
        "guardrail_triggered": guardrail_triggered,
    }


def guardrail_message(guardrail_result: dict) -> str:
    categories = guardrail_result.get("categories", [])
    if "out_of_domain_request" in categories:
        return (
            "I am primarily here to help with financial questions and planning. For questions outside financial "
            "coaching, consider consulting an appropriate qualified professional. If there is a related financial "
            "concern, I can help you think through that."
        )
    if "personalized_financial_advice_boundary" in categories:
        return (
            "I can explain the relevant considerations and tradeoffs in general terms, but I cannot recommend a "
            "specific choice, percentage, or range for your personal situation. For guidance tailored to your "
            "circumstances, consider consulting an appropriate qualified professional."
        )
    if guardrail_result.get("mode") != "standard":
        return (
            "I can explain the risks and tradeoffs in general terms, but I cannot tell you what is right for your "
            "personal situation. For guidance tailored to your circumstances, consider consulting an appropriate "
            "qualified professional."
        )
    return ""
