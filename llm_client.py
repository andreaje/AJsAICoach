import re

from guardrails import evaluate_guardrails, guardrail_message


REFLECTIONS = {
    "emotional_concern": "I hear that this feels uncertain, and it makes sense to slow down and make the risk easier to understand.",
    "decision_support": "You are weighing a financial decision and want to understand the tradeoffs before acting.",
    "product_exploration": "You are exploring a financial product and want a clearer sense of how it fits into the bigger picture.",
    "learn_before_investing": "You are interested in investing, but you want to understand the product before putting money into it.",
    "goal_discovery": "You are starting with the goal, which is a useful place to begin.",
}


def format_tool_results(tool_results: dict) -> str:
    parts = []
    if "market_price" in tool_results:
        result = tool_results["market_price"]
        parts.append(f"For this demo, the mocked market price for {result['asset']} is ${result['value']:,}.")
    if "interest_rate" in tool_results:
        result = tool_results["interest_rate"]
        parts.append(f"For this demo, the mocked interest rate for {result['product']} is {result['value']}.")
    return " ".join(parts)


def address_question_or_concern(user_message: str, understanding: dict) -> str:
    concern = understanding.get("concern_type")
    goal = understanding.get("current_goal")
    parent_topic = understanding.get("parent_topic")
    tokens = set(re.findall(r"[a-z0-9']+", user_message.lower()))

    if understanding.get("primary_topic") == "bitcoin" and {"safe", "risky", "risk"} & tokens:
        return (
            "Bitcoin is not guaranteed to be safe: its price can move sharply, and it can lose value. It is worth "
            "understanding that volatility and protecting money you may need soon before considering any investment."
        )
    if concern == "starting too late":
        return (
            "Starting later than you hoped does not mean it is too late. A useful first step is to look at your "
            "timeline, what you have already saved, and an amount you could contribute consistently."
        )
    if concern in ["losing money", "losing money again"]:
        return (
            "That concern is worth taking seriously. You can reduce avoidable risk by protecting emergency savings, "
            "using diversification, and choosing a level of investment risk you can live with over time."
        )
    if concern == "not saving enough":
        return (
            "It is understandable to worry about whether you are saving enough. You do not need a perfect number "
            "before you begin; a realistic contribution and a plan to revisit it can create momentum."
        )
    if concern == "falling into debt":
        return (
            "Wanting to stay comfortable and avoid debt is a concrete goal. A useful starting point is to protect "
            "some monthly breathing room, build a small emergency cushion, and make a plan for any high-interest "
            "debt before taking on new financial commitments."
        )
    if concern == "not understanding investing":
        return (
            "You do not need to understand every investing term before taking a next step. It helps to begin with "
            "risk, diversification, time horizon, and the difference between saving and investing."
        )
    if concern == "general financial uncertainty" and (
        parent_topic == "retirement planning"
        or understanding.get("primary_topic") == "retirement planning"
        or goal == "retirement security"
    ):
        return (
            "You do not need a complete retirement plan before you begin. A simple first pass at your timeline and "
            "current savings is enough to make the next step more concrete."
        )
    if goal == "comfortable retirement":
        return (
            "That is a clear and human goal. A comfortable retirement becomes easier to plan for when you translate "
            "it into a rough timeline, expected everyday expenses, and a sustainable saving habit."
        )
    if goal == "financial comfort and stability":
        return (
            "That is a useful goal. Financial comfort often starts with a little monthly breathing room and a buffer "
            "for unexpected expenses, then grows from there."
        )
    if understanding.get("asked_question") and not understanding.get("primary_topic") == "general finances":
        return "It makes sense to pause and get a clear answer before deciding what to do next."
    return ""


def plan_guidance(dialogue_plan: dict, understanding: dict) -> str:
    if dialogue_plan.get("dialogue_act") != "summarize":
        return ""
    goal = understanding.get("current_goal")
    if goal and goal not in ["Not yet detected", "general financial progress"]:
        return (
            f"So far, the useful anchor is your goal: {goal}. A practical next step is to turn that into one "
            "small, concrete action before adding more questions."
        )
    return (
        "A useful place to pause is to choose one priority, such as building emergency savings, preparing for "
        "retirement, or learning about investing. That gives the next step a clearer purpose."
    )


def repeated_turn_guidance(understanding: dict) -> str:
    if understanding.get("concern_type") == "falling into debt":
        return (
            "Let us make that practical. Start with one small snapshot: monthly take-home income, essential expenses, "
            "minimum debt payments, and the amount left over. That shows whether the first move is creating breathing "
            "room, building a starter emergency cushion, or tackling high-interest debt."
        )
    return (
        "Let us turn that into one practical move. Start with the smallest action that would make the situation feel "
        "more concrete, then use what you learn to choose the next step."
    )


def generate_response(
    user_message: str,
    understanding: dict,
    retrieved_knowledge: dict,
    tool_results: dict,
    conversation_context: dict,
    dialogue_plan: dict,
) -> str:
    guardrail_result = evaluate_guardrails(user_message)
    parts = []

    if dialogue_plan.get("dialogue_act") == "clarify_reference":
        return "\n\n".join([
            "I want to make sure I understand what you mean before answering.",
            dialogue_plan["follow_up_question"],
        ])
    if dialogue_plan.get("avoid_repetition"):
        return "\n\n".join(part for part in [
            repeated_turn_guidance(understanding),
            dialogue_plan.get("follow_up_question"),
        ] if part)

    if guardrail_result["mode"] == "educational_only":
        parts.append(guardrail_message(guardrail_result))

    if understanding["intent"] == "follow_up_question" and understanding["resolved_reference"] != "None":
        parts.append(f"You are asking about {understanding['resolved_reference']}.")
    elif understanding["intent"] != "educational_query":
        reflection = REFLECTIONS.get(understanding["intent"])
        if reflection:
            parts.append(reflection)

    concern_response = address_question_or_concern(user_message, understanding)
    if concern_response and (
        understanding.get("expressed_concern")
        or not retrieved_knowledge
        or "safe" in re.findall(r"[a-z0-9']+", user_message.lower())
    ):
        parts.append(concern_response)

    if retrieved_knowledge:
        parts.append(retrieved_knowledge["content"])

    tool_text = format_tool_results(tool_results)
    if tool_text:
        parts.append(tool_text)

    if understanding["intent"] == "decision_support" and not retrieved_knowledge:
        parts.append("A useful next step is to compare risk, timeline, access to your money, and whether your emergency savings are protected.")

    parts.append(plan_guidance(dialogue_plan, understanding))
    parts.append(dialogue_plan.get("follow_up_question"))
    return "\n\n".join(part for part in parts if part)
