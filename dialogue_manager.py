import re


FOLLOW_UPS = {
    "cd": "Are you exploring CDs as a place to keep savings, or comparing them with investing options?",
    "bitcoin": "Would it help to talk through how volatility, time horizon, and diversification affect that decision?",
    "etf": "Are you learning about ETFs for a first investment, or comparing them with another option?",
    "roth_ira": "Are you trying to understand how a Roth IRA works, or whether opening one belongs on your next-step list?",
    "traditional_ira": "Are you comparing a Traditional IRA with another retirement account?",
    "emergency_fund": "What kind of unexpected expense would you most want that fund to cover?",
    "diversification": "Would an example of a more diversified approach make this clearer?",
    "compound_interest": "Would it help to walk through a small numerical example?",
}

EXPLORATORY_ACTS = {"explore_goal", "gather_information"}
REFERENCE_WORDS = {"it", "that", "this", "they", "them"}


def normalize_question(question: str | None) -> str:
    return re.sub(r"\s+", " ", (question or "").strip().lower())


def normalize_message(message: str | None) -> str:
    return re.sub(r"\s+", " ", (message or "").strip().lower())


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def needs_reference_clarification(understanding: dict, latest_user_message: str) -> bool:
    return (
        bool(understanding.get("asked_question"))
        and bool(tokens(latest_user_message) & REFERENCE_WORDS)
        and understanding.get("resolved_reference") == "None"
    )


def choose_follow_up(understanding: dict, user_model: dict) -> tuple[str | None, str | None]:
    topic = understanding.get("primary_topic")
    concern = understanding.get("concern_type")
    goal = user_model.get("current_goal")

    if concern == "starting too late":
        return "retirement timeline", "Roughly how many years do you have until you would like to retire?"
    if concern == "falling into debt":
        return "whether the user wants to prevent or reduce debt", "Is your main concern avoiding new debt, paying down debt you already have, or both?"
    if concern == "general financial uncertainty" and (
        understanding.get("parent_topic") == "retirement planning"
        or topic == "retirement planning"
        or goal == "retirement security"
    ):
        return "preferred retirement starting point", "Would it be easier to start with your timeline or with a manageable monthly saving target?"
    if goal == "comfortable retirement":
        return "meaning of a comfortable retirement", "When you picture comfortable, is the priority covering everyday expenses, having room for extras, or both?"
    if goal == "financial comfort and stability":
        return "first priority for financial stability", "Would it help to start with monthly breathing room, an emergency cushion, or a plan for existing debt?"
    if topic in FOLLOW_UPS:
        return f"how {topic} fits the user's goal", FOLLOW_UPS[topic]
    if understanding.get("intent") == "emotional_concern":
        return "most important source of uncertainty", "What part of this feels most important to make less uncertain first?"
    if understanding.get("intent") == "goal_discovery":
        return "desired financial outcome", "What outcome would make the biggest difference for you right now?"
    return "remaining uncertainty", "What would you like to make clearer next?"


def decide_dialogue_plan(
    understanding: dict,
    user_model: dict,
    latest_user_message: str,
    conversation_context: dict | None = None,
) -> dict:
    context = conversation_context or {}
    asked_question = bool(understanding.get("asked_question"))
    expressed_concern = bool(understanding.get("expressed_concern"))
    intent = understanding.get("intent")
    exploratory_streak = int(context.get("exploratory_questions_in_a_row", 0))
    repeated_user_message = normalize_message(latest_user_message) == normalize_message(context.get("last_analyzed_text"))

    if needs_reference_clarification(understanding, latest_user_message):
        dialogue_act = "clarify_reference"
        response_goal = "Clarify the user's reference before giving guidance."
        must_address = ["ambiguous reference"]
        next_information_needed = "the product, goal, or concept the user means"
        follow_up_question = "What does that refer to?"
    elif asked_question:
        dialogue_act = "answer_question"
        response_goal = "Answer the user's question directly before offering any next step."
        must_address = ["direct question"]
        if expressed_concern:
            must_address.append("expressed concern")
        next_information_needed, follow_up_question = choose_follow_up(understanding, user_model)
    elif expressed_concern:
        dialogue_act = "reassure"
        response_goal = "Acknowledge the concern and reduce uncertainty with a concrete, supportive perspective."
        must_address = ["expressed concern"]
        next_information_needed, follow_up_question = choose_follow_up(understanding, user_model)
    elif intent in ["educational_query", "learn_before_investing", "product_exploration"]:
        dialogue_act = "educate"
        response_goal = "Explain the relevant concept in plain language and connect it to the user's goal."
        must_address = ["knowledge gap"]
        next_information_needed, follow_up_question = choose_follow_up(understanding, user_model)
    elif user_model.get("current_goal") in [None, "Not yet detected", "general financial progress"]:
        dialogue_act = "explore_goal"
        response_goal = "Help the user identify the outcome that matters most."
        must_address = ["unclear goal"]
        next_information_needed, follow_up_question = choose_follow_up(understanding, user_model)
    elif intent == "decision_support":
        dialogue_act = "suggest_next_step"
        response_goal = "Offer a practical way to compare tradeoffs without making the decision for the user."
        must_address = ["decision tradeoffs"]
        next_information_needed, follow_up_question = choose_follow_up(understanding, user_model)
    else:
        dialogue_act = "gather_information"
        response_goal = "Gather one useful detail that moves the user toward a concrete next step."
        must_address = ["next planning detail"]
        next_information_needed, follow_up_question = choose_follow_up(understanding, user_model)

    if repeated_user_message and dialogue_act != "clarify_reference":
        dialogue_act = "suggest_next_step"
        response_goal = "Offer a concrete next step instead of restating the previous response."
        must_address = ["practical next step"]
        follow_up_question = "Would it help to focus on one practical next step together?"

    previous_question = normalize_question(context.get("last_assistant_question"))
    if normalize_question(follow_up_question) == previous_question:
        follow_up_question = "Would it help to focus on one practical next step together?"
    if normalize_question(follow_up_question) == previous_question:
        follow_up_question = None
    if dialogue_act in EXPLORATORY_ACTS and exploratory_streak >= 2:
        dialogue_act = "summarize"
        response_goal = "Summarize what is known and provide guidance before asking for more information."
        must_address = ["conversation progress", "useful guidance"]
        follow_up_question = None

    return {
        "dialogue_act": dialogue_act,
        "response_goal": response_goal,
        "must_address": must_address,
        "next_information_needed": next_information_needed,
        "follow_up_question": follow_up_question,
        "avoid_repetition": repeated_user_message,
    }


def record_dialogue_plan(context: dict, dialogue_plan: dict) -> dict:
    if dialogue_plan["dialogue_act"] in EXPLORATORY_ACTS and dialogue_plan.get("follow_up_question"):
        context["exploratory_questions_in_a_row"] = int(context.get("exploratory_questions_in_a_row", 0)) + 1
    else:
        context["exploratory_questions_in_a_row"] = 0
    context["last_dialogue_act"] = dialogue_plan["dialogue_act"]
    return context
