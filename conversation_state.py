import re
from copy import deepcopy


DEFAULT_CONTEXT = {
    "primary_topic": None,
    "parent_topic": None,
    "last_product_or_concept": None,
    "resolved_reference": "None",
    "current_goal": "Not yet detected",
    "current_fear": "Not yet detected",
    "unresolved_question": None,
    "persona": "Unknown",
    "confidence_level": "Unknown",
    "financial_literacy": "Unknown",
    "coaching_style": "Supportive",
    "risk_level": "Unknown",
    "recent_referents": [],
    "last_analyzed_text": "",
    "last_assistant_question": None,
    "last_dialogue_act": None,
    "exploratory_questions_in_a_row": 0,
}

TOPIC_RULES = {
    "bitcoin": {"label": "Bitcoin", "terms": ["bitcoin", "btc"], "parent_topic": "investing", "specificity": 100},
    "crypto": {"label": "crypto", "terms": ["crypto", "cryptocurrency", "coin", "token"], "parent_topic": "investing", "specificity": 70},
    "stocks": {"label": "stocks", "terms": ["stock", "stocks", "share", "shares", "equity", "equities"], "parent_topic": "investing", "specificity": 80},
    "etf": {"label": "ETF", "terms": ["index fund", "index funds", "etf", "etfs"], "parent_topic": "investing", "specificity": 90},
    "roth_ira": {"label": "Roth IRA", "terms": ["roth ira", "roth"], "parent_topic": "retirement planning", "specificity": 95},
    "traditional_ira": {"label": "Traditional IRA", "terms": ["traditional ira"], "parent_topic": "retirement planning", "specificity": 95},
    "cd": {"label": "CD", "terms": ["cd", "certificate of deposit", "certificates of deposit"], "parent_topic": "saving", "specificity": 85},
    "emergency_fund": {"label": "emergency fund", "terms": ["emergency fund", "rainy day fund", "cash cushion", "emergencies"], "parent_topic": "financial security", "specificity": 90},
    "diversification": {"label": "diversification", "terms": ["diversification", "diversify", "diversified"], "parent_topic": "investing", "specificity": 82},
    "compound_interest": {"label": "compound interest", "terms": ["compound interest", "compounding"], "parent_topic": "saving", "specificity": 82},
}

BROAD_TOPIC_RULES = {
    "investing": ["invest", "investing", "investment", "investments"],
    "retirement planning": ["retire", "retirement", "401k", "401(k)", "ira"],
    "financial security": ["safe", "secure", "security", "saving", "savings", "save"],
    "saving": ["save", "saving", "savings", "deposit", "bank"],
}

PRONOUN_REFERENCES = {"it", "that", "this", "they", "them"}
ABSTRACT_TOPICS = {"financial security", "general finances", "investing", "retirement planning", "saving"}
CONCERN_TERMS = {"afraid", "anxious", "behind", "concerned", "nervous", "overwhelmed", "scared", "worried"}


def create_conversation_context() -> dict:
    return deepcopy(DEFAULT_CONTEXT)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def topic_term_matches(term: str, lower_text: str, tokens: set[str]) -> bool:
    if " " in term or "(" in term or ")" in term:
        return term in lower_text
    return term in tokens


def topic_context_score(lower_text: str, matched_term: str, specificity: int) -> int:
    mention_position = lower_text.rfind(matched_term)
    recency_bonus = round(30 * mention_position / max(1, len(lower_text)))
    focus_bonus = 0

    for phrase in ["but", "more about", "learn about", "know about", "interested in", "focus on", "tell me about"]:
        phrase_position = lower_text.rfind(phrase)
        if phrase_position != -1 and phrase_position <= mention_position:
            focus_bonus += 15

    return specificity + recency_bonus + focus_bonus


def extract_topics(text: str) -> list[dict]:
    lower_text = text.lower()
    tokens = set(tokenize(text))
    matches = []

    for topic, rule in TOPIC_RULES.items():
        matched_terms = [term for term in rule["terms"] if topic_term_matches(term, lower_text, tokens)]
        if matched_terms:
            matched_term = max(matched_terms, key=lower_text.rfind)
            matches.append({
                "topic": topic,
                "label": rule["label"],
                "parent_topic": rule["parent_topic"],
                "specificity": rule["specificity"],
                "contextual_salience": topic_context_score(lower_text, matched_term, rule["specificity"]),
            })

    return sorted(matches, key=lambda match: match["contextual_salience"], reverse=True)


def detect_parent_topic(text: str) -> str | None:
    lower_text = text.lower()
    tokens = set(tokenize(text))
    for parent_topic, terms in BROAD_TOPIC_RULES.items():
        if any(term in lower_text or term in tokens for term in terms):
            return parent_topic
    return None


def detect_topic(text: str) -> dict | None:
    topics = extract_topics(text)
    broad_parent = detect_parent_topic(text)

    if topics:
        primary = topics[0]
        parent_topic = primary["parent_topic"]
        if broad_parent in ["investing", "retirement planning"]:
            parent_topic = broad_parent
        return {**primary, "parent_topic": parent_topic, "entities": [topic["label"] for topic in topics]}

    if broad_parent:
        return {
            "topic": broad_parent,
            "label": broad_parent,
            "parent_topic": "general finances",
            "entities": [broad_parent],
            "specificity": 20,
            "contextual_salience": 20,
        }
    return None


def uses_context_reference(text: str) -> bool:
    return bool(set(tokenize(text)) & PRONOUN_REFERENCES)


def detect_concern(text: str) -> str | None:
    tokens = set(tokenize(text))
    lower_text = text.lower()
    if "too late" in lower_text or "behind" in tokens:
        return "starting too late"
    if {"debt", "debts", "owe", "owing"} & tokens:
        return "falling into debt"
    if ("lost" in tokens or "loss" in tokens) and {"invested", "investment", "money"} & tokens:
        return "losing money again"
    if {"lose", "losing", "risk"} & tokens:
        return "losing money"
    if "enough" in tokens and {"save", "saving", "saved", "retirement"} & tokens:
        return "not saving enough"
    if any(phrase in lower_text for phrase in ["don't know", "do not know", "don't understand", "do not understand"]):
        return "not understanding investing"
    if CONCERN_TERMS & tokens:
        return "general financial uncertainty"
    return None


def topic_specificity(topic: str | None, label: str | None) -> int:
    if topic in TOPIC_RULES:
        return TOPIC_RULES[topic]["specificity"]
    for rule in TOPIC_RULES.values():
        if label == rule["label"]:
            return rule["specificity"]
    return 20


def semantic_plausibility_score(text: str, candidate: dict) -> int:
    tokens = set(tokenize(text))
    label = str(candidate.get("label"))
    topic = str(candidate.get("topic"))
    is_abstract = label in ABSTRACT_TOPICS or topic in ABSTRACT_TOPICS
    score = -35 if is_abstract else 25

    if {"safe", "risky", "risk", "lose", "losing"} & tokens:
        score += -40 if is_abstract else 35
    if {"open", "account", "contribute"} & tokens and label in ["Roth IRA", "Traditional IRA"]:
        score += 30
    if {"save", "saving", "emergency", "emergencies"} & tokens and label == "emergency fund":
        score += 30
    if {"rate", "interest"} & tokens and label == "CD":
        score += 30
    return score


def rank_reference_candidates(text: str, context: dict) -> list[dict]:
    candidates = []
    for index, referent in enumerate(context.get("recent_referents", [])):
        if isinstance(referent, dict):
            candidates.append({**referent, "recency_score": max(0, 40 - index * 10)})

    if context.get("primary_topic") and context.get("last_product_or_concept"):
        candidates.append({
            "topic": context["primary_topic"],
            "label": context["last_product_or_concept"],
            "parent_topic": context.get("parent_topic"),
            "reference_score": topic_specificity(context["primary_topic"], context["last_product_or_concept"]),
            "recency_score": 35,
        })

    if context.get("parent_topic"):
        candidates.append({
            "topic": context["parent_topic"],
            "label": context["parent_topic"],
            "parent_topic": "general finances",
            "reference_score": 20,
            "recency_score": 15,
        })

    unique_candidates = {}
    for candidate in candidates:
        label = str(candidate.get("label"))
        if label not in unique_candidates or candidate["recency_score"] > unique_candidates[label]["recency_score"]:
            unique_candidates[label] = candidate

    ranked = []
    for candidate in unique_candidates.values():
        reference_score = int(candidate.get("reference_score", topic_specificity(candidate.get("topic"), candidate.get("label"))))
        plausibility = semantic_plausibility_score(text, candidate)
        ranked.append({
            **candidate,
            "specificity_or_context_score": reference_score,
            "plausibility_score": plausibility,
            "score": candidate["recency_score"] + reference_score + plausibility,
        })
    return sorted(ranked, key=lambda candidate: candidate["score"], reverse=True)


def resolve_reference(text: str, context: dict) -> str | None:
    if extract_topics(text) or not uses_context_reference(text):
        return None
    candidates = rank_reference_candidates(text, context)
    return str(candidates[0]["label"]) if candidates else None


def enrich_with_resolved_reference(text: str, resolved_reference: str | None) -> str:
    if not resolved_reference:
        return text
    return f"{text} Context: the pronoun refers to {resolved_reference}."


def classify_intent(text: str, resolved_reference: str | None = None) -> str:
    tokens = set(tokenize(text))
    lower_text = text.lower()
    educational_patterns = ["what is", "what are", "what's", "explain", "define", "meaning of", "how does", "how do"]

    if resolved_reference:
        return "follow_up_question"
    if "?" in text and any(pattern in lower_text for pattern in educational_patterns):
        return "educational_query"
    if detect_concern(text):
        return "emotional_concern"
    if "?" in text and ({"should", "compare", "better", "safe", "risky", "risk", "buy", "choose"} & tokens):
        return "decision_support"
    if ({"invest", "investing", "buy", "open"} & tokens) and ({"learn", "understand", "know"} & tokens):
        return "product_exploration"
    if {"explore", "considering", "interested"} & tokens:
        return "product_exploration"
    return "goal_discovery"


def infer_user_model(text: str, topic: dict | None) -> dict:
    tokens = set(tokenize(text))
    lower_text = text.lower()
    confusion = bool({"confused", "beginner", "new", "learn", "understand", "know", "explain"} & tokens)
    negated_knowledge = any(phrase in lower_text for phrase in ["don't know", "do not know", "don't understand", "do not understand"])
    fear_loss = bool({"lose", "losing", "lost", "loss", "afraid", "scared", "worried", "risk"} & tokens)
    starting_late = bool({"late", "behind", "old", "catch"} & tokens) or "too late" in lower_text
    not_saving_enough = "enough" in tokens and bool({"save", "saving", "saved", "retirement"} & tokens)

    current_goal = "general financial progress"
    if topic:
        current_goal = f"learn about {topic['label']}"
    if {"retire", "retirement"} & tokens:
        current_goal = "retirement security"
    elif {"family", "kids", "children", "college"} & tokens:
        current_goal = "family financial stability"
    elif {"emergency", "emergencies", "buffer", "cushion"} & tokens:
        current_goal = "emergency savings"
    elif "comfortable" in tokens:
        current_goal = "financial comfort and stability"

    current_fear = "not yet clear"
    if "lost" in tokens and bool({"invested", "investment", "money"} & tokens):
        current_fear = "losing money again"
    elif fear_loss:
        current_fear = "losing money"
    elif starting_late:
        current_fear = "starting too late"
    elif not_saving_enough:
        current_fear = "not saving enough"
    elif {"debt", "debts", "owe", "owing"} & tokens:
        current_fear = "falling into debt"
    elif negated_knowledge:
        current_fear = "not understanding investing"

    confidence_level = "low" if current_fear != "not yet clear" or negated_knowledge else "medium"
    financial_literacy = "beginner" if confusion or negated_knowledge else "unknown"
    persona = "steady builder"
    if topic and topic["topic"] == "bitcoin" and confidence_level == "low":
        persona = "cautious_beginner"
    elif current_fear in ["losing money", "losing money again"]:
        persona = "market-anxious investor"
    elif current_fear == "starting too late":
        persona = "late-start planner"
    elif current_fear == "falling into debt":
        persona = "security-focused planner"
    elif financial_literacy == "beginner":
        persona = "new investor"

    coaching_style = "reassuring" if confidence_level == "low" else "supportive"
    risk_level = "low tolerance" if confidence_level == "low" or fear_loss else "moderate"
    return {
        "current_goal": current_goal,
        "current_fear": current_fear,
        "confidence_level": confidence_level,
        "financial_literacy": financial_literacy,
        "coaching_style": coaching_style,
        "persona": persona,
        "risk_level": risk_level,
    }


def analyze_message(user_message: str, conversation_context: dict) -> dict:
    resolved_reference = resolve_reference(user_message, conversation_context)
    analyzed_text = enrich_with_resolved_reference(user_message, resolved_reference)
    topic = detect_topic(analyzed_text)
    user_model = infer_user_model(user_message, topic)
    intent = classify_intent(user_message, resolved_reference)
    concern = detect_concern(user_message)

    if user_model["current_goal"] == "general financial progress":
        prior_goal = conversation_context.get("current_goal")
        if prior_goal and prior_goal != "Not yet detected":
            user_model["current_goal"] = prior_goal
    if "comfortable" in tokenize(user_message) and (
        conversation_context.get("parent_topic") == "retirement planning"
        or conversation_context.get("current_goal") == "retirement security"
    ):
        user_model["current_goal"] = "comfortable retirement"
    if user_model["current_fear"] == "not yet clear":
        prior_fear = conversation_context.get("current_fear")
        if prior_fear and prior_fear != "Not yet detected":
            user_model["current_fear"] = prior_fear
            user_model["confidence_level"] = conversation_context.get("confidence_level", user_model["confidence_level"])
            user_model["coaching_style"] = conversation_context.get("coaching_style", user_model["coaching_style"])
            user_model["persona"] = conversation_context.get("persona", user_model["persona"])
            user_model["risk_level"] = conversation_context.get("risk_level", user_model["risk_level"])

    if (
        topic
        and topic["topic"] == "bitcoin"
        and user_model["financial_literacy"] == "beginner"
        and any(phrase in user_message.lower() for phrase in ["don't know", "do not know", "don't understand", "do not understand"])
    ):
        intent = "learn_before_investing"

    return {
        "intent": intent,
        "primary_topic": topic["topic"] if topic else conversation_context.get("primary_topic") or "general finances",
        "topic_label": topic["label"] if topic else conversation_context.get("last_product_or_concept") or "general finances",
        "parent_topic": topic["parent_topic"] if topic else conversation_context.get("parent_topic") or "general finances",
        "resolved_reference": resolved_reference or "None",
        "recognized_topics": extract_topics(analyzed_text),
        "asked_question": "?" in user_message,
        "expressed_concern": concern is not None,
        "concern_type": concern,
        **user_model,
    }


def update_conversation_context(context: dict, user_message: str, understanding: dict) -> dict:
    context["primary_topic"] = understanding["primary_topic"]
    context["parent_topic"] = understanding["parent_topic"]
    context["last_product_or_concept"] = understanding["topic_label"]
    context["resolved_reference"] = understanding["resolved_reference"]
    context["current_goal"] = understanding["current_goal"]
    context["current_fear"] = understanding["current_fear"]
    context["persona"] = understanding["persona"]
    context["confidence_level"] = understanding["confidence_level"]
    context["financial_literacy"] = understanding["financial_literacy"]
    context["coaching_style"] = understanding["coaching_style"]
    context["risk_level"] = understanding["risk_level"]
    context["unresolved_question"] = user_message if "?" in user_message else None
    context["last_analyzed_text"] = user_message

    topic = detect_topic(user_message)
    if topic and topic["label"] not in ABSTRACT_TOPICS:
        referent = {
            "topic": topic["topic"],
            "label": topic["label"],
            "parent_topic": topic["parent_topic"],
            "reference_score": topic["contextual_salience"],
        }
        recent = [item for item in context.get("recent_referents", []) if item.get("label") != referent["label"]]
        context["recent_referents"] = [referent, *recent][:5]
    return context


def record_assistant_response(context: dict, response: str) -> dict:
    questions = re.findall(r"[^?\n]+\?", response)
    context["last_assistant_question"] = questions[-1].strip() if questions else None
    return context
