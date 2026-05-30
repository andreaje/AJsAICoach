import random
from datetime import date, timedelta

import pandas as pd


RANDOM_SEED = 42
USER_COUNT = 5000

PERSONAS = [
    "market-anxious investor",
    "new investor",
    "late-start planner",
    "action-oriented saver",
    "steady builder",
    "family-focused planner",
]

FUNNEL_STAGES = [
    "started_chat",
    "shared_goal",
    "received_guidance",
    "viewed_recommendation",
    "activated",
]

PERSONA_WEIGHTS = {
    "market-anxious investor": 0.18,
    "new investor": 0.24,
    "late-start planner": 0.14,
    "action-oriented saver": 0.19,
    "steady builder": 0.16,
    "family-focused planner": 0.09,
}

PERSONA_TRAITS = {
    "market-anxious investor": {
        "base_confidence": (1.6, 3.2),
        "literacy": "Beginner",
        "coaching_style": "Reassuring",
        "risk_level": "Low tolerance",
        "escalation_rate": 0.18,
        "activation_rate": 0.28,
        "satisfaction": (3.2, 4.3),
    },
    "new investor": {
        "base_confidence": (1.8, 3.4),
        "literacy": "Beginner",
        "coaching_style": "Educational",
        "risk_level": "Low tolerance",
        "escalation_rate": 0.11,
        "activation_rate": 0.34,
        "satisfaction": (3.4, 4.5),
    },
    "late-start planner": {
        "base_confidence": (2.0, 3.6),
        "literacy": "Beginner",
        "coaching_style": "Reassuring",
        "risk_level": "Moderate",
        "escalation_rate": 0.13,
        "activation_rate": 0.31,
        "satisfaction": (3.3, 4.4),
    },
    "action-oriented saver": {
        "base_confidence": (3.0, 4.5),
        "literacy": "Intermediate",
        "coaching_style": "Direct",
        "risk_level": "Moderate",
        "escalation_rate": 0.05,
        "activation_rate": 0.52,
        "satisfaction": (3.8, 4.8),
    },
    "steady builder": {
        "base_confidence": (3.1, 4.4),
        "literacy": "Intermediate",
        "coaching_style": "Supportive",
        "risk_level": "Open to growth",
        "escalation_rate": 0.06,
        "activation_rate": 0.46,
        "satisfaction": (3.7, 4.7),
    },
    "family-focused planner": {
        "base_confidence": (2.4, 4.0),
        "literacy": "Beginner",
        "coaching_style": "Supportive",
        "risk_level": "Moderate",
        "escalation_rate": 0.09,
        "activation_rate": 0.39,
        "satisfaction": (3.5, 4.6),
    },
}

TOPIC_CATEGORIES = {
    "Bitcoin": "investing",
    "ETFs": "investing",
    "Roth IRA": "retirement planning",
    "emergency fund": "financial security",
    "stocks": "investing",
    "retirement": "retirement planning",
}

GOALS_BY_PERSONA = {
    "market-anxious investor": ["avoid losses", "understand risk", "rebuild confidence"],
    "new investor": ["learn investing basics", "start small", "choose first investment"],
    "late-start planner": ["catch up for retirement", "build monthly habit", "estimate retirement gap"],
    "action-oriented saver": ["pick next step", "automate investing", "compare options"],
    "steady builder": ["stay consistent", "optimize portfolio", "increase recurring deposit"],
    "family-focused planner": ["family security", "college savings", "emergency cushion"],
}


def weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    return rng.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def pick_funnel_stage(rng: random.Random, activated: bool, persona: str) -> str:
    if activated:
        return "activated"

    if persona in ["action-oriented saver", "steady builder"]:
        weights = [0.08, 0.16, 0.28, 0.48]
    elif persona == "market-anxious investor":
        weights = [0.18, 0.30, 0.34, 0.18]
    else:
        weights = [0.12, 0.26, 0.38, 0.24]

    return rng.choices(FUNNEL_STAGES[:-1], weights=weights, k=1)[0]


def generate_users(rng: random.Random, user_count: int) -> pd.DataFrame:
    rows = []
    start_date = date(2026, 1, 1)

    for index in range(1, user_count + 1):
        persona = weighted_choice(rng, PERSONA_WEIGHTS)
        traits = PERSONA_TRAITS[persona]
        confidence_before = round(rng.uniform(*traits["base_confidence"]), 1)
        confidence_lift = round(clamp(rng.gauss(0.9, 0.45), 0.0, 2.4), 1)
        confidence_after = round(clamp(confidence_before + confidence_lift, 1.0, 5.0), 1)
        activated = rng.random() < traits["activation_rate"]
        escalated = rng.random() < traits["escalation_rate"]
        satisfaction = round(clamp(rng.uniform(*traits["satisfaction"]) - (0.35 if escalated else 0), 1.0, 5.0), 1)

        recurring_deposit_started = activated and rng.random() < 0.72
        first_investment_started = activated and rng.random() < 0.58
        education_completed = rng.random() < (0.67 if traits["literacy"] == "Beginner" else 0.43)
        user_type = rng.choices(["New User", "Existing Customer"], weights=[0.62, 0.38], k=1)[0]
        age_band = rng.choices(["18-24", "25-34", "35-44", "45-54", "55+"], weights=[0.13, 0.31, 0.25, 0.18, 0.13], k=1)[0]

        rows.append({
            "user_id": f"user_{index:05d}",
            "signup_date": start_date + timedelta(days=rng.randint(0, 149)),
            "persona": persona,
            "primary_goal": rng.choice(GOALS_BY_PERSONA[persona]),
            "financial_literacy": traits["literacy"],
            "user_type": user_type,
            "age_band": age_band,
            "coaching_style": traits["coaching_style"],
            "risk_level": traits["risk_level"],
            "confidence_before": confidence_before,
            "confidence_after": confidence_after,
            "confidence_lift": round(confidence_after - confidence_before, 1),
            "inferred_satisfaction": satisfaction,
            "escalated_to_human": escalated,
            "activated": activated,
            "funnel_stage": pick_funnel_stage(rng, activated, persona),
            "recurring_deposit_started": recurring_deposit_started,
            "first_investment_started": first_investment_started,
            "education_completed": education_completed,
        })

    return pd.DataFrame(rows)


def generate_sessions(rng: random.Random, users: pd.DataFrame) -> pd.DataFrame:
    rows = []
    topics = list(TOPIC_CATEGORIES)
    session_id = 1

    for user in users.itertuples(index=False):
        session_count = rng.choices([1, 2, 3, 4], weights=[0.42, 0.31, 0.19, 0.08], k=1)[0]
        first_session_date = user.signup_date + timedelta(days=rng.randint(0, 14))

        for session_index in range(session_count):
            completed = rng.random() < (0.82 if user.inferred_satisfaction >= 3.8 else 0.68)
            escalated = user.escalated_to_human and session_index == session_count - 1
            messages = rng.randint(4, 14) + (3 if escalated else 0)
            confidence_delta = round(clamp(rng.gauss(user.confidence_lift / session_count, 0.25), -0.2, 1.4), 1)
            topic = rng.choice(topics)
            accepted_next_step = user.activated and rng.random() < 0.76
            resolved_reference_success = rng.random() < (0.93 if user.financial_literacy == "Intermediate" else 0.86)

            rows.append({
                "session_id": f"session_{session_id:06d}",
                "user_id": user.user_id,
                "session_date": first_session_date + timedelta(days=session_index * rng.randint(1, 10)),
                "persona": user.persona,
                "user_type": user.user_type,
                "age_band": user.age_band,
                "financial_literacy": user.financial_literacy,
                "coaching_style": user.coaching_style,
                "risk_level": user.risk_level,
                "topic": topic,
                "topic_category": TOPIC_CATEGORIES[topic],
                "funnel_stage": user.funnel_stage if session_index == session_count - 1 else "received_guidance",
                "message_count": messages,
                "completed_session": completed,
                "inferred_satisfaction": user.inferred_satisfaction,
                "confidence_lift": confidence_delta,
                "escalated_to_human": escalated,
                "activated_after_session": user.activated and session_index == session_count - 1,
                "recommended_action_taken": accepted_next_step,
                "resolved_reference_success": resolved_reference_success,
            })
            session_id += 1

    return pd.DataFrame(rows)


def generate_metric_summary(users: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "users": len(users),
        "sessions": len(sessions),
        "activation_rate": round(users["activated"].mean(), 3),
        "escalation_rate": round(sessions["escalated_to_human"].mean(), 3),
        "avg_inferred_satisfaction": round(users["inferred_satisfaction"].mean(), 2),
        "avg_confidence_lift": round(users["confidence_lift"].mean(), 2),
        "recurring_deposit_rate": round(users["recurring_deposit_started"].mean(), 3),
        "first_investment_rate": round(users["first_investment_started"].mean(), 3),
        "education_completion_rate": round(users["education_completed"].mean(), 3),
        "accepted_next_step_rate": round(sessions["recommended_action_taken"].mean(), 3),
        "resolved_reference_success_rate": round(sessions["resolved_reference_success"].mean(), 3),
    }])


def generate_mock_data(user_count: int = USER_COUNT, seed: int = RANDOM_SEED) -> dict[str, pd.DataFrame]:
    rng = random.Random(seed)
    users = generate_users(rng, user_count)
    sessions = generate_sessions(rng, users)
    metric_summary = generate_metric_summary(users, sessions)
    funnel = users["funnel_stage"].value_counts().reindex(FUNNEL_STAGES, fill_value=0).reset_index()
    funnel.columns = ["funnel_stage", "users"]

    return {
        "users": users,
        "sessions": sessions,
        "metric_summary": metric_summary,
        "funnel": funnel,
    }
