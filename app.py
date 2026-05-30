import importlib

import pandas as pd
import streamlit as st

from conversation_state import (
    TOPIC_RULES,
    analyze_message,
    create_conversation_context,
    record_assistant_response,
    update_conversation_context,
)
from dialogue_manager import decide_dialogue_plan, record_dialogue_plan
from guardrails import evaluate_guardrails
from knowledge_base import retrieve_knowledge
from llm_client import generate_response_details
import mock_data as mock_data_module
from tools import collect_tool_results

st.set_page_config(page_title="Acorns AI Coach", layout="wide")

st.title("🌱 Acorns AI Coach")
st.caption("Helping everyday people build financial confidence.")

ASSISTANT_AVATAR = "🐿️"
MOCK_DATA_SCHEMA_VERSION = 6

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "When you think about your financial future, what are you hoping for most?",
        }
    ]

fresh_context = create_conversation_context()
existing_context = st.session_state.get("conversation_context", {})
if existing_context.get("active_topic") and not existing_context.get("primary_topic"):
    existing_context["primary_topic"] = existing_context["active_topic"]
if existing_context.get("active_topic_label") and not existing_context.get("last_product_or_concept"):
    existing_context["last_product_or_concept"] = existing_context["active_topic_label"]
st.session_state.conversation_context = {**fresh_context, **existing_context}

if "last_understanding" not in st.session_state:
    st.session_state.last_understanding = {}

if "last_retrieved_knowledge" not in st.session_state:
    st.session_state.last_retrieved_knowledge = {}

if "last_tool_results" not in st.session_state:
    st.session_state.last_tool_results = {}

if "last_guardrail_result" not in st.session_state:
    st.session_state.last_guardrail_result = {"mode": "standard", "categories": []}

if "last_dialogue_plan" not in st.session_state:
    st.session_state.last_dialogue_plan = {}

if "last_response_debug" not in st.session_state:
    st.session_state.last_response_debug = {}


@st.cache_data
def load_mock_data(schema_version: int):
    return mock_data_module.generate_mock_data()


def reset_conversation():
    st.session_state.clear()
    st.rerun()


def multi_filter(label: str, values: pd.Series, key: str) -> list[str]:
    options = sorted(values.dropna().unique().tolist())
    return st.multiselect(label, options, default=options, key=key)


REQUIRED_USER_COLUMNS = {
    "user_id",
    "persona",
    "user_type",
    "age_band",
    "financial_literacy",
    "coaching_style",
    "risk_level",
    "funnel_stage",
    "activated",
    "inferred_satisfaction",
    "confidence_before",
    "confidence_after",
    "confidence_lift",
    "helpfulness",
    "trust",
    "confidence_building",
    "efficiency",
    "adaptability",
    "recurring_deposit_started",
    "signup_date",
}

REQUIRED_SESSION_COLUMNS = {
    "user_id",
    "persona",
    "topic_category",
    "recommended_action_taken",
    "escalated_to_human",
    "resolved_reference_success",
    "intent_accuracy",
    "slot_completion_accuracy",
    "context_retention",
    "dialogue_policy_accuracy",
    "retrieval_relevance",
    "groundedness",
    "perceived_understanding",
    "factual_accuracy",
    "safety_compliance",
    "integrity_policy_compliance",
    "session_date",
    "completed_session",
}


def validate_dashboard_schema(users: pd.DataFrame, sessions: pd.DataFrame, show_warning: bool = True) -> bool:
    missing_user_columns = sorted(REQUIRED_USER_COLUMNS - set(users.columns))
    missing_session_columns = sorted(REQUIRED_SESSION_COLUMNS - set(sessions.columns))

    if not missing_user_columns and not missing_session_columns:
        return True

    missing_details = []
    if missing_user_columns:
        missing_details.append(f"users DataFrame: {', '.join(missing_user_columns)}")
    if missing_session_columns:
        missing_details.append(f"sessions DataFrame: {', '.join(missing_session_columns)}")

    if show_warning:
        st.warning(
            "Analytics cannot render because the mock-data schema is missing required columns. "
            + " | ".join(missing_details)
            + ". Clear the Streamlit cache or update mock_data.py so the generated schema matches the dashboard."
        )
    return False


def horizontal_bar_chart(data: pd.DataFrame, category: str, values: list[str]):
    chart_data = data.reset_index()
    if category not in chart_data.columns:
        chart_data = chart_data.rename(columns={chart_data.columns[0]: category})

    if len(values) == 1:
        value = values[0]
        spec = {
            "mark": {"type": "bar", "cornerRadiusEnd": 3},
            "encoding": {
                "y": {"field": category, "type": "nominal", "sort": "-x", "title": None},
                "x": {"field": value, "type": "quantitative", "title": value.replace("_", " ").title()},
                "tooltip": [
                    {"field": category, "type": "nominal", "title": category.replace("_", " ").title()},
                    {"field": value, "type": "quantitative", "title": value.replace("_", " ").title()},
                ],
            },
        }
    else:
        chart_data = chart_data.melt(id_vars=[category], value_vars=values, var_name="measure", value_name="value")
        spec = {
            "mark": {"type": "bar", "cornerRadiusEnd": 3},
            "encoding": {
                "y": {"field": category, "type": "nominal", "title": None},
                "x": {"field": "value", "type": "quantitative", "title": "Average Confidence"},
                "yOffset": {"field": "measure"},
                "color": {"field": "measure", "type": "nominal", "title": None},
                "tooltip": [
                    {"field": category, "type": "nominal", "title": category.replace("_", " ").title()},
                    {"field": "measure", "type": "nominal", "title": "Measure"},
                    {"field": "value", "type": "quantitative", "title": "Value", "format": ".2f"},
                ],
            },
        }
    st.vega_lite_chart(chart_data, spec, width="stretch")


def filter_mock_data(users: pd.DataFrame, sessions: pd.DataFrame, key_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    with st.popover("Filters"):
        st.caption("Select one or more values in each dropdown.")
        personas = multi_filter("Persona", users["persona"], f"{key_prefix}_persona")
        user_types = multi_filter("User type", users["user_type"], f"{key_prefix}_user_type")
        literacy_levels = multi_filter("Financial literacy", users["financial_literacy"], f"{key_prefix}_literacy")
        coaching_styles = multi_filter("Coaching style", users["coaching_style"], f"{key_prefix}_coaching_style")
        risk_levels = multi_filter("Risk level", users["risk_level"], f"{key_prefix}_risk_level")
        topic_categories = multi_filter("Topic category", sessions["topic_category"], f"{key_prefix}_topic_category")

    filtered_users = users[
        users["persona"].isin(personas)
        & users["user_type"].isin(user_types)
        & users["financial_literacy"].isin(literacy_levels)
        & users["coaching_style"].isin(coaching_styles)
        & users["risk_level"].isin(risk_levels)
    ]
    filtered_sessions = sessions[
        sessions["user_id"].isin(filtered_users["user_id"])
        & sessions["topic_category"].isin(topic_categories)
    ]
    filtered_users = filtered_users[filtered_users["user_id"].isin(filtered_sessions["user_id"])]
    return filtered_users, filtered_sessions


def render_funnel(stage_counts: list[tuple[str, int]]):
    counts = [count for _, count in stage_counts]
    colors = ["#2f855a", "#3f956a", "#52a67a", "#69b98d", "#82c9a1"]
    rows = []

    for (stage, count), color in zip(stage_counts, colors * 2):
        width = max(34, round(count / max(counts) * 100))
        rows.append(
            f"""
            <div style="width:{width}%; background:{color}; color:white; margin:0 auto 6px; padding:10px 8px;
                text-align:center; border-radius:4px; font-size:14px; line-height:1.2;">
                <strong>{stage}</strong><br><span>{count:,} users</span>
            </div>
            """
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def load_analytics_data() -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    mock_data = load_mock_data(MOCK_DATA_SCHEMA_VERSION)
    users = mock_data["users"]
    sessions = mock_data["sessions"]

    if not validate_dashboard_schema(users, sessions, show_warning=False):
        load_mock_data.clear()
        importlib.reload(mock_data_module)
        mock_data = load_mock_data(MOCK_DATA_SCHEMA_VERSION)
        users = mock_data["users"]
        sessions = mock_data["sessions"]
        if not validate_dashboard_schema(users, sessions):
            return None, None
    return users, sessions


def active_user_trend(sessions: pd.DataFrame) -> pd.DataFrame:
    activity = sessions[["session_date", "user_id"]].copy()
    activity["session_date"] = pd.to_datetime(activity["session_date"])
    dates = pd.date_range(activity["session_date"].min(), activity["session_date"].max(), freq="D")
    rows = []
    for current_date in dates:
        rows.append({
            "date": current_date,
            "DAU": activity.loc[activity["session_date"] == current_date, "user_id"].nunique(),
            "WAU": activity.loc[activity["session_date"].between(current_date - pd.Timedelta(days=6), current_date), "user_id"].nunique(),
            "MAU": activity.loc[activity["session_date"].between(current_date - pd.Timedelta(days=29), current_date), "user_id"].nunique(),
        })
    return pd.DataFrame(rows).set_index("date")


def retention_trend(users: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    session_counts = sessions["user_id"].value_counts()
    retention = users[["user_id", "signup_date"]].copy()
    retention["signup_week"] = pd.to_datetime(retention["signup_date"]).dt.to_period("W").dt.start_time
    retention["returned_user"] = retention["user_id"].map(session_counts).fillna(0).gt(1)
    return retention.groupby("signup_week")["returned_user"].mean().mul(100).to_frame("Return Usage Rate")


def render_business_metrics():
    users, sessions = load_analytics_data()
    if users is None:
        return
    users, sessions = filter_mock_data(users, sessions, "business")
    if users.empty or sessions.empty:
        st.warning("No mocked sessions match these filters.")
        return

    st.markdown("**Business Metrics answer:** \"Is the product creating value?\"")
    st.caption(f"{len(users):,} synthetic users and {len(sessions):,} coaching sessions in the filtered view.")

    st.header("Acquisition & Engagement")
    trend = active_user_trend(sessions)
    latest = trend.iloc[-1]
    engagement = st.columns(5)
    engagement[0].metric("DAU", f"{latest['DAU']:,.0f}")
    engagement[1].metric("WAU", f"{latest['WAU']:,.0f}")
    engagement[2].metric("MAU", f"{latest['MAU']:,.0f}")
    engagement[3].metric("DAU / MAU", f"{latest['DAU'] / max(1, latest['MAU']):.1%}")
    engagement[4].metric("Average Sessions / User", f"{len(sessions) / len(users):.2f}")
    st.subheader("DAU / WAU / MAU Trend")
    st.line_chart(trend)

    st.header("Funnel")
    old_stage_rank = {"started_chat": 0, "shared_goal": 1, "received_guidance": 2, "viewed_recommendation": 3, "activated": 4}
    user_stage_rank = users["funnel_stage"].map(old_stage_rank)
    completed_onboarding = sessions.loc[sessions["completed_session"], "user_id"].nunique()
    raw_counts = [
        ("Started conversation", len(users)),
        ("Completed onboarding", completed_onboarding),
        ("Goal identified", user_stage_rank.ge(1).sum()),
        ("Recommendation delivered", user_stage_rank.ge(2).sum()),
        ("Next step accepted", sessions.loc[sessions["recommended_action_taken"], "user_id"].nunique()),
        ("Returned user", sessions["user_id"].value_counts().gt(1).sum()),
    ]
    funnel_counts = []
    previous_count = len(users)
    for label, count in raw_counts:
        previous_count = min(previous_count, int(count))
        funnel_counts.append((label, previous_count))
    render_funnel(funnel_counts)

    st.header("Business Outcomes")
    returning_users = sessions["user_id"].value_counts().gt(1).sum() / len(users)
    outcomes = st.columns(5)
    outcomes[0].metric("Activation Rate", f"{users['activated'].mean():.1%}")
    outcomes[1].metric("Accepted Next-Step Rate", f"{sessions['recommended_action_taken'].mean():.1%}")
    outcomes[2].metric("Return Usage Rate", f"{returning_users:.1%}")
    outcomes[3].metric("Retention Proxy", f"{users['recurring_deposit_started'].mean():.1%}")
    outcomes[4].metric("Escalation Rate", f"{sessions['escalated_to_human'].mean():.1%}")

    left, right = st.columns(2)
    with left:
        st.subheader("Persona Breakdown")
        horizontal_bar_chart(users["persona"].value_counts().rename_axis("persona").to_frame("users"), "persona", ["users"])
    with right:
        st.subheader("Retention Trend")
        st.line_chart(retention_trend(users, sessions))


def render_system_metrics():
    users, sessions = load_analytics_data()
    if users is None:
        return
    users, sessions = filter_mock_data(users, sessions, "system")
    if users.empty or sessions.empty:
        st.warning("No mocked sessions match these filters.")
        return

    st.markdown("**System Metrics answer:** \"Is the AI system behaving correctly and why?\"")
    st.caption(f"{len(users):,} synthetic users and {len(sessions):,} coaching sessions in the filtered view.")

    st.header("Critical Quality Metrics")
    st.caption("Non-negotiable gating metrics. These must remain healthy before optimizing other outcomes.")
    quality = st.columns(3)
    quality[0].metric("Accuracy", f"{sessions['factual_accuracy'].mean():.1%}")
    quality[1].metric("Safety", f"{sessions['safety_compliance'].mean():.1%}")
    quality[2].metric("Integrity / Policy Compliance", f"{sessions['integrity_policy_compliance'].mean():.1%}")
    st.markdown(
        "**Accuracy:** Is the information factually correct?  \n"
        "**Safety:** Does the response avoid causing harm?  \n"
        "**Integrity:** Does the response stay within intended behavioral and regulatory boundaries?"
    )
    st.caption("Examples include harmful financial advice, failure to escalate, insider trading guidance, and unauthorized personalized investment recommendations.")

    st.header("User Experience Metrics")
    st.caption("Primary user outcomes: whether the coaching experience creates value with reasonable effort.")
    ux_metrics = {
        "Helpfulness": users["helpfulness"].mean(),
        "Trust": users["trust"].mean(),
        "Confidence Building": users["confidence_building"].mean(),
        "Adaptability": users["adaptability"].mean(),
        "Efficiency": users["efficiency"].mean(),
        "Satisfaction": users["inferred_satisfaction"].mean(),
    }
    scorecards = st.columns(6)
    for column, (label, value) in zip(scorecards, ux_metrics.items()):
        column.metric(label, f"{value:.2f} / 5")
    st.markdown(
        "**Helpfulness:** Did the response help the user move toward their goal?  \n"
        "**Trust:** Would the user reasonably rely on the coach?  \n"
        "**Confidence Building:** Did the interaction increase confidence?  \n"
        "**Adaptability:** Did the coach appropriately adjust to the user?  \n"
        "**Efficiency:** Did the user achieve value with reasonable effort?  \n"
        "**Satisfaction:** Overall user sentiment."
    )
    ux_daily = sessions[["session_date", "user_id"]].merge(
        users[["user_id", "helpfulness", "trust", "confidence_building", "adaptability", "efficiency", "inferred_satisfaction"]],
        on="user_id",
    )
    ux_daily["session_date"] = pd.to_datetime(ux_daily["session_date"])
    ux_trend = ux_daily.groupby("session_date")[["helpfulness", "trust", "confidence_building", "adaptability", "efficiency", "inferred_satisfaction"]].mean()
    st.subheader("UX Metric Trend")
    st.line_chart(ux_trend)

    st.header("Diagnostic Metrics")
    st.caption("Explanatory metrics that show why quality and UX outcomes are moving.")
    diagnostic_chart = pd.DataFrame([
        {"metric": "Intent Accuracy", "module": "Conversation Understanding", "score_pct": sessions["intent_accuracy"].mean() * 100},
        {"metric": "Reference Resolution Accuracy", "module": "Conversation Understanding", "score_pct": sessions["resolved_reference_success"].mean() * 100},
        {"metric": "Slot Completion Accuracy", "module": "Conversation Understanding", "score_pct": sessions["slot_completion_accuracy"].mean() * 100},
        {"metric": "Context Retention", "module": "Conversation Understanding", "score_pct": sessions["context_retention"].mean() * 100},
        {"metric": "Dialogue Policy Accuracy", "module": "Dialogue Management", "score_pct": sessions["dialogue_policy_accuracy"].mean() * 100},
        {"metric": "Perceived Understanding", "module": "Dialogue Management", "score_pct": sessions["perceived_understanding"].mean() / 5 * 100},
        {"metric": "Retrieval Relevance", "module": "Knowledge Retrieval", "score_pct": sessions["retrieval_relevance"].mean() * 100},
        {"metric": "Groundedness", "module": "Knowledge Retrieval", "score_pct": sessions["groundedness"].mean() * 100},
    ])
    diagnostic_spec = {
        "mark": {"type": "bar", "cornerRadiusEnd": 3},
        "encoding": {
            "y": {"field": "metric", "type": "nominal", "sort": "-x", "title": None},
            "x": {"field": "score_pct", "type": "quantitative", "title": "Score (%)"},
            "color": {"field": "module", "type": "nominal", "title": "Diagnostic Layer"},
            "tooltip": [
                {"field": "metric", "type": "nominal", "title": "Metric"},
                {"field": "module", "type": "nominal", "title": "Diagnostic Layer"},
                {"field": "score_pct", "type": "quantitative", "title": "Score (%)", "format": ".1f"},
            ],
        },
    }
    st.vega_lite_chart(diagnostic_chart, diagnostic_spec, width="stretch")
    st.info(
        "High Perceived Understanding + Low Helpfulness may indicate a Dialogue Manager failure. "
        "The system understood the user but selected the wrong conversational action."
    )


def render_sidebar():
    context = st.session_state.conversation_context
    st.header("Customer Understanding")
    sidebar_fields = {
        "Primary Topic": context["primary_topic"] or "None",
        "Parent Topic": context["parent_topic"] or "None",
        "Resolved Reference": context["resolved_reference"],
        "Current Goal": context["current_goal"],
        "Current Fear": context["current_fear"],
        "Persona": context["persona"],
        "Confidence": context["confidence_level"],
        "Coaching Style": context["coaching_style"],
        "Risk Level": context["risk_level"],
    }
    for label, value in sidebar_fields.items():
        st.markdown(f"**{label}:** {value}")
    if st.button("Reset conversation"):
        reset_conversation()


def render_understanding_tab():
    st.subheader("Conversation Context")
    st.json(st.session_state.conversation_context)

    st.subheader("Current Message Understanding")
    if st.session_state.last_understanding:
        st.json(st.session_state.last_understanding)
    else:
        st.info("No user message has been analyzed yet.")

    st.subheader("Dialogue Plan")
    if st.session_state.last_dialogue_plan:
        st.json(st.session_state.last_dialogue_plan)
    else:
        st.info("No dialogue objective has been selected yet.")

    st.subheader("Recognized Financial Topics")
    topics = st.session_state.last_understanding.get("recognized_topics", [])
    if topics:
        rows = [
            {
                "primary_topic": index == 0,
                "topic": topic["topic"],
                "label": topic["label"],
                "parent_topic": topic["parent_topic"],
                "specificity": topic["specificity"],
                "contextual_salience": topic["contextual_salience"],
            }
            for index, topic in enumerate(topics)
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("No financial topic has been recognized in the current conversation yet.")

    st.subheader("RAG Data")
    if st.session_state.last_retrieved_knowledge:
        st.json(st.session_state.last_retrieved_knowledge)
    else:
        st.info("No local knowledge entry was retrieved for the current message.")

    st.subheader("Tool Results")
    if st.session_state.last_tool_results:
        st.json(st.session_state.last_tool_results)
    else:
        st.info("No mocked tool was needed for the current message.")

    st.subheader("Guardrails")
    st.json(st.session_state.last_guardrail_result)


with st.sidebar:
    render_sidebar()

coach_tab, understanding_tab, business_metrics_tab, system_metrics_tab = st.tabs(
    ["Coach", "Conversational Understanding", "Business Metrics", "System Metrics"]
)

with coach_tab:
    for message in st.session_state.messages:
        avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.write(message["content"])

    with st.expander("Response Debug", expanded=False):
        if st.session_state.last_response_debug:
            st.json(st.session_state.last_response_debug)
        else:
            st.info("No response has been generated yet.")

    prompt = st.chat_input("Share what's on your mind...")
    if prompt:
        understanding = analyze_message(prompt, st.session_state.conversation_context)
        user_model = {
            key: understanding.get(key)
            for key in [
                "current_goal",
                "current_fear",
                "confidence_level",
                "financial_literacy",
                "coaching_style",
                "persona",
                "risk_level",
            ]
        }
        dialogue_plan = decide_dialogue_plan(
            understanding,
            user_model,
            prompt,
            st.session_state.conversation_context,
        )
        retrieved_knowledge = retrieve_knowledge(understanding["primary_topic"])
        tool_results = collect_tool_results(prompt, understanding["primary_topic"])
        guardrail_result = evaluate_guardrails(prompt)
        response_details = generate_response_details(
            prompt,
            understanding,
            retrieved_knowledge,
            tool_results,
            st.session_state.conversation_context,
            dialogue_plan,
            guardrail_result,
        )
        response = response_details["text"]
        update_conversation_context(st.session_state.conversation_context, prompt, understanding)
        record_assistant_response(st.session_state.conversation_context, response)
        record_dialogue_plan(st.session_state.conversation_context, dialogue_plan)
        st.session_state.last_understanding = understanding
        st.session_state.last_dialogue_plan = dialogue_plan
        st.session_state.last_retrieved_knowledge = retrieved_knowledge
        st.session_state.last_tool_results = tool_results
        st.session_state.last_guardrail_result = guardrail_result
        st.session_state.last_response_debug = {
            "detected_intent": understanding["intent"],
            "primary_topic": understanding["primary_topic"],
            "parent_topic": understanding["parent_topic"],
            "dialogue_act": dialogue_plan["dialogue_act"],
            "response_source": response_details["response_source"],
            "retrieved_knowledge_used": response_details["retrieved_knowledge_used"],
        }
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

with understanding_tab:
    render_understanding_tab()

with business_metrics_tab:
    render_business_metrics()

with system_metrics_tab:
    render_system_metrics()
