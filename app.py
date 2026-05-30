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
from llm_client import generate_response
from mock_data import generate_mock_data
from tools import collect_tool_results

st.set_page_config(page_title="Acorns AI Coach", layout="wide")

st.title("🌱 Acorns AI Coach")
st.caption("Helping everyday people build financial confidence.")

ASSISTANT_AVATAR = "🐿️"
MOCK_DATA_SCHEMA_VERSION = 3

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


@st.cache_data
def load_mock_data(schema_version: int):
    return generate_mock_data()


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
}

REQUIRED_SESSION_COLUMNS = {
    "user_id",
    "persona",
    "topic_category",
    "recommended_action_taken",
    "escalated_to_human",
    "resolved_reference_success",
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
            "Product Metrics cannot render because the mock-data schema is missing required columns. "
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


def filter_mock_data(users: pd.DataFrame, sessions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    st.subheader("Filters")
    first_row = st.columns(4)
    second_row = st.columns(3)

    with first_row[0]:
        personas = multi_filter("Persona", users["persona"], "metrics_persona")
    with first_row[1]:
        user_types = multi_filter("User type", users["user_type"], "metrics_user_type")
    with first_row[2]:
        age_bands = multi_filter("Age band", users["age_band"], "metrics_age_band")
    with first_row[3]:
        literacy_levels = multi_filter("Financial literacy", users["financial_literacy"], "metrics_literacy")
    with second_row[0]:
        coaching_styles = multi_filter("Coaching style", users["coaching_style"], "metrics_coaching_style")
    with second_row[1]:
        risk_levels = multi_filter("Risk level", users["risk_level"], "metrics_risk_level")
    with second_row[2]:
        topic_categories = multi_filter("Topic category", sessions["topic_category"], "metrics_topic_category")

    filtered_users = users[
        users["persona"].isin(personas)
        & users["user_type"].isin(user_types)
        & users["age_band"].isin(age_bands)
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


def render_funnel(users: pd.DataFrame):
    stages = ["started_chat", "shared_goal", "received_guidance", "viewed_recommendation", "activated"]
    stage_rank = {stage: index for index, stage in enumerate(stages)}
    user_stage_rank = users["funnel_stage"].map(stage_rank)
    counts = [(user_stage_rank >= stage_rank[stage]).sum() for stage in stages]
    colors = ["#2f855a", "#3f956a", "#52a67a", "#69b98d", "#82c9a1"]
    rows = []

    for stage, count, color in zip(stages, counts, colors):
        width = max(34, round(count / max(counts) * 100))
        label = stage.replace("_", " ").title()
        rows.append(
            f"""
            <div style="width:{width}%; background:{color}; color:white; margin:0 auto 6px; padding:10px 8px;
                text-align:center; border-radius:4px; font-size:14px; line-height:1.2;">
                <strong>{label}</strong><br><span>{count:,} users</span>
            </div>
            """
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_product_metrics():
    mock_data = load_mock_data(MOCK_DATA_SCHEMA_VERSION)
    users = mock_data["users"]
    sessions = mock_data["sessions"]

    if not validate_dashboard_schema(users, sessions, show_warning=False):
        load_mock_data.clear()
        mock_data = load_mock_data(MOCK_DATA_SCHEMA_VERSION)
        users = mock_data["users"]
        sessions = mock_data["sessions"]
        if not validate_dashboard_schema(users, sessions):
            return

    users, sessions = filter_mock_data(users, sessions)
    if users.empty or sessions.empty:
        st.warning("No mocked sessions match these filters.")
        return

    metrics = st.columns(6)
    metrics[0].metric("Activation Rate", f"{users['activated'].mean():.1%}")
    metrics[1].metric("Accepted Next Step", f"{sessions['recommended_action_taken'].mean():.1%}")
    metrics[2].metric("Satisfaction", f"{users['inferred_satisfaction'].mean():.2f} / 5")
    metrics[3].metric("Confidence Lift", f"+{users['confidence_lift'].mean():.2f}")
    metrics[4].metric("Escalation Rate", f"{sessions['escalated_to_human'].mean():.1%}")
    metrics[5].metric("Reference Success", f"{sessions['resolved_reference_success'].mean():.1%}")
    st.caption(f"{len(users):,} synthetic users and {len(sessions):,} coaching sessions in the filtered view.")

    left, right = st.columns(2)
    with left:
        st.subheader("Persona Distribution")
        horizontal_bar_chart(users["persona"].value_counts().rename_axis("persona").to_frame("users"), "persona", ["users"])
    with right:
        st.subheader("Funnel Progression")
        render_funnel(users)

    left, right = st.columns(2)
    with left:
        st.subheader("Confidence Start vs End")
        horizontal_bar_chart(users.groupby("persona")[["confidence_before", "confidence_after"]].mean().round(2), "persona", ["confidence_before", "confidence_after"])
    with right:
        st.subheader("Activation Rate By Persona")
        horizontal_bar_chart(users.groupby("persona")["activated"].mean().mul(100).round(1).to_frame("activation_rate_pct"), "persona", ["activation_rate_pct"])

    st.subheader("Key Metrics By Persona")
    persona_metrics = users.groupby("persona").agg(
        users=("user_id", "count"),
        activation_rate=("activated", "mean"),
        inferred_satisfaction=("inferred_satisfaction", "mean"),
        confidence_lift=("confidence_lift", "mean"),
    )
    session_metrics = sessions.groupby("persona").agg(
        accepted_next_step_rate=("recommended_action_taken", "mean"),
        escalation_rate=("escalated_to_human", "mean"),
        resolved_reference_success_rate=("resolved_reference_success", "mean"),
    )
    table = persona_metrics.join(session_metrics).reset_index()
    rate_columns = ["activation_rate", "accepted_next_step_rate", "escalation_rate", "resolved_reference_success_rate"]
    table[rate_columns] = table[rate_columns].mul(100).round(1)
    table[["inferred_satisfaction", "confidence_lift"]] = table[["inferred_satisfaction", "confidence_lift"]].round(2)
    st.dataframe(table, hide_index=True, width="stretch")


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

coach_tab, understanding_tab, metrics_tab = st.tabs(["Coach", "Conversational Understanding", "Product Metrics"])

with coach_tab:
    for message in st.session_state.messages:
        avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.write(message["content"])

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
        response = generate_response(
            prompt,
            understanding,
            retrieved_knowledge,
            tool_results,
            st.session_state.conversation_context,
            dialogue_plan,
        )
        update_conversation_context(st.session_state.conversation_context, prompt, understanding)
        record_assistant_response(st.session_state.conversation_context, response)
        record_dialogue_plan(st.session_state.conversation_context, dialogue_plan)
        st.session_state.last_understanding = understanding
        st.session_state.last_dialogue_plan = dialogue_plan
        st.session_state.last_retrieved_knowledge = retrieved_knowledge
        st.session_state.last_tool_results = tool_results
        st.session_state.last_guardrail_result = guardrail_result
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

with understanding_tab:
    render_understanding_tab()

with metrics_tab:
    render_product_metrics()
