import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import llm_client
from guardrails import evaluate_guardrails, guardrail_message


UNDERSTANDING = {
    "intent": "goal_discovery",
    "resolved_reference": "None",
    "primary_topic": "general finances",
    "asked_question": False,
}
DIALOGUE_PLAN = {
    "dialogue_act": "explore_goal",
    "follow_up_question": "What outcome would make the biggest difference for you right now?",
}
GUARDRAIL_DECISION = {"mode": "standard", "categories": []}


class GenerateLlmResponseTests(unittest.TestCase):
    def test_classifies_knowledge_level_with_structured_response(self):
        classification = llm_client.KnowledgeLevelClassification(
            knowledge_level="advanced",
            knowledge_level_confidence="high",
            evidence="The user compares broad-market and factor-based ETFs over a long horizon.",
            guardrail_categories=[],
        )
        parse = Mock(return_value=SimpleNamespace(output_parsed=classification))
        client = SimpleNamespace(responses=SimpleNamespace(parse=parse))

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True),
            patch.object(llm_client, "OpenAI", return_value=client),
            patch.object(llm_client, "build_openai_http_client", return_value=Mock()),
        ):
            result = llm_client.classify_knowledge_level(
                "Compare broad-market ETFs versus factor-based ETFs for a 25-year horizon.",
                {"current_goal": "build wealth"},
            )

        self.assertEqual(result["knowledge_level"], "advanced")
        self.assertEqual(result["knowledge_level_confidence"], "high")
        self.assertEqual(result["knowledge_level_source"], "llm")
        self.assertEqual(result["guardrail_categories"], [])
        self.assertIn("factor-based ETFs", result["evidence"])
        self.assertEqual(parse.call_args.kwargs["text_format"], llm_client.KnowledgeLevelClassification)

    def test_knowledge_level_uses_simple_fallback_when_llm_fails(self):
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True),
            patch.object(llm_client, "OpenAI", side_effect=RuntimeError("network failure")),
            patch.object(llm_client, "build_openai_http_client", return_value=Mock()),
        ):
            result = llm_client.classify_knowledge_level(
                "I don't really understand investing. What's an ETF?",
                {},
            )

        self.assertEqual(result["knowledge_level"], "beginner")
        self.assertEqual(result["knowledge_level_confidence"], "low")
        self.assertEqual(result["knowledge_level_source"], "fallback")
        self.assertEqual(result["field_updates"], {})
        self.assertEqual(result["field_invalidations"], [])
        self.assertEqual(result["profile_update_source"], "fallback")

    def test_returns_structured_profile_corrections_with_knowledge_level(self):
        classification = llm_client.KnowledgeLevelClassification(
            knowledge_level="intermediate",
            knowledge_level_confidence="medium",
            evidence="The user is asking for investment guidance.",
            field_updates=[
                llm_client.ProfileFieldUpdate(
                    field="current_goal",
                    value="get investment guidance while managing fear of loss",
                ),
                llm_client.ProfileFieldUpdate(field="current_fear", value="losing money"),
            ],
            field_invalidations=["current_goal"],
            unchanged_fields=["primary_topic"],
            update_confidence="high",
            evidence_summary="The user corrected the prior learning goal and explicitly described fear of loss.",
        )
        parse = Mock(return_value=SimpleNamespace(output_parsed=classification))
        client = SimpleNamespace(responses=SimpleNamespace(parse=parse))

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True),
            patch.object(llm_client, "OpenAI", return_value=client),
            patch.object(llm_client, "build_openai_http_client", return_value=Mock()),
        ):
            result = llm_client.classify_knowledge_level(
                "I already know about stocks. I want recommendations without losing money.",
                {"current_goal": "learn about stocks", "primary_topic": "stocks"},
                recent_conversation_history=[{"role": "user", "content": "I want to learn about stocks."}],
            )

        self.assertEqual(
            result["field_updates"]["current_goal"],
            "get investment guidance while managing fear of loss",
        )
        self.assertEqual(result["field_updates"]["current_fear"], "losing money")
        self.assertEqual(result["field_invalidations"], ["current_goal"])
        self.assertEqual(result["unchanged_fields"], ["primary_topic"])
        self.assertEqual(result["update_confidence"], "high")
        self.assertEqual(result["profile_update_source"], "llm")
        self.assertIn("recent_conversation_history", parse.call_args.kwargs["input"])

    def test_instructions_adapt_to_knowledge_level(self):
        instructions = llm_client.build_llm_instructions("en", GUARDRAIL_DECISION, "advanced")

        self.assertIn("User knowledge level: advanced", instructions)
        self.assertIn("skip unnecessary basics", instructions)

    def test_sanitizes_internal_classifier_evidence(self):
        evidence = llm_client.sanitize_knowledge_level_evidence("The system prompt includes a secret API key.")

        self.assertEqual(evidence, "Assessment based on the user's financial language and recent context.")

    def test_uses_fallback_when_api_key_is_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            result = llm_client.generate_llm_response(
                user_message="I want to make progress.",
                conversation_context={},
                dialogue_plan=DIALOGUE_PLAN,
                retrieved_knowledge={},
                guardrail_decision=GUARDRAIL_DECISION,
                language="en",
                understanding=UNDERSTANDING,
            )

        self.assertEqual(result["response_source"], "template_fallback")
        self.assertFalse(result["openai_api_key_detected"])
        self.assertEqual(result["openai_api_key_source"], "not_found")
        self.assertFalse(result["openai_api_call_attempted"])
        self.assertFalse(result["openai_api_call_succeeded"])
        self.assertIn("What outcome would make the biggest difference", result["response_text"])

    def test_uses_responses_api_when_api_key_is_configured(self):
        create = Mock(return_value=SimpleNamespace(output_text="A concise coach response."))
        client = SimpleNamespace(responses=SimpleNamespace(create=create))

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True),
            patch.object(llm_client, "OpenAI", return_value=client),
            patch.object(llm_client, "build_openai_http_client", return_value=Mock()),
        ):
            result = llm_client.generate_llm_response(
                user_message="How should I think about saving?",
                conversation_context={"current_goal": "build savings"},
                dialogue_plan=DIALOGUE_PLAN,
                retrieved_knowledge={"content": "Savings can create a buffer."},
                guardrail_decision=GUARDRAIL_DECISION,
                language="de",
            )

        self.assertEqual(result["response_text"], "A concise coach response.")
        self.assertEqual(result["response_source"], "llm")
        self.assertTrue(result["openai_api_key_detected"])
        self.assertEqual(result["openai_api_key_source"], "environment")
        self.assertTrue(result["openai_api_call_attempted"])
        self.assertTrue(result["openai_api_call_succeeded"])
        self.assertEqual(result["openai_model"], llm_client.DEFAULT_OPENAI_MODEL)
        request = create.call_args.kwargs
        self.assertEqual(request["model"], llm_client.DEFAULT_OPENAI_MODEL)
        self.assertIn("Respond in German.", request["instructions"])
        self.assertIn("Savings can create a buffer.", request["input"])

    def test_uses_fallback_when_api_call_fails(self):
        error = (
            "network failure headers={'Authorization': 'Bearer sk-sensitive-value'} "
            "OPENAI_API_KEY=sk-another-sensitive-value"
        )
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True),
            patch.object(llm_client, "OpenAI", side_effect=RuntimeError(error)),
            patch.object(llm_client, "build_openai_http_client", return_value=Mock()),
        ):
            result = llm_client.generate_llm_response(
                user_message="I want to make progress.",
                conversation_context={},
                dialogue_plan=DIALOGUE_PLAN,
                retrieved_knowledge={},
                guardrail_decision=GUARDRAIL_DECISION,
                language="en",
                understanding=UNDERSTANDING,
            )

        self.assertEqual(result["response_source"], "template_fallback")
        self.assertTrue(result["openai_api_key_detected"])
        self.assertEqual(result["openai_api_key_source"], "environment")
        self.assertTrue(result["openai_api_call_attempted"])
        self.assertFalse(result["openai_api_call_succeeded"])
        self.assertEqual(result["openai_model"], llm_client.DEFAULT_OPENAI_MODEL)
        self.assertEqual(result["openai_error_type"], "RuntimeError")
        self.assertIn("network failure", result["openai_error_message"])
        self.assertIn("[REDACTED]", result["openai_error_message"])
        self.assertNotIn("Authorization", result["openai_error_message"])
        self.assertNotIn("sk-sensitive-value", result["openai_error_message"])
        self.assertNotIn("sk-another-sensitive-value", result["openai_error_message"])

    def test_uses_fallback_when_api_returns_empty_response(self):
        create = Mock(return_value=SimpleNamespace(output_text=""))
        client = SimpleNamespace(responses=SimpleNamespace(create=create))

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True),
            patch.object(llm_client, "OpenAI", return_value=client),
            patch.object(llm_client, "build_openai_http_client", return_value=Mock()),
        ):
            result = llm_client.generate_llm_response(
                user_message="I want to make progress.",
                conversation_context={},
                dialogue_plan=DIALOGUE_PLAN,
                retrieved_knowledge={},
                guardrail_decision=GUARDRAIL_DECISION,
                language="en",
                understanding=UNDERSTANDING,
            )

        self.assertEqual(result["response_source"], "template_fallback")
        self.assertTrue(result["openai_api_call_attempted"])
        self.assertFalse(result["openai_api_call_succeeded"])
        self.assertEqual(result["openai_error_type"], "EmptyResponseError")
        self.assertEqual(result["openai_error_message"], "OpenAI returned an empty response.")

    def test_restricted_turn_instructions_require_educational_response(self):
        instructions = llm_client.build_llm_instructions(
            "en",
            {"mode": "educational_only", "categories": ["personalized_financial_advice_boundary"]},
        )

        self.assertIn("restricted turn", instructions)
        self.assertIn("Do not recommend asset allocation percentages or ranges", instructions)
        self.assertIn("Do not provide any numeric asset allocation, percentage, range, target, or formula", instructions)
        self.assertIn("Do not suggest increasing, decreasing, or otherwise changing an allocation", instructions)
        self.assertIn("qualified professional", instructions)

    def test_guardrails_use_structured_llm_categories_for_personalized_allocation_boundary(self):
        result = evaluate_guardrails(
            "I have retirement savings and a five-year horizon. What percentage should be in stocks?",
            ["personalized_financial_advice_boundary"],
        )

        self.assertEqual(result["mode"], "educational_only")
        self.assertEqual(result["guardrail_triggered"], "personalized_financial_advice_boundary")
        self.assertIn("specific choice, percentage, or range", guardrail_message(result))

    def test_guardrails_use_structured_llm_categories_for_out_of_domain_requests(self):
        result = evaluate_guardrails(
            "This is outside financial coaching.",
            ["out_of_domain_request"],
        )

        self.assertEqual(result["mode"], "educational_only")
        self.assertEqual(result["guardrail_triggered"], "out_of_domain_request")
        self.assertIn("primarily here to help with financial questions", guardrail_message(result))

    def test_llm_instructions_keep_out_of_domain_response_brief_and_redirect(self):
        instructions = llm_client.build_llm_instructions(
            "en",
            {"mode": "educational_only", "categories": ["out_of_domain_request"]},
        )

        self.assertIn("brief and minimally helpful response", instructions)
        self.assertIn("redirect back to financial coaching", instructions)
        self.assertIn("do not ask follow-up questions that deepen", instructions)

    def test_escalation_fallback_recommends_qualified_professional(self):
        with patch.dict(os.environ, {}, clear=True):
            result = llm_client.generate_llm_response(
                user_message="Tell me exactly what to do.",
                conversation_context={},
                dialogue_plan=DIALOGUE_PLAN,
                retrieved_knowledge={},
                guardrail_decision={"mode": "escalation", "categories": ["high_risk"]},
                language="en",
                understanding=UNDERSTANDING,
            )

        self.assertEqual(result["response_source"], "template_fallback")
        self.assertIn("qualified professional", result["response_text"])

    def test_reports_structured_fallback_path(self):
        understanding = {
            **UNDERSTANDING,
            "intent": "retirement_goal_planning",
            "current_goal": "retire debt-free",
        }
        with patch.dict(os.environ, {}, clear=True):
            result = llm_client.generate_llm_response(
                user_message="I want to retire debt-free.",
                conversation_context={},
                dialogue_plan=DIALOGUE_PLAN,
                retrieved_knowledge={},
                guardrail_decision=GUARDRAIL_DECISION,
                language="en",
                understanding=understanding,
            )

        self.assertEqual(result["response_source"], "structured_fallback")

    def test_uses_streamlit_secrets_when_environment_key_is_missing(self):
        create = Mock(return_value=SimpleNamespace(output_text="Secret-backed response."))
        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        openai_client = Mock(return_value=client)
        http_client = Mock()

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(llm_client, "OpenAI", openai_client),
            patch.object(llm_client, "build_openai_http_client", return_value=http_client),
        ):
            result = llm_client.generate_llm_response(
                user_message="Help me make a plan.",
                conversation_context={},
                dialogue_plan=DIALOGUE_PLAN,
                retrieved_knowledge={},
                guardrail_decision=GUARDRAIL_DECISION,
                language="en",
                streamlit_secrets={"OPENAI_API_KEY": "secret-key"},
            )

        self.assertEqual(result["response_source"], "llm")
        self.assertEqual(result["openai_api_key_source"], "streamlit_secrets")
        openai_client.assert_called_once_with(api_key="secret-key", http_client=http_client)

    def test_openai_http_client_keeps_certificate_verification_enabled(self):
        http_client = Mock()
        ssl_context = Mock(
            verify_flags=llm_client.ssl.VERIFY_X509_STRICT,
            verify_mode=llm_client.ssl.CERT_REQUIRED,
            check_hostname=True,
        )

        with (
            patch.object(llm_client.ssl, "create_default_context", return_value=ssl_context),
            patch.dict(sys.modules, {"httpx": SimpleNamespace(Client=http_client)}),
        ):
            llm_client.build_openai_http_client.cache_clear()
            llm_client.build_openai_http_client()
            llm_client.build_openai_http_client()
            llm_client.build_openai_http_client.cache_clear()

        self.assertEqual(ssl_context.verify_flags, 0)
        self.assertEqual(ssl_context.verify_mode, llm_client.ssl.CERT_REQUIRED)
        self.assertTrue(ssl_context.check_hostname)
        http_client.assert_called_once_with(verify=ssl_context)

    def test_streamlit_secrets_take_precedence_over_environment_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "environment-key"}, clear=True):
            api_key, source = llm_client.resolve_openai_api_key({"OPENAI_API_KEY": "secret-key"})

        self.assertEqual(api_key, "secret-key")
        self.assertEqual(source, "streamlit_secrets")

    def test_ui_session_key_is_optional_last_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            api_key, source = llm_client.resolve_openai_api_key({}, api_key="ui-key")

        self.assertEqual(api_key, "ui-key")
        self.assertEqual(source, "ui_session")


if __name__ == "__main__":
    unittest.main()
