# Conversational Financial Coach – Architecture & Design Log

## Vision

Build a conversational financial coach that helps users understand options, reduce uncertainty, build confidence, and take appropriate next steps.

The coach should combine:

* LLM-based understanding
* structured user state
* dialogue management
* retrieval/tools
* guardrails
* measurable system quality

---

# 1. Core Product Principles

## Coach, Not Questionnaire

The experience should begin with goals, concerns, and context rather than demographic intake.

## Helpfulness First

The coach should help users move toward clarity, confidence, or actionability.

## Responsible Financial Guidance

The coach may educate, explain tradeoffs, and suggest next steps, but should avoid personalized investment, legal, or tax advice.

---

# 2. LLM-Based Conversation Understanding

The LLM should handle semantic interpretation, including:

* intent detection
* topic and goal extraction
* salience assessment
* cross-turn reference resolution
* knowledge-level estimation
* user corrections
* emotional signals

The system should avoid brittle hand-coded rules wherever interpretation depends on natural language nuance.

## Structured Understanding Output

The LLM should return structured fields such as:

* intent
* primary_topic
* parent_topic
* current_goal
* current_fear
* knowledge_level
* knowledge_level_confidence
* persona
* risk_level
* coaching_style
* evidence_summary

---

# 3. User Model and State Management

The user model should not be a simple accumulation of guesses.

It should support controlled updates.

## State Update Policy

* Explicit user statements override inferred values.
* User corrections invalidate stale assumptions.
* Newer explicit evidence should replace older inferred state.
* Uncertain fields should remain unknown rather than forced.

## Update Operations

The LLM should return update operations, not merely a full rewritten profile.

Examples:

* field_updates
* field_invalidations
* unchanged_fields
* update_confidence
* evidence_summary

This allows the state manager to apply updates consistently and visibly.

---

# 4. Dialogue Management

The Dialogue Manager decides what the coach should do next.

It uses:

* latest user message
* user model
* conversation history
* pending questions
* missing information
* guardrail status

## Dialogue Responsibilities

* answer direct questions
* reassure when appropriate
* gather missing information
* avoid repetitive questioning
* recognize short answers to prior questions
* move the conversation forward

## Slot / Pending Question Tracking

The system should track:

* pending_question
* expected_answer_type
* slot_to_fill
* extracted_slots
* missing_information

Example:

Coach:

> How many years until retirement?

User:

> 10

Interpretation:

```text
retirement_timeline_years = 10
```

---

# 5. Retrieval, Tools, and Grounding

The coach should use retrieved knowledge and tools when needed.

## Knowledge Retrieval

Static knowledge may include:

* ETFs
* CDs
* Roth IRAs
* emergency funds
* budgeting
* debt payoff
* college savings
* retirement planning

## Tool Use

Prototype tools may include:

* mock market data
* mock interest rates
* future product/account data

## LLM Role

The LLM should generate responses using:

```text
user message
+ conversation state
+ dialogue plan
+ retrieved knowledge
+ tool results
+ guardrail decision
```

not from unsupported free-form guessing.

---

# 6. Guardrails and Safety

Critical safety and integrity constraints must be enforced outside pure response generation.

The system should detect and handle:

* personalized investment advice
* legal/tax advice
* harmful financial behavior
* insider trading requests
* requests requiring escalation

Accuracy, safety, and integrity are topline quality requirements, not merely UX contributors.

---

# 7. Response Generation

## Target State

LLM-generated responses should replace brittle templates.

Responses should:

* answer the actual user question
* reflect relevant context
* adapt to user knowledge level
* use retrieved knowledge where available
* ask at most one useful follow-up
* stay within safety boundaries
* respond in the selected language

## Fallback Behavior

Fallback logic should remain available only when:

* no API key is configured
* the LLM call fails
* safety logic requires constrained output

Fallback responses should be clearly marked in debug mode.

---

# 8. Multilingual Experience

The prototype supports:

* English
* German
* Spanish

Production multilingual support would require language-specific evaluation for:

* terminology
* tone
* intent detection
* safety
* localization quality
* cultural expectations

---

# 9. Evaluation Framework

## Business Metrics

Business metrics answer:

> Is the product creating value?

Examples:

* funnel progression
* activation rate
* return usage
* DAU / WAU / MAU
* retention proxy
* escalation rate

## System Metrics

System metrics answer:

> Is the AI system behaving correctly and why?

### Critical Quality

* accuracy
* safety
* integrity

### UX Outcomes

* helpfulness
* trust
* confidence building
* adaptability
* efficiency
* satisfaction

### Diagnostic Metrics

* intent accuracy
* reference resolution quality
* slot completion quality
* dialogue policy quality
* retrieval relevance
* groundedness
* context retention
* perceived understanding

Perceived understanding is diagnostic, not a topline UX metric.

---

# 10. Future Architecture Direction

## Near-Term

* stabilize LLM response generation
* improve state update operations
* refine dialogue manager behavior
* improve latency and perceived responsiveness
* instrument thumbs up/down feedback
* improve evaluation dashboards

## Medium-Term

* add richer retrieval
* add real tool integrations
* add autojudges for evaluation
* add human review workflows
* improve multilingual validation

## LangGraph Consideration

The architecture is becoming graph-shaped:

```text
User Message
→ LLM Understanding
→ State Update
→ Dialogue Manager
→ Retrieval / Tools
→ Guardrails
→ LLM Response
→ Evaluation
```

LangGraph may become useful when orchestration requires:

* branching workflows
* repeated tool calls
* persistent state
* evaluation loops
* human-in-the-loop review
* clearer observability

For now, modular Python is sufficient; LangGraph is a logical next-step candidate once the workflow stabilizes.

---

# Current Prototype Assessment

## Strengths

* clear architecture
* LLM-backed response direction
* structured state model
* state consistency under user correction
* dialogue-management focus
* multilingual POC
* business and system metrics separation
* safety and integrity awareness

## Remaining Gaps

* dialogue-manager maturity
* fallback quality
* production-grade evaluation
* cloud secrets and access hardening

The prototype is now best understood as an architecture demonstration for a responsible, adaptive, multilingual financial coaching system.
