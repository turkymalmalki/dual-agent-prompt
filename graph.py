"""
graph.py
--------
Dual-Agent Prompt Engineering System built with LangGraph.

Nodes
-----
  engineer  – Prompt Engineer: drives the conversation toward a 7-layer prompt.
  expert    – Domain Expert: answers questions and provides domain knowledge.
  extract   – Extraction: uses structured output to pull the 7 layers into JSON.

Flow
----
  engineer → expert → engineer → expert → engineer → extract
  (MAX_REVISIONS = 3 rounds; each round = one engineer + one expert turn)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_REVISIONS: int = 3

SEVEN_LAYERS = [
    "Role",
    "Context",
    "Constraints",
    "Structure",
    "Examples",
    "Vocabulary",
    "Quality_Metrics",
]

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(BaseModel):
    """Shared state passed between every node in the graph."""

    task: str = Field(description="The user's original task / goal.")
    domain: str = Field(description="The expert domain (e.g. 'Medical', 'Legal').")
    use_rag: bool = Field(default=False, description="Whether RAG context is provided.")
    rag_context: Optional[str] = Field(
        default=None, description="Extracted text from the uploaded document."
    )
    messages: List[BaseMessage] = Field(
        default_factory=list,
        description="Full conversation history between Engineer and Expert.",
    )
    revision_count: int = Field(
        default=0,
        description="Number of completed Engineer→Expert rounds.",
    )
    final_prompt: Optional[Dict[str, str]] = Field(
        default=None,
        description="The extracted 7-layer prompt as a JSON dict.",
    )
    last_speaker: str = Field(
        default="",
        description="Tracks which agent spoke last ('engineer' | 'expert').",
    )

    class Config:
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# Pydantic model for structured extraction
# ---------------------------------------------------------------------------


class SevenLayerPrompt(BaseModel):
    """The seven essential layers of a world-class prompt."""

    Role: str = Field(description="The persona or role the model should adopt.")
    Context: str = Field(
        description="Background information, goals, and situational framing."
    )
    Constraints: str = Field(
        description="Rules, limitations, and boundaries the model must respect."
    )
    Structure: str = Field(
        description="Expected output format, length, and organisation."
    )
    Examples: str = Field(
        description="Concrete input/output examples that demonstrate expectations."
    )
    Vocabulary: str = Field(
        description="Preferred terminology, tone, and language style."
    )
    Quality_Metrics: str = Field(
        description="Criteria to evaluate whether the output meets the goal."
    )


# ---------------------------------------------------------------------------
# LLM factories
# ---------------------------------------------------------------------------


def _get_llm(temperature: float = 0.7) -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file or "
            "Streamlit Cloud secrets."
        )
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        api_key=api_key,
    )


# ---------------------------------------------------------------------------
# Round directive helper
# ---------------------------------------------------------------------------

_ROUND_DIRECTIVES: Dict[int, str] = {
    1: (
        "ROUND 1 – BASICS: Ask the Domain Expert up to 3 foundational questions "
        "about the task. Focus on core objectives, primary audience, and key "
        "deliverables. Do NOT attempt to write the full prompt yet."
    ),
    2: (
        "ROUND 2 – DEEP DIVE: Ask the Domain Expert up to 3 advanced questions. "
        "Probe edge cases, constraints, vocabulary, and quality standards. "
        "Build on what you learned in Round 1."
    ),
    3: (
        "ROUND 3 – FINAL DRAFT: Do NOT ask any more questions. "
        "Using everything you have learned, produce the complete 7-layer prompt "
        "in the exact format below — nothing else:\n\n"
        "**Role:** <text>\n"
        "**Context:** <text>\n"
        "**Constraints:** <text>\n"
        "**Structure:** <text>\n"
        "**Examples:** <text>\n"
        "**Vocabulary:** <text>\n"
        "**Quality_Metrics:** <text>"
    ),
}


# ---------------------------------------------------------------------------
# Node: Engineer
# ---------------------------------------------------------------------------


def engineer_node(state: AgentState) -> AgentState:
    """
    The Prompt Engineer. Drives the conversation toward a 7-layer prompt.
    On rounds 1-2 it asks clarifying questions; on round 3 it outputs the draft.
    """
    llm = _get_llm(temperature=0.6)

    # Advance the round counter (engineer always acts first inside each round)
    current_round = state.revision_count + 1

    system_prompt = (
        "You are an elite Prompt Engineer. Your mission is to craft a world-class, "
        "7-layer prompt for the task described by the user. "
        "You will collaborate with a Domain Expert over three structured rounds.\n\n"
        f"TASK: {state.task}\n"
        f"DOMAIN: {state.domain}\n"
    )

    if state.use_rag and state.rag_context:
        system_prompt += (
            "\nREFERENCE DOCUMENT (use as grounding knowledge):\n"
            f"{state.rag_context[:4000]}\n"  # truncate to avoid token overrun
        )

    # Inject the round-specific directive as the last HumanMessage so the LLM
    # treats it as an explicit instruction for this turn.
    round_directive = _ROUND_DIRECTIVES.get(current_round, _ROUND_DIRECTIVES[3])
    directive_message = HumanMessage(
        content=f"[SYSTEM DIRECTIVE – ROUND {current_round}] {round_directive}"
    )

    messages_for_llm: List[BaseMessage] = (
        [SystemMessage(content=system_prompt)]
        + list(state.messages)
        + [directive_message]
    )

    response: AIMessage = llm.invoke(messages_for_llm)
    response.name = "Engineer"

    updated_messages = list(state.messages) + [response]

    return state.model_copy(
        update={
            "messages": updated_messages,
            "revision_count": current_round,
            "last_speaker": "engineer",
        }
    )


# ---------------------------------------------------------------------------
# Node: Expert
# ---------------------------------------------------------------------------


def expert_node(state: AgentState) -> AgentState:
    """
    The Domain Expert. Answers the Engineer's questions with domain knowledge
    and, if available, grounding from the RAG context.
    """
    llm = _get_llm(temperature=0.5)

    system_prompt = (
        f"You are a seasoned {state.domain} expert. "
        "A Prompt Engineer is asking you questions to help craft the perfect prompt "
        f"for the following task: {state.task}\n\n"
        "Answer concisely yet thoroughly. Provide concrete specifics, not generalities. "
        "If you are shown a reference document, ground your answers in it."
    )

    if state.use_rag and state.rag_context:
        system_prompt += (
            "\n\nREFERENCE DOCUMENT:\n"
            f"{state.rag_context[:4000]}"
        )

    messages_for_llm: List[BaseMessage] = [
        SystemMessage(content=system_prompt)
    ] + list(state.messages)

    response: AIMessage = llm.invoke(messages_for_llm)
    response.name = "Expert"

    updated_messages = list(state.messages) + [response]

    return state.model_copy(
        update={
            "messages": updated_messages,
            "last_speaker": "expert",
        }
    )


# ---------------------------------------------------------------------------
# Node: Extraction
# ---------------------------------------------------------------------------


def extraction_node(state: AgentState) -> AgentState:
    """
    Uses structured output (with_structured_output) to extract the 7-layer
    prompt from the Engineer's final message into a typed JSON dict.
    """
    llm = _get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(SevenLayerPrompt)

    # Find the engineer's final message (last message with name="Engineer")
    engineer_messages = [
        m for m in state.messages if getattr(m, "name", "") == "Engineer"
    ]
    final_draft = engineer_messages[-1].content if engineer_messages else ""

    extraction_prompt = (
        "You are a precision extraction assistant. "
        "Given the following 7-layer prompt draft, extract each layer verbatim "
        "(or faithfully paraphrased if the labels differ) into the structured schema.\n\n"
        f"DRAFT:\n{final_draft}"
    )

    result: SevenLayerPrompt = structured_llm.invoke(extraction_prompt)
    extracted: Dict[str, str] = result.model_dump()

    return state.model_copy(update={"final_prompt": extracted})


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def router(state: AgentState) -> str:
    """
    Decides the next node:
    - After Engineer speaks on rounds 1-2  → route to Expert.
    - After Expert speaks (round < MAX)    → route back to Engineer.
    - After Engineer speaks on round 3     → route to Extraction.
    - After Expert speaks on round MAX     → route to Engineer (for final draft).
    """
    if state.last_speaker == "engineer":
        if state.revision_count >= MAX_REVISIONS:
            # Engineer has produced the final draft → extract
            return "extract"
        else:
            return "expert"
    else:  # last_speaker == "expert"
        return "engineer"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("engineer", engineer_node)
    graph.add_node("expert", expert_node)
    graph.add_node("extract", extraction_node)

    graph.set_entry_point("engineer")

    graph.add_conditional_edges(
        "engineer",
        router,
        {
            "expert": "expert",
            "extract": "extract",
        },
    )

    graph.add_conditional_edges(
        "expert",
        router,
        {
            "engineer": "engineer",
            "extract": "extract",
        },
    )

    graph.add_edge("extract", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pipeline(
    task: str,
    domain: str,
    rag_context: Optional[str] = None,
) -> AgentState:
    """
    Entry point called by app.py.

    Parameters
    ----------
    task : str
        The user's goal / task description.
    domain : str
        The expert domain label.
    rag_context : str | None
        Extracted document text for RAG grounding (optional).

    Returns
    -------
    AgentState
        The fully populated state after the pipeline completes, including
        ``messages`` (chat history) and ``final_prompt`` (7-layer JSON dict).
    """
    use_rag = bool(rag_context)

    initial_state = AgentState(
        task=task,
        domain=domain,
        use_rag=use_rag,
        rag_context=rag_context,
        messages=[],
        revision_count=0,
        final_prompt=None,
        last_speaker="",
    )

    compiled_graph = build_graph()
    result: AgentState = compiled_graph.invoke(initial_state)
    return result
