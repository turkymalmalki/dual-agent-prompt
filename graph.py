"""
graph.py — Dual-Agent Prompt Engineering System (Claude 3.5 Sonnet Edition)
"""
from __future__ import annotations
import json
import logging
from typing import Annotated, Any, Literal, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_REVISIONS = 3

class SevenLayerPrompt(BaseModel):
    Role: str = Field(description="The precise persona and professional identity.")
    Context: str = Field(description="Background information and situational framing.")
    Constraints: str = Field(description="Explicit boundaries and prohibited behaviors.")
    Structure: str = Field(description="The required output format and structure.")
    Examples: str = Field(description="Concrete few-shot examples.")
    Vocabulary: str = Field(description="Domain-specific terminology and linguistic register.")
    Quality_Metrics: str = Field(description="Explicit success criteria and self-evaluation.")

class AgentState(TypedDict):
    task: str
    domain: str
    use_rag: bool
    rag_context: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    revision_count: int
    final_prompt: dict[str, Any]
    last_speaker: str

ENGINEER_SYSTEM_PROMPT = """\
You are an elite Prompt Engineer. Build a 7-layer prompt by interviewing an Expert.
Round 1: Ask foundational questions.
Round 2: Ask deep dive/edge-case questions.
Round 3: Draft the final 7-layer prompt based on answers. NO MORE QUESTIONS.
Do not hallucinate domain knowledge.
"""

def build_expert_system_prompt(domain: str, rag_context: str) -> str:
    base = f"You are a world-class domain Expert in: {domain}. Answer the Engineer's questions precisely."
    if rag_context:
        base += f"\n\nReference Material:\n---\n{rag_context[:6000]}\n---"
    return base

def get_llm(temperature: float = 0.7) -> ChatAnthropic:
    return ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=temperature)

def get_extraction_llm() -> ChatAnthropic:
    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.0)
    return llm.with_structured_output(SevenLayerPrompt)

def _build_engineer_round_directive(revision: int, task: str, domain: str) -> HumanMessage:
    current_round = revision + 1
    if revision == 0:
        return HumanMessage(content=f"ROUND 1 of {MAX_REVISIONS} — BASICS\nTask: {task}\nDomain: {domain}\nAsk 3-4 foundational questions.")
    elif revision == 1:
        return HumanMessage(content=f"ROUND 2 of {MAX_REVISIONS} — DEEP DIVE\nAsk 3-4 targeted follow-up questions.")
    else:
        return HumanMessage(content=f"ROUND 3 of {MAX_REVISIONS} — FINAL DRAFT ⚠️ MANDATORY\nDO NOT ASK QUESTIONS. Write the complete 7-layer prompt using the 7 specific headers.")

def engineer_node(state: AgentState) -> dict:
    revision = state["revision_count"]
    llm = get_llm(temperature=0.6)
    system = SystemMessage(content=ENGINEER_SYSTEM_PROMPT)
    round_directive = _build_engineer_round_directive(revision, state["task"], state["domain"])
    input_messages = [system] + list(state["messages"]) + [round_directive]
    response = llm.invoke(input_messages)
    return {"messages": [AIMessage(content=response.content, name="Engineer")], "last_speaker": "engineer"}

def expert_node(state: AgentState) -> dict:
    llm = get_llm(temperature=0.5)
    system = SystemMessage(content=build_expert_system_prompt(state["domain"], state["rag_context"]))
    input_messages = [system] + list(state["messages"])
    response = llm.invoke(input_messages)
    return {"messages": [AIMessage(content=response.content, name="Expert")], "last_speaker": "expert", "revision_count": state["revision_count"] + 1}

def extraction_node(state: AgentState) -> dict:
    extraction_llm = get_extraction_llm()
    engineer_messages = [m for m in state["messages"] if isinstance(m, AIMessage) and m.name == "Engineer"]
    final_draft = engineer_messages[-1].content
    prompt = [
        SystemMessage(content="Extract the 7-layer prompt faithfully into the JSON schema."),
        HumanMessage(content=f"Draft:\n{final_draft}")
    ]
    structured_output = extraction_llm.invoke(prompt)
    return {"final_prompt": structured_output.model_dump(), "last_speaker": "done"}

def route_after_expert(state: AgentState) -> Literal["engineer", "extract"]:
    return "extract" if state["revision_count"] >= MAX_REVISIONS else "engineer"

def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("engineer", engineer_node)
    builder.add_node("expert", expert_node)
    builder.add_node("extract", extraction_node)
    builder.add_edge(START, "engineer")
    builder.add_edge("engineer", "expert")
    builder.add_conditional_edges("expert", route_after_expert, {"engineer": "engineer", "extract": "extract"})
    builder.add_edge("extract", END)
    return builder.compile()

def run_pipeline(task: str, domain: str, use_rag: bool = False, rag_context: str = "") -> tuple[dict[str, Any], list[BaseMessage]]:
    graph = build_graph()
    initial_state = {"task": task, "domain": domain, "use_rag": use_rag, "rag_context": rag_context, "messages": [], "revision_count": 0, "final_prompt": {}, "last_speaker": ""}
    final_state = graph.invoke(initial_state, {"recursion_limit": 20})
    return final_state["final_prompt"], list(final_state["messages"])
