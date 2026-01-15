"""- langgraph wiring
- agent sequence
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph

from .agents import Action, RiskOutput, decision_agent, explanation_agent, risk_agent


class AgentState(TypedDict, total=False):
    scene: dict
    risk: RiskOutput
    action: Action
    explanation: str


def _risk_node(state: AgentState) -> AgentState:
    risk = risk_agent(state["scene"])
    return {"risk": risk}


def _decision_node(state: AgentState) -> AgentState:
    risk: RiskOutput = state["risk"]
    action = decision_agent(state["scene"], risk)
    return {"action": action}


def _explanation_node(state: AgentState) -> AgentState:
    risk: RiskOutput = state["risk"]
    action: Action = state["action"]
    explanation = explanation_agent(action, risk)
    return {"explanation": explanation}


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("risk", _risk_node)
    graph.add_node("decision", _decision_node)
    graph.add_node("explanation", _explanation_node)

    graph.set_entry_point("risk")
    graph.add_edge("risk", "decision")
    graph.add_edge("decision", "explanation")
    graph.add_edge("explanation", END)
    return graph


def run_graph(scene: dict) -> AgentState:
    compiled = build_graph().compile()
    result = compiled.invoke({"scene": scene})
    return result
