from typing import TypedDict
from langgraph.graph import StateGraph, END

from planner import plan
from coder import code as generate_code
from tester import run_tests
from debugger import debug

MAX_RETRIES = 3

class PipelineState(TypedDict):
    problem: str
    test_code: str
    plan: str
    code: str
    passed: bool
    error: str
    retries: int

def planner_node(state: PipelineState) -> PipelineState:
    state["plan"] = plan(state["problem"])
    return state

def coder_node(state: PipelineState) -> PipelineState:
    state["code"] = generate_code(state["problem"], state["plan"])
    return state

def tester_node(state: PipelineState) -> PipelineState:
    passed, error = run_tests(state["code"], state["test_code"])
    state["passed"] = passed
    state["error"] = error
    return state

def debugger_node(state: PipelineState) -> PipelineState:
    state["code"] = debug(state["problem"], state["code"], state["error"])
    state["retries"] = state.get("retries", 0) + 1
    return state

def route_after_test(state: PipelineState) -> str:
    if state["passed"]:
        return "done"
    if state.get("retries", 0) >= MAX_RETRIES:
        return "done"
    return "debug"

def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("planner", planner_node)
    graph.add_node("coder", coder_node)
    graph.add_node("tester", tester_node)
    graph.add_node("debugger", debugger_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "coder")
    graph.add_edge("coder", "tester")
    graph.add_conditional_edges("tester", route_after_test, {"debug": "debugger", "done": END})
    graph.add_edge("debugger", "tester")

    return graph.compile()