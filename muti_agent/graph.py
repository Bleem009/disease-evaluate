from langgraph.graph import StateGraph, END
from muti_agent.state import AgentState
from muti_agent.nodes import (
    planning_node,
    inference_node,
    evaluation_node,
    visualization_node,
    reporting_node
)
from muti_agent.safety_agent_node import safety_agent_node
from muti_agent.feedback_node import feedback_node

workflow = StateGraph(AgentState)

workflow.add_node("planner", planning_node)
workflow.add_node("safety", safety_agent_node)
workflow.add_node("inferencer", inference_node)
workflow.add_node("evaluator", evaluation_node)
workflow.add_node("visualizer", visualization_node)
workflow.add_node("feedback", feedback_node)
workflow.add_node("reporter", reporting_node)

def after_safety(state: AgentState) -> str:
    if state.get("safety_check_passed", False):
        return "inferencer"
    else:
        return "reporter"

def after_feedback(state: AgentState) -> str:
    if state.get("current_step") == "replanning":
        return "planner"
    else:
        return "reporter"

workflow.set_entry_point("planner")
workflow.add_edge("planner", "safety")
workflow.add_conditional_edges("safety", after_safety)
workflow.add_edge("inferencer", "evaluator")
workflow.add_edge("evaluator", "visualizer")
workflow.add_edge("visualizer", "feedback")
workflow.add_conditional_edges("feedback", after_feedback)
workflow.add_edge("reporter", END)

agent_workflow = workflow.compile()