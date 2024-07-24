from langgraph.graph import StateGraph, END
from graph.state import AppState
from graph.node_functions import generate_prompt, grade_prompt, finalize_prompt

def create_graph() -> StateGraph:
    graph_builder = StateGraph(AppState)
    
    graph_builder.set_entry_point("generate_prompt")

    graph_builder.add_node("generate_prompt", generate_prompt)
    graph_builder.add_node("grade_prompt", grade_prompt)
    graph_builder.add_node("finalize_prompt", finalize_prompt)

    graph_builder.add_edge("generate_prompt", "grade_prompt")
    graph_builder.add_edge("grade_prompt", "finalize_prompt")
    graph_builder.add_edge("finalize_prompt", END)

    return graph_builder.compile()