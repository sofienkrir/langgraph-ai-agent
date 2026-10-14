"""Build and compile the research graph.

    plan -> search -> coverage -> (search again | synthesize) -> END

The loop back into `search` is driven by `route_after_coverage`, which is plain
Python working on counted sources. No model decides whether the work is done.
"""

from langgraph.graph import END, START, StateGraph

from .coverage import coverage_node, route_after_coverage
from .planner import plan_node
from .search import search_node
from .state import ResearchState
from .synthesis import synthesize_node


def build_research_graph(checkpointer=None):
    """Compile the workflow. No API call happens until the graph is invoked."""
    builder = StateGraph(ResearchState)
    builder.add_node("plan", plan_node)
    builder.add_node("search", search_node)
    builder.add_node("coverage", coverage_node)
    builder.add_node("synthesize", synthesize_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "search")
    builder.add_edge("search", "coverage")
    builder.add_conditional_edges(
        "coverage",
        route_after_coverage,
        {"search": "search", "synthesize": "synthesize"},
    )
    builder.add_edge("synthesize", END)
    return builder.compile(checkpointer=checkpointer)
