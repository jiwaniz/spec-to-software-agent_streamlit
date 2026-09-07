"""
LangGraph skeleton for the Spec-to-Software Agent pipeline.

Every node below is a STUB — the real logic gets filled in on its
corresponding day (see comments). This file exists so the graph
shape and edges (including the correction loop and the separate
refinement entry point) are settled before you write agent internals.

Run `python -m app.graph_sketch` to confirm the graph compiles and
the linear happy-path runs end-to-end with placeholder data.
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

from app.agents.requirement_agent import run_requirement_agent
from app.agents.specification_agent import run_specification_agent


class PipelineState(TypedDict, total=False):
    # Inputs
    raw_requirement: str
    auth_enabled: bool

    # Populated as the graph runs
    requirement: dict          # RequirementOutput
    spec: dict                 # SpecOutput
    plan: dict                 # PlanOutput
    tasks: dict                # TaskOutput
    retrieved_example: dict    # nearest gold-standard SpecOutput
    generated_files: list      # list[GeneratedFile]
    test_file: dict            # GeneratedFile for tests/test_api.py
    validation: dict           # ValidationReport
    test_failure_details: dict # {test_name: failure_text} for the correction loop
    report_md: str
    zip_bytes: bytes
    checklist_md: str
    diagram: str               # Mermaid syntax
    correction_cycle: int

    # Refinement entry point
    refinement_message: str
    refinement_patch: dict


# ---------------------------------------------------------------------------
# Node stubs — replace bodies with real agent calls on the day noted
# ---------------------------------------------------------------------------

def requirement_node(state: PipelineState) -> PipelineState:
    print("[Requirement Agent] classifying request...")
    requirement = run_requirement_agent(state["raw_requirement"])
    state["requirement"] = requirement.model_dump()
    if not requirement.in_scope:
        print(f"[Requirement Agent] REJECTED: {requirement.rejection_reason}")
    return state


def specification_node(state: PipelineState) -> PipelineState:
    from app.schemas import RequirementOutput  # local import to avoid top-level cycle

    requirement = RequirementOutput.model_validate(state["requirement"])
    if not requirement.in_scope:
        # Nothing to specify — downstream nodes should check for this and short-circuit.
        # Day 3+ will add a proper "rejected" exit edge; for now we just skip work.
        print("[Specification Agent] skipped — requirement was rejected")
        state["spec"] = {}
        return state

    print("[Specification Agent] generating structured spec...")
    spec = run_specification_agent(requirement)
    state["spec"] = spec.model_dump()
    return state


def planning_node(state: PipelineState) -> PipelineState:
    if not state.get("spec"):
        print("[Planning Agent] skipped -- no spec")
        state["plan"] = {}
        return state

    from app.schemas import SpecOutput
    from app.agents.planning_agent import run_planning_agent

    print("[Planning Agent] deriving folder structure and DB schema notes...")
    spec = SpecOutput.model_validate(state["spec"])
    state["plan"] = run_planning_agent(spec).model_dump()
    return state


def task_node(state: PipelineState) -> PipelineState:
    if not state.get("spec"):
        print("[Task Agent] skipped -- no spec")
        state["tasks"] = {}
        return state

    from app.schemas import SpecOutput
    from app.agents.task_agent import run_task_agent

    print("[Task Agent] building dependency-ordered task list...")
    spec = SpecOutput.model_validate(state["spec"])
    state["tasks"] = run_task_agent(spec).model_dump()
    return state


def retrieval_node(state: PipelineState) -> PipelineState:
    if not state.get("spec"):
        print("[Retrieval step] skipped -- no spec to ground against")
        state["retrieved_example"] = {}
        return state

    from app.schemas import SpecOutput
    from app.rag.retrieval import retrieve

    print("[Retrieval step] finding closest gold-standard example...")
    spec = SpecOutput.model_validate(state["spec"])
    matches = retrieve(spec, top_k=1)
    state["retrieved_example"] = matches[0].model_dump() if matches else {}
    return state


def coding_node(state: PipelineState) -> PipelineState:
    if not state.get("spec"):
        print("[Coding Agent] skipped -- no spec to generate from")
        state["generated_files"] = []
        return state

    from app.schemas import SpecOutput
    from app.agents.coding_agent import run_coding_agent

    print("[Coding Agent] rendering templates + generating custom endpoint logic...")
    spec = SpecOutput.model_validate(state["spec"])
    files = run_coding_agent(spec)
    state["generated_files"] = [f.model_dump() for f in files]
    return state


def testing_node(state: PipelineState) -> PipelineState:
    if not state.get("generated_files"):
        print("[Testing Agent] skipped -- no generated files")
        state["test_file"] = {}
        return state

    from app.schemas import SpecOutput
    from app.agents.testing_agent import run_testing_agent

    print("[Testing Agent] generating pytest suite...")
    spec = SpecOutput.model_validate(state["spec"])
    test_file = run_testing_agent(spec)
    state["test_file"] = test_file.model_dump()
    return state


def validation_node(state: PipelineState) -> PipelineState:
    if not state.get("generated_files") or not state.get("test_file"):
        print("[Validation Agent] skipped -- nothing to validate")
        state["validation"] = {"overall_status": "FAIL", "correction_cycles_used": state.get("correction_cycle", 0)}
        return state

    from app.schemas import SpecOutput, GeneratedFile
    from app.agents.validation_agent import run_validation_agent_full

    print("[Validation Agent] running deterministic checks + real pytest...")
    spec = SpecOutput.model_validate(state["spec"])
    main_files = [GeneratedFile.model_validate(f) for f in state["generated_files"]]
    test_file = GeneratedFile.model_validate(state["test_file"])
    report, failure_details = run_validation_agent_full(
        spec, main_files, test_file, correction_cycles_used=state.get("correction_cycle", 0)
    )
    state["validation"] = report.model_dump()
    state["test_failure_details"] = failure_details
    return state


def correction_node(state: PipelineState) -> PipelineState:
    state["correction_cycle"] = state.get("correction_cycle", 0) + 1
    cycle = state["correction_cycle"]

    failed_tests = state.get("validation", {}).get("failed_test_names", [])
    failure_details = state.get("test_failure_details", {})
    if not failed_tests:
        print(f"[Correction Agent] cycle {cycle}: nothing to fix (no failed tests recorded)")
        return state

    from app.schemas import SpecOutput, GeneratedFile
    from app.agents.coding_agent import regenerate_endpoint_with_feedback

    spec = SpecOutput.model_validate(state["spec"])
    main_files = [GeneratedFile.model_validate(f) for f in state["generated_files"]]

    for test_name in failed_tests:
        if not test_name.startswith("test_"):
            continue
        func_name = test_name[len("test_"):]
        detail = failure_details.get(test_name, "(no failure detail captured)")
        print(f"[Correction Agent] cycle {cycle}: regenerating {func_name} using test failure feedback...")
        try:
            main_files = regenerate_endpoint_with_feedback(spec, main_files, func_name, detail)
        except ValueError as e:
            # Not every failing test maps to a custom endpoint (e.g. a
            # standard CRUD test failing would indicate a template bug,
            # not something an endpoint-level correction can fix) --
            # skip it rather than crashing the whole pipeline.
            print(f"[Correction Agent] cycle {cycle}: could not correct {test_name}: {e}")

    state["generated_files"] = [f.model_dump() for f in main_files]
    return state


def diagram_node(state: PipelineState) -> PipelineState:
    if not state.get("spec"):
        state["diagram"] = ""
        return state
    from app.schemas import SpecOutput
    from app.agents.diagram_agent import run_diagram_agent
    print("[Diagram Agent] generating Mermaid ER diagram...")
    state["diagram"] = run_diagram_agent(SpecOutput.model_validate(state["spec"]))
    return state


def report_node(state: PipelineState) -> PipelineState:
    if not state.get("spec") or not state.get("generated_files"):
        state["report_md"] = ""
        state["zip_bytes"] = b""
        state["checklist_md"] = ""
        return state
    from app.schemas import SpecOutput, GeneratedFile, ValidationReport
    from app.agents.report_agent import run_report_agent
    print("[Report Agent] building report + ZIP + test checklist...")
    spec = SpecOutput.model_validate(state["spec"])
    main_files = [GeneratedFile.model_validate(f) for f in state["generated_files"]]
    test_file = GeneratedFile.model_validate(state["test_file"])
    report = ValidationReport.model_validate(state["validation"])
    report_md, zip_bytes, checklist_md = run_report_agent(spec, main_files, test_file, report)
    state["report_md"] = report_md
    state["zip_bytes"] = zip_bytes
    state["checklist_md"] = checklist_md
    return state


def refinement_node(state: PipelineState) -> PipelineState:
    from app.schemas import SpecOutput
    from app.agents.refinement_agent import run_refinement_agent, apply_patch

    spec = SpecOutput.model_validate(state["spec"])
    message = state.get("refinement_message", "")
    print(f"[Refinement Agent] applying: {message!r}")
    patch = run_refinement_agent(spec, message)
    new_spec = apply_patch(spec, patch)
    state["spec"] = new_spec.model_dump()
    state["refinement_patch"] = patch.model_dump()
    state["correction_cycle"] = 0  # fresh validation budget for the refined spec
    return state


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_validation(state: PipelineState) -> Literal["diagram", "correction"]:
    # A rejected requirement never had a spec to begin with -- there's
    # nothing for the correction loop to fix, so skip it entirely rather
    # than burning 2 pointless cycles before giving up.
    requirement = state.get("requirement", {})
    if not requirement.get("in_scope", False):
        return "diagram"

    validation = state.get("validation", {})
    cycles_used = state.get("correction_cycle", 0)
    if validation.get("overall_status") == "PASS" or cycles_used >= 2:
        return "diagram"
    return "correction"


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("requirement", requirement_node)
    graph.add_node("specification", specification_node)
    graph.add_node("planning", planning_node)
    graph.add_node("task", task_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("coding", coding_node)
    graph.add_node("testing", testing_node)
    graph.add_node("validation", validation_node)
    graph.add_node("correction", correction_node)
    graph.add_node("diagram", diagram_node)
    graph.add_node("report", report_node)
    graph.add_node("refinement", refinement_node)

    graph.set_entry_point("requirement")
    graph.add_edge("requirement", "specification")
    graph.add_edge("specification", "planning")
    graph.add_edge("planning", "task")
    graph.add_edge("task", "retrieval")
    graph.add_edge("retrieval", "coding")
    graph.add_edge("coding", "testing")
    graph.add_edge("testing", "validation")

    graph.add_conditional_edges(
        "validation",
        route_after_validation,
        {"diagram": "diagram", "correction": "correction"},
    )
    graph.add_edge("correction", "validation")  # re-check with the SAME test suite, capped by route_after_validation
    graph.add_edge("diagram", "report")
    graph.add_edge("report", END)

    # Refinement is a SEPARATE entry point, not part of the linear happy path.
    # On Day 7, wire the Gradio chat handler to invoke the graph starting at
    # "refinement" with the existing state, then let it flow:
    # refinement -> planning -> task -> retrieval -> coding -> testing
    #            -> validation -> (diagram|correction) -> diagram -> report
    graph.add_edge("refinement", "planning")

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    print("Graph compiled successfully. Running happy-path smoke test...\n")
    final_state = app.invoke({"raw_requirement": "Build an inventory API", "auth_enabled": False})
    print("\nFinal state keys:", list(final_state.keys()))
