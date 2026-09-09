"""
Streamlit UI -- alternative to ui.py (Gradio). Same backend pipeline
(app/graph_sketch.py), different frontend framework, for deployment on
Streamlit Community Cloud.

Run locally: streamlit run streamlit_app.py
"""

import json
import zipfile
import io

import streamlit as st

from app.graph_sketch import build_graph

st.set_page_config(page_title="Spec-to-Software Agent", layout="wide")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

SUPPORTED_DOMAINS = [
    "Inventory Management", "Expense Tracking", "Leave Management",
    "Student Registration", "Library Management",
]

st.title("Spec-to-Software Agent")
st.caption("Describe a small CRUD app. Supported domains: " + ", ".join(SUPPORTED_DOMAINS))

if st.session_state.get("last_refine_error"):
    st.error(f"Last refinement failed: {st.session_state.last_refine_error}")

from app.rag.example_bank import EXAMPLE_BANK

DOMAIN_TO_SPEC = {spec.domain: spec for spec in EXAMPLE_BANK}
APP_OPTIONS = list(DOMAIN_TO_SPEC.keys()) + ["Custom (describe below)"]

col1, col2 = st.columns([4, 1])
with col1:
    app_choice = st.selectbox("Application type", APP_OPTIONS)
    requirement = ""
    if app_choice == "Custom (describe below)":
        requirement = st.text_input(
            "Describe your app",
            placeholder="Build an inventory API with products, categories, and low-stock alerts.",
        )
with col2:
    auth_enabled = st.checkbox("Add JWT auth")

if st.button("Generate", type="primary"):
    if app_choice == "Custom (describe below)" and not requirement.strip():
        st.warning("Please enter a requirement.")
    else:
        with st.spinner("Generating..."):
            try:
                if app_choice == "Custom (describe below)":
                    result = st.session_state.graph.invoke(
                        {"raw_requirement": requirement, "auth_enabled": auth_enabled}
                    )
                else:
                    # Known domain: use the guaranteed-correct spec directly,
                    # skipping Requirement/Specification LLM calls entirely
                    # for 100% reliability. Use Refine afterward to customize
                    # (add fields/entities like Category, Department, etc.).
                    from app.graph_sketch import (
                        planning_node, task_node, retrieval_node, coding_node,
                        testing_node, validation_node, correction_node,
                        diagram_node, report_node,
                    )
                    base_spec = DOMAIN_TO_SPEC[app_choice].model_copy(deep=True)
                    base_spec.auth_enabled = auth_enabled
                    if auth_enabled:
                        for ep in base_spec.endpoints:
                            if ep.method in ("POST", "PUT", "DELETE"):
                                ep.protected = True
                    s = {
                        "spec": base_spec.model_dump(),
                        "requirement": {"in_scope": True, "app_name": base_spec.app_name, "domain": base_spec.domain},
                        "correction_cycle": 0,
                    }
                    s = planning_node(s); s = task_node(s); s = retrieval_node(s)
                    s = coding_node(s); s = testing_node(s); s = validation_node(s)
                    cycles = 0
                    while s.get("validation", {}).get("overall_status") != "PASS" and cycles < 2:
                        s = correction_node(s); s = validation_node(s); cycles += 1
                    s = diagram_node(s); s = report_node(s)
                    result = s
                st.session_state.result = result
                st.session_state.chat_history = []
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.session_state.result = None

result = st.session_state.result

if result is not None:
    if not result.get("requirement", {}).get("in_scope", False):
        reason = result.get("requirement", {}).get("rejection_reason", "Out of scope.")
        st.error(f"Rejected: {reason}")
    else:
        tabs = st.tabs(["Spec", "Plan", "Code", "Tests", "Validation + Diagram"])

        with tabs[0]:
            st.json(result["spec"])
        with tabs[1]:
            st.json(result["plan"])
        with tabs[2]:
            for f in result["generated_files"]:
                with st.expander(f["path"]):
                    st.code(f["content"], language="python")
        with tabs[3]:
            st.code(result["test_file"]["content"], language="python")
        with tabs[4]:
            st.json(result["validation"])
            st.markdown("### Diagram")
            st.code(result["diagram"], language="text")

        # ZIP download -- reuse the same package_zip() the Gradio UI uses,
        # so run_locally.py and spec.json are always included consistently.
        from app.agents.report_agent import package_zip
        from app.schemas import GeneratedFile
        main_files = [GeneratedFile.model_validate(f) for f in result["generated_files"]]
        test_file = GeneratedFile.model_validate(result["test_file"])
        zip_bytes = package_zip(
            main_files, test_file,
            result.get("report_md", ""),
            json.dumps(result["spec"], indent=2),
            result.get("checklist_md", ""),
        )
        st.download_button(
            "Download project ZIP", zip_bytes,
            file_name=f"{result['spec']['app_name']}.zip",
        )

        with st.expander("✅ How to test this app"):
            st.markdown(result.get("checklist_md", "Run `start.bat` (Windows) or `python run_locally.py`, then explore the tabs."))

        st.markdown("## Refine")
        for msg, reply in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(msg)
            with st.chat_message("assistant"):
                st.write(reply)

        refine_msg = st.chat_input("e.g. 'add an email field to Category'")
        if refine_msg:
            from app.graph_sketch import (
                refinement_node, planning_node, task_node, retrieval_node,
                coding_node, testing_node, validation_node, correction_node,
                diagram_node, report_node,
            )
            with st.spinner("Refining..."):
                try:
                    s = dict(result)
                    s["refinement_message"] = refine_msg
                    s = refinement_node(s)
                    s = planning_node(s); s = task_node(s); s = retrieval_node(s)
                    s = coding_node(s); s = testing_node(s); s = validation_node(s)
                    cycles = 0
                    while s.get("validation", {}).get("overall_status") != "PASS" and cycles < 2:
                        s = correction_node(s); s = validation_node(s); cycles += 1
                    s = diagram_node(s); s = report_node(s)
                    st.session_state.result = s
                    st.session_state.last_refine_error = None
                    reply = s.get("refinement_patch", {}).get("notes", "Updated.")
                    st.session_state.chat_history.append((refine_msg, reply))
                    st.rerun()
                except Exception as e:
                    st.session_state.last_refine_error = str(e)
                    st.session_state.chat_history.append((refine_msg, f"Failed: {e}"))
                    st.rerun()
