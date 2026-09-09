"""Report Agent -- validation_report.md text + ZIP packaging of the full project."""

import io
import zipfile

from app.schemas import SpecOutput, GeneratedFile, ValidationReport


def build_validation_report_md(spec: SpecOutput, report: ValidationReport) -> str:
    lines = [
        f"# Validation Report -- {spec.app_name}",
        "",
        f"**Overall status:** {report.overall_status}",
        "",
        f"- Files present: {'PASS' if report.files_ok else 'FAIL'}",
        f"- Syntax errors: {len(report.syntax_errors)}",
        f"- Endpoints implemented: {report.endpoints_detected}/{report.endpoints_required}",
        f"- Tests passed: {report.tests_passed}/{report.tests_total}",
        f"- Requirement coverage: {report.requirement_coverage_pct}%",
        f"- Avg. embedding similarity: {report.avg_embedding_similarity}",
        f"- Correction cycles used: {report.correction_cycles_used}",
        "",
    ]
    if report.syntax_errors:
        lines.append("## Syntax errors")
        lines += [f"- {e}" for e in report.syntax_errors]
        lines.append("")
    if report.failed_test_names:
        lines.append("## Failed tests")
        lines += [f"- {t}" for t in report.failed_test_names]
        lines.append("")

    lines.append("## Requirement Traceability")
    for fr in spec.functional_requirements:
        eps = [ep for ep in spec.endpoints if fr.id in ep.fr_ids]
        ep_str = ", ".join(f"{e.method} {e.path}" for e in eps) or "(none)"
        lines.append(f"- **{fr.id}**: {fr.description} -> {ep_str}")

    return "\n".join(lines)


_RUN_LOCALLY_SCRIPT = '''"""
Run this generated project locally: installs dependencies, starts the
server, and opens the live API docs in your browser.

Usage: python run_locally.py
"""
import subprocess
import sys
import time
import webbrowser

def main():
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    print("Starting server at http://127.0.0.1:8000 ...")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--port", "8000"])

    time.sleep(2)
    webbrowser.open("http://127.0.0.1:8000")
    print("Opened http://127.0.0.1:8000 -- press Ctrl+C here to stop the server.")

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()

if __name__ == "__main__":
    main()
'''


def build_test_checklist(spec: SpecOutput) -> str:
    lines = ["# How to Test This App", "", "Run `run_locally.py` (or double-click `start.bat` on Windows) first, then:", ""]
    n = 1
    for entity in spec.entities:
        table = entity.table_name
        if any(ep.entity == entity.name and ep.method == "POST" for ep in spec.endpoints):
            lines.append(f"{n}. Go to the **Data Entry** tab, fill in the **{entity.name}** form, click 'Add {entity.name}' -> A new row should appear in the {entity.name} table.")
            n += 1
        if any(ep.entity == entity.name and ep.method == "GET" and ep.path == f"/{table}" for ep in spec.endpoints):
            lines.append(f"{n}. Look at the **{entity.name}** table -> It should list all {entity.name} records you've added.")
            n += 1
        if any(ep.entity == entity.name and ep.method == "DELETE" for ep in spec.endpoints):
            lines.append(f"{n}. Click 'Delete' next to any {entity.name} row -> That row should disappear from the table.")
            n += 1
    for ep in spec.endpoints:
        if ep.method == "GET" and "{" not in ep.path and not any(ep.path == f"/{e.table_name}" for e in spec.entities):
            lines.append(f"{n}. Check the **Reports** tab -> {ep.description}")
            n += 1
    return "\n".join(lines)


_START_BAT = "@echo off\r\npython run_locally.py\r\npause\r\n"


def package_zip(main_files: list[GeneratedFile], test_file: GeneratedFile, report_md: str, spec_json: str, checklist_md: str = "") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in main_files:
            zf.writestr(f.path, f.content)
        zf.writestr(test_file.path, test_file.content)
        zf.writestr("validation_report.md", report_md)
        zf.writestr("spec.json", spec_json)
        zf.writestr("run_locally.py", _RUN_LOCALLY_SCRIPT)
        zf.writestr("start.bat", _START_BAT)
        if checklist_md:
            zf.writestr("TEST_CHECKLIST.md", checklist_md)
    return buf.getvalue()


def run_report_agent(
    spec: SpecOutput, main_files: list[GeneratedFile], test_file: GeneratedFile, report: ValidationReport
) -> tuple[str, bytes, str]:
    report_md = build_validation_report_md(spec, report)
    checklist_md = build_test_checklist(spec)
    zip_bytes = package_zip(main_files, test_file, report_md, spec.model_dump_json(indent=2), checklist_md)
    return report_md, zip_bytes, checklist_md
