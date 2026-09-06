"""
Planning Agent -- Day 3.

Deliberately deterministic, not LLM-based. The project's file structure
is fixed by design (see the guide's "don't let the LLM invent
architecture" decision) and the spec already contains everything needed
to describe the DB schema and endpoint list -- there's nothing here that
benefits from an extra Groq call, and skipping it saves time and avoids
a whole class of failure modes.
"""

from app.schemas import SpecOutput, PlanOutput


def run_planning_agent(spec: SpecOutput) -> PlanOutput:
    folder_structure = [
        "main.py",
        "database.py",
        "models.py",
        "schemas.py",
        "services.py",
        "frontend.html",
        "requirements.txt",
        "README.md",
        "tests/test_api.py",
    ]
    if spec.auth_enabled:
        folder_structure.insert(4, "security.py")

    schema_lines = []
    for entity in spec.entities:
        field_summary = ", ".join(f"{f.name}: {f.type}" for f in entity.fields)
        schema_lines.append(f"{entity.name} ({entity.table_name}) -- {field_summary}")
    db_schema_notes = "\n".join(schema_lines)

    endpoint_summary = [
        f"{ep.method} {ep.path} -- {ep.description}" for ep in spec.endpoints
    ]

    return PlanOutput(
        folder_structure=folder_structure,
        db_schema_notes=db_schema_notes,
        endpoint_summary=endpoint_summary,
    )


if __name__ == "__main__":
    from app.rag.example_bank import EXAMPLE_BANK

    plan = run_planning_agent(EXAMPLE_BANK[0])
    print(plan.model_dump_json(indent=2))
