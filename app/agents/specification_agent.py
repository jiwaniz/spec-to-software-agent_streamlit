"""
Specification Agent — Day 2.

Takes a validated RequirementOutput and produces a full structured
SpecOutput: entities, fields, endpoints, FR-IDs, validation rules.

This is the single most important agent's output in the whole pipeline —
everything downstream (planning, coding, testing, validation) reads from
this SpecOutput rather than going back to raw text.
"""

from app.schemas import RequirementOutput, SpecOutput
from app.agents.llm_json import call_llm_for_json


SYSTEM_PROMPT = """You are the Specification Agent in a spec-driven software generation pipeline.

Given a confirmed in-scope requirement, produce a complete structured specification
for a small Python FastAPI + SQLite CRUD application.

Rules:
- Every entity needs a singular PascalCase "name" (e.g. "Product") and a
  snake_case plural "table_name" (e.g. "products").
- EVERY entity must have BOTH a GET list endpoint (e.g. GET /categories) AND
  a POST create endpoint (e.g. POST /categories), even simple lookup/reference
  entities like Category, Department, or Supplier -- otherwise the user has
  no way to add records for that entity through the generated UI at all.
- Do NOT include an "id" field in any entity's field list — a primary key
  "id" is added automatically by the code templates. Listing it yourself
  causes the generated Create schema to wrongly require the client to
  supply one.
- Every field has a "type" which MUST be one of: "str", "int", "float", "bool", "datetime".
- Standard CRUD endpoints (POST/GET list/GET one/PUT/DELETE) should exist for the
  main entity, plus any domain-specific endpoints implied by the description
  (e.g. a low-stock filter, a category filter).
- Every functional requirement gets a sequential ID like "FR-01", "FR-02", ...
- Every endpoint's "fr_ids" list must reference at least one real FR-ID from
  functional_requirements.
- If auth_enabled is true, mark write endpoints (POST/PUT/DELETE) as "protected": true,
  and add a functional requirement covering that protected endpoints reject
  unauthenticated requests.
- Keep it reasonably scoped: up to 4 entities, up to 12 endpoints. A
  transaction/history-log entity (e.g. tracking stock movements, changes,
  or events over time, linked via FK to the main entity) is encouraged
  when the description implies tracking activity over time.
- validation_rules should be short plain-English business rules (e.g.
  "stock_quantity must be >= 0"), not code.
- non_functional_requirements MUST stay within this project's actual stack
  (Python, FastAPI, SQLite, Pydantic, pytest). NEVER mention Docker, containerization,
  Kubernetes, cloud deployment, load balancing, or any other infrastructure outside
  that stack — those are explicitly out of scope for this tool.

Return JSON matching this exact structure:
{
  "app_name": string,
  "domain": string,
  "auth_enabled": bool,
  "entities": [
    {"name": string, "table_name": string, "fields": [
      {"name": string, "type": "str"|"int"|"float"|"bool"|"datetime",
       "required": bool, "unique": bool, "description": string or null}
    ]}
  ],
  "endpoints": [
    {"method": "GET"|"POST"|"PUT"|"DELETE", "path": string, "entity": string,
     "description": string, "fr_ids": [string], "protected": bool}
  ],
  "functional_requirements": [
    {"id": string, "description": string, "acceptance_criteria": string}
  ],
  "non_functional_requirements": [string],
  "validation_rules": [string]
}
"""


def _sanitize_spec(spec: SpecOutput) -> SpecOutput:
    """
    Deterministic safety net — prompt instructions aren't 100% reliable,
    so enforce the two hard rules in code as well:
    1. Strip any explicit "id" field (primary keys are template-generated).
    2. Drop any NFR that mentions out-of-scope infrastructure (Docker, k8s, etc.).
    """
    for entity in spec.entities:
        entity.fields = [f for f in entity.fields if f.name.lower() != "id"]

    banned_terms = ["docker", "kubernetes", "container", "cloud deploy", "load balanc"]
    spec.non_functional_requirements = [
        nfr for nfr in spec.non_functional_requirements
        if not any(term in nfr.lower() for term in banned_terms)
    ]
    return spec


def run_specification_agent(requirement: RequirementOutput) -> SpecOutput:
    if not requirement.in_scope:
        raise ValueError(
            f"Cannot specify an out-of-scope requirement: {requirement.rejection_reason}"
        )

    user_prompt = (
        f"App name: {requirement.app_name}\n"
        f"Domain: {requirement.domain}\n"
        f"Auth enabled: {requirement.auth_enabled}\n"
        f"Original description: {requirement.raw_description}\n"
    )

    result = call_llm_for_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=SpecOutput,
    )
    return _sanitize_spec(result)  # type: ignore[arg-type]


if __name__ == "__main__":
    from app.agents.requirement_agent import run_requirement_agent

    description = "Build an inventory management API with products, categories, stock quantities, and low-stock alerts."
    requirement = run_requirement_agent(description)
    print("Requirement:")
    print(requirement.model_dump_json(indent=2))

    if requirement.in_scope:
        spec = run_specification_agent(requirement)
        print("\nSpecification:")
        print(spec.model_dump_json(indent=2))
