"""
Renders a SpecOutput into the full set of GeneratedFile objects using
the Jinja2 templates in app/templates/. This is the deterministic half
of the Coding Agent -- the LLM (Day 4) only fills in custom endpoint
bodies after this runs.
"""

import os
from jinja2 import Environment, FileSystemLoader

from app.schemas import SpecOutput, GeneratedFile
from app.codegen.context import build_template_context, build_test_context

_TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
_env = Environment(
    loader=FileSystemLoader(_TEMPLATES_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_project(spec: SpecOutput) -> list[GeneratedFile]:
    context = build_template_context(spec)

    files = [
        GeneratedFile(path="database.py", content=_env.get_template("database.py.j2").render(**context)),
        GeneratedFile(path="models.py", content=_env.get_template("models.py.j2").render(**context)),
        GeneratedFile(path="schemas.py", content=_env.get_template("schemas.py.j2").render(**context)),
        GeneratedFile(path="services.py", content=_env.get_template("services.py.j2").render(**context)),
        GeneratedFile(path="main.py", content=_env.get_template("main.py.j2").render(**context)),
        GeneratedFile(path="frontend.html", content=_env.get_template("frontend.html.j2").render(**context)),
        GeneratedFile(path="requirements.txt", content=_env.get_template("requirements.txt.j2").render(**context)),
        GeneratedFile(path="README.md", content=_env.get_template("README.md.j2").render(**context)),
    ]

    if spec.auth_enabled:
        files.append(
            GeneratedFile(path="security.py", content=_env.get_template("security.py.j2").render(**context))
        )

    return files


def render_test_file(spec: SpecOutput) -> GeneratedFile:
    context = build_test_context(spec)
    content = _env.get_template("test_api.py.j2").render(**context)
    return GeneratedFile(path="tests/test_api.py", content=content)


if __name__ == "__main__":
    # Manual smoke test using a hand-built spec (no Groq call needed --
    # this whole module is deterministic).
    from app.rag.example_bank import EXAMPLE_BANK

    spec = EXAMPLE_BANK[0]  # Inventory Management gold-standard spec
    files = render_project(spec)
    for f in files:
        print(f"--- {f.path} ({len(f.content)} chars) ---")
