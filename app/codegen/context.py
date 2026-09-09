"""
Turns a SpecOutput into the exact data structure the Jinja2 templates need:
SQLAlchemy/Pydantic type mappings, foreign-key detection, and classification
of each endpoint as "standard CRUD" (template auto-implements it) or
"custom" (gets a stub for the Day 4 Coding Agent to fill in).

This is all deterministic — no LLM calls — which is why it's fully
testable offline.
"""

import re
from app.schemas import SpecOutput, EntityDef

SQLA_TYPE_MAP = {
    "str": "String",
    "int": "Integer",
    "float": "Float",
    "bool": "Boolean",
    "datetime": "DateTime",
}

PY_TYPE_MAP = {
    "str": "str",
    "int": "int",
    "float": "float",
    "bool": "bool",
    "datetime": "datetime",
}


def _sanitize_func_name(method: str, path: str) -> str:
    stripped = path.strip("/")
    stripped = re.sub(r"[{}]", "", stripped)
    stripped = re.sub(r"[^a-zA-Z0-9]+", "_", stripped)
    return f"{method.lower()}_{stripped}".strip("_")


def build_entity_fk_map(entities: list[EntityDef]) -> dict[str, str]:
    """Maps lowercased entity name -> class name, for FK lookups."""
    return {e.name.lower(): e.name for e in entities}


_PRICE_KEYWORDS = ("price", "cost", "value", "amount")
_QUANTITY_KEYWORDS = ("quantity", "stock", "count", "qty")
_THRESHOLD_KEYWORDS = ("threshold", "reorder", "minimum", "min_stock")


def _field_role(field_name: str) -> str | None:
    lower = field_name.lower()
    if any(k in lower for k in _THRESHOLD_KEYWORDS):
        return "threshold"
    if any(k in lower for k in _QUANTITY_KEYWORDS):
        return "quantity"
    if any(k in lower for k in _PRICE_KEYWORDS):
        return "price"
    return None


def build_entity_context(spec: SpecOutput) -> list[dict]:
    entity_by_lower = build_entity_fk_map(spec.entities)
    table_by_class = {e.name: e.table_name for e in spec.entities}

    entities_ctx = []
    for entity in spec.entities:
        var_name = entity.name.lower()
        fields_ctx = []
        fk_fields_ctx = []

        for field in entity.fields:
            is_fk = False
            fk_target_class = None
            fk_target_table = None
            if field.name.endswith("_id"):
                stripped = field.name[:-3].lower()
                if stripped in entity_by_lower and stripped != var_name:
                    is_fk = True
                    fk_target_class = entity_by_lower[stripped]
                    fk_target_table = table_by_class[fk_target_class]

            py_type = PY_TYPE_MAP[field.type]
            field_ctx = {
                "name": field.name,
                "sqla_type": SQLA_TYPE_MAP[field.type],
                "py_type": py_type,
                "type_annotation": f"{py_type} | None = None" if not field.required else py_type,
                "optional_type_annotation": f"{py_type} | None = None",
                "nullable": not field.required,
                "unique": field.unique,
                "is_fk": is_fk,
                "fk_target_table": fk_target_table,
                "fk_target_class": fk_target_class,
                "role": _field_role(field.name) if field.type in ("int", "float") and not is_fk else None,
            }
            fields_ctx.append(field_ctx)
            if is_fk:
                fk_fields_ctx.append(field_ctx)

        entities_ctx.append({
            "class_name": entity.name,
            "table_name": entity.table_name,
            "var_name": var_name,
            "fields": fields_ctx,
            "fk_fields": fk_fields_ctx,
            "has_fields": len(fields_ctx) > 0,
            "price_field": next((f["name"] for f in fields_ctx if f["role"] == "price"), None),
            "quantity_field": next((f["name"] for f in fields_ctx if f["role"] == "quantity"), None),
            "threshold_field": next((f["name"] for f in fields_ctx if f["role"] == "threshold"), None),
            # standard CRUD paths, precomputed as plain strings to avoid
            # Jinja2 vs FastAPI curly-brace collisions in the templates
            "create_path": f"/{entity.table_name}",
            "list_path": f"/{entity.table_name}",
            "get_one_path": f"/{entity.table_name}/{{{var_name}_id}}",
            "update_path": f"/{entity.table_name}/{{{var_name}_id}}",
            "delete_path": f"/{entity.table_name}/{{{var_name}_id}}",
            "has_create": False,
            "has_list": False,
            "has_get_one": False,
            "has_update": False,
            "has_delete": False,
            "create_fr_ids": [],
            "list_fr_ids": [],
            "get_one_fr_ids": [],
            "update_fr_ids": [],
            "delete_fr_ids": [],
            "create_protected": False,
            "update_protected": False,
            "delete_protected": False,
        })

    return entities_ctx


_GET_ONE_PATTERN = re.compile(r"^/[a-zA-Z0-9_\-]+/\{[a-zA-Z0-9_]+\}$")


def _normalize(path: str) -> str:
    """Kebab-case ('/leave-requests') and snake_case ('/leave_requests') should
    be treated as the same path when classifying endpoints -- both are common
    REST conventions and real LLM-generated specs use either interchangeably."""
    return path.lower().replace("-", "_")


def classify_endpoints(spec: SpecOutput, entities_ctx: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Returns (entities_ctx with standard_ops filled in, custom_endpoints list).
    Any spec endpoint matching a canonical CRUD shape for its entity gets
    marked on that entity. Everything else becomes a custom endpoint stub.
    """
    entity_by_name = {e["class_name"].lower(): e for e in entities_ctx}
    custom_endpoints = []

    for ep in spec.endpoints:
        entity_ctx = entity_by_name.get(ep.entity.lower())
        table_name = entity_ctx["table_name"] if entity_ctx else None
        matched = False
        norm_path = _normalize(ep.path)

        if entity_ctx is not None:
            norm_table = _normalize(f"/{table_name}")
            if ep.method == "POST" and norm_path == norm_table:
                entity_ctx["has_create"] = True
                entity_ctx["create_protected"] = ep.protected
                entity_ctx["create_fr_ids"] = ep.fr_ids
                matched = True
            elif ep.method == "GET" and norm_path == norm_table:
                entity_ctx["has_list"] = True
                entity_ctx["list_fr_ids"] = ep.fr_ids
                matched = True
            elif ep.method == "GET" and _GET_ONE_PATTERN.match(ep.path) and norm_path.startswith(norm_table + "/"):
                entity_ctx["has_get_one"] = True
                entity_ctx["get_one_fr_ids"] = ep.fr_ids
                matched = True
            elif ep.method == "PUT" and _GET_ONE_PATTERN.match(ep.path) and norm_path.startswith(norm_table + "/"):
                entity_ctx["has_update"] = True
                entity_ctx["update_protected"] = ep.protected
                entity_ctx["update_fr_ids"] = ep.fr_ids
                matched = True
            elif ep.method == "DELETE" and _GET_ONE_PATTERN.match(ep.path) and norm_path.startswith(norm_table + "/"):
                entity_ctx["has_delete"] = True
                entity_ctx["delete_protected"] = ep.protected
                entity_ctx["delete_fr_ids"] = ep.fr_ids
                matched = True

        if not matched:
            custom_endpoints.append({
                "method": ep.method,
                "method_lower": ep.method.lower(),
                "path": ep.path,
                "entity": ep.entity,
                "description": ep.description,
                "protected": ep.protected,
                "func_name": _sanitize_func_name(ep.method, ep.path),
                "fr_ids": ep.fr_ids,
            })

    return entities_ctx, custom_endpoints


def build_template_context(spec: SpecOutput) -> dict:
    entities_ctx = build_entity_context(spec)
    entities_ctx, custom_endpoints = classify_endpoints(spec, entities_ctx)

    needs_datetime_import = any(
        f.type == "datetime" for e in spec.entities for f in e.fields
    )

    all_endpoints_summary = [
        {"method": ep.method, "path": ep.path, "description": ep.description}
        for ep in spec.endpoints
    ]

    label_field_by_class = {}
    for e in spec.entities:
        first_str = next((f.name for f in e.fields if f.type == "str"), "id")
        label_field_by_class[e.name] = first_str

    for entity_ctx in entities_ctx:
        entity_ctx["date_field"] = next((f["name"] for f in entity_ctx["fields"] if f["py_type"] == "datetime"), None)
        entity_ctx["fk_breakdowns"] = [
            {
                "field_name": fk["name"],
                "target_class": fk["fk_target_class"],
                "target_var": fk["fk_target_class"].lower(),
                "label_field": label_field_by_class.get(fk["fk_target_class"], "id"),
            }
            for fk in entity_ctx["fk_fields"]
        ]

    # Bounded UI plan: tab TYPE choices are fixed; only which entities land
    # in which tab (and whether a tab appears at all) varies per app. Keeps
    # per-app customization without LLM involvement in the frontend at all.
    entry_entities = [e for e in entities_ctx if e["has_list"]]
    report_entities = [e for e in entities_ctx if e["has_list"] and (e["price_field"] or e["quantity_field"] or e["date_field"] or e["fk_fields"])]
    alert_entities = [e for e in entities_ctx if e["quantity_field"] and e["threshold_field"]]

    ui_plan = [{"id": "entry", "label": "Data Entry", "type": "entry", "entities": entry_entities}]
    if report_entities:
        ui_plan.append({"id": "reports", "label": "Reports", "type": "reports", "entities": report_entities})
    if alert_entities:
        ui_plan.append({"id": "alerts", "label": "Alerts", "type": "alerts", "entities": alert_entities})
    if spec.auth_enabled:
        ui_plan.append({"id": "settings", "label": "Settings", "type": "settings", "entities": []})

    return {
        "app_name": spec.app_name,
        "domain": spec.domain,
        "auth_enabled": spec.auth_enabled,
        "entities": entities_ctx,
        "custom_endpoints": custom_endpoints,
        "needs_datetime_import": needs_datetime_import,
        "all_endpoints": all_endpoints_summary,
        "ui_plan": ui_plan,
    }


# ---------------------------------------------------------------------------
# Test data generation (Day 5) -- deterministic sample values + FK setup,
# so the Testing Agent doesn't need an LLM call for standard CRUD tests.
# ---------------------------------------------------------------------------

_SAMPLE_COUNTER = {"n": 0}


def _sample_literal(field_type: str, field_name: str) -> str:
    """A Python source literal (as a string) for a field's test value."""
    if field_type == "str":
        return f'"Sample {field_name}"'
    if field_type == "int":
        return "42"
    if field_type == "float":
        return "9.99"
    if field_type == "bool":
        return "True"
    if field_type == "datetime":
        return '"2024-01-01T00:00:00"'
    return "None"


def _entity_by_name(spec: SpecOutput, name: str) -> EntityDef | None:
    return next((e for e in spec.entities if e.name.lower() == name.lower()), None)


def build_test_context(spec: SpecOutput) -> dict:
    """
    For each entity, precomputes:
      - parent_setup_lines: source lines that create any FK-required parent
        records directly via the ORM (always via direct DB insert, regardless
        of whether the parent entity has its own create endpoint -- simplest
        reliable approach for test setup).
      - create_payload_literal: a Python dict literal (as source text) for
        the entity's create payload, with FK fields pointing at the parent
        variables created above.
      - sample_field / sample_value_literal: one non-FK field to assert
        against after create, so tests check more than just status codes.
    Only supports one level of FK nesting (sufficient for this project's
    5 supported domains) -- a parent's own FK fields, if any, are left
    unset in its own test-setup construction.
    """
    entities_ctx = build_entity_context(spec)
    entities_ctx, custom_endpoints = classify_endpoints(spec, entities_ctx)

    for entity_ctx in entities_ctx:
        entity_def = _entity_by_name(spec, entity_ctx["class_name"])
        construct_lines: list[str] = []
        add_lines: list[str] = []
        refresh_lines: list[str] = []
        payload_items: list[str] = []
        standalone_payload_items: list[str] = []
        sample_field = None
        sample_value_literal = None

        for field_ctx in entity_ctx["fields"]:
            if field_ctx["is_fk"]:
                parent_class = field_ctx["fk_target_class"]
                parent_var = f"{parent_class.lower()}_obj"
                parent_def = _entity_by_name(spec, parent_class)
                # Only set the parent's own non-FK fields -- one level deep.
                parent_field_assignments = ", ".join(
                    f"{f.name}={_sample_literal(f.type, f.name)}"
                    for f in (parent_def.fields if parent_def else [])
                    if not f.name.endswith("_id")
                )
                construct_lines.append(f"{parent_var} = models.{parent_class}({parent_field_assignments})")
                add_lines.append(f"db.add({parent_var})")
                refresh_lines.append(f"db.refresh({parent_var})")
                payload_items.append(f'"{field_ctx["name"]}": {parent_var}.id')
                standalone_payload_items.append(f'"{field_ctx["name"]}": 1')
            else:
                literal = _sample_literal(field_ctx["py_type"], field_ctx["name"])
                payload_items.append(f'"{field_ctx["name"]}": {literal}')
                standalone_payload_items.append(f'"{field_ctx["name"]}": {literal}')
                if sample_field is None:
                    sample_field = field_ctx["name"]
                    sample_value_literal = literal

        # IMPORTANT: batch into a SINGLE commit, not one commit per parent.
        # SQLAlchemy's default expire_on_commit=True invalidates every
        # already-loaded object in the session on each commit -- with 2+
        # FK parents (e.g. an Enrollment needing both Student and Course),
        # committing after each one silently expires the earlier object,
        # and accessing its .id after db.close() raises DetachedInstanceError.
        parent_setup_lines = construct_lines + add_lines
        if add_lines:
            parent_setup_lines.append("db.commit()")
        parent_setup_lines += refresh_lines

        entity_ctx["parent_setup_lines"] = parent_setup_lines
        entity_ctx["needs_db_setup"] = len(parent_setup_lines) > 0
        entity_ctx["create_payload_literal"] = "{" + ", ".join(payload_items) + "}"
        # Used only where no parent object is created in that test (e.g. the
        # invalid-FK test, which immediately overrides the FK value anyway) --
        # referencing a parent_obj.id here would crash with NameError since
        # the parent was never constructed in that test's scope.
        entity_ctx["standalone_payload_literal"] = "{" + ", ".join(standalone_payload_items) + "}"
        entity_ctx["sample_field"] = sample_field
        entity_ctx["sample_value_literal"] = sample_value_literal
        entity_ctx["has_fk_fields"] = len(entity_ctx["fk_fields"]) > 0

    return {
        "app_name": spec.app_name,
        "auth_enabled": spec.auth_enabled,
        "entities": entities_ctx,
        "custom_endpoints": custom_endpoints,
    }
