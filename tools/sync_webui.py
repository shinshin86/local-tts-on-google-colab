"""Generate docs/engines.json from multi_tts_openai_colab.py + README.md.

multi_tts_openai_colab.py is the source of truth for the Colab cell. This
script parses its `#@param` declarations and `#@markdown` section headers to
build a JSON schema the static WebUI (docs/) consumes.

Run this whenever multi_tts_openai_colab.py changes:

    python tools/sync_webui.py
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COLAB_SCRIPT = REPO_ROOT / "multi_tts_openai_colab.py"
README = REPO_ROOT / "README.md"
OUTPUT = REPO_ROOT / "docs" / "engines.json"

PARAM_RE = re.compile(
    r"^(?P<var>[A-Z_][A-Z0-9_]*)\s*=\s*(?P<value>.+?)\s*#@param(?:\s+(?P<meta>.+))?$"
)
MD_RE = re.compile(r"^#@markdown\s?(?P<text>.*)$")
DIVIDER = "#@markdown ---"

# Section header → ENGINE dropdown id, for cases where the #@markdown title
# does not exactly match the engine id used in INSTALLERS / --engine.
SECTION_TO_ENGINE = {
    "OpenVoice V2": "OpenVoice-V2",
    "Sesame CSM-1B": "CSM-1B",
    "StyleTTS 2": "StyleTTS2",
    "Higgs Audio v2": "Higgs-Audio-v2",
}


def parse_param_spec(meta: str | None) -> dict:
    """Parse the metadata after `#@param`.

    `["a", "b"]` → {"type": "select", "options": [...]}
    `{type:"string"}` → {"type": "string"}
    `{type:"number"}` → {"type": "number"}
    `{type:"integer"}` → {"type": "integer"}
    `{type:"boolean"}` → {"type": "boolean"}
    """
    if not meta:
        return {"type": "string"}
    meta = meta.strip()
    if meta.startswith("["):
        return {"type": "select", "options": list(ast.literal_eval(meta))}
    if meta.startswith("{"):
        # Convert JS-style {type:"string"} into JSON {"type":"string"}.
        normalized = re.sub(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'"\1":', meta)
        spec = json.loads(normalized)
        return {"type": spec.get("type", "string")}
    return {"type": "string"}


def parse_literal(raw: str):
    """Parse the RHS of a Python assignment (handles strings, numbers, booleans)."""
    return ast.literal_eval(raw)


def normalize_section_title(title: str) -> str:
    """`F5-TTS (GPU required)` → `F5-TTS`. Section titles often carry a parenthetical."""
    return re.split(r"\s*[(–—]", title)[0].strip()


def parse_sections(source: str) -> list[dict]:
    """Walk the Colab script line-by-line and bucket params under sections.

    Returns a list of section dicts: {raw_title, notes, params}.
    The first section ("repo_settings" + "common_cli") has raw_title=None and
    contains REPO_URL through OPENAI_MODEL_ID.
    """
    sections: list[dict] = [{"raw_title": None, "notes": [], "params": []}]
    for line in source.splitlines():
        stripped = line.rstrip()
        if stripped == DIVIDER:
            sections.append({"raw_title": None, "notes": [], "params": []})
            continue

        md_match = MD_RE.match(stripped)
        if md_match:
            text = md_match.group("text").strip()
            section = sections[-1]
            if section["raw_title"] is None and section is not sections[0]:
                section["raw_title"] = text
            else:
                if text:
                    section["notes"].append(text)
            continue

        param_match = PARAM_RE.match(stripped)
        if param_match:
            var = param_match.group("var")
            raw_value = param_match.group("value")
            meta = param_match.group("meta")
            spec = parse_param_spec(meta)
            try:
                default = parse_literal(raw_value)
            except (SyntaxError, ValueError):
                default = raw_value
            sections[-1]["params"].append(
                {"name": var, "default": default, **spec}
            )

    return sections


def parse_cmd_mapping(source: str) -> tuple[dict[str, str], list[dict]]:
    """Walk build_bootstrap_command() to extract var→flag mapping + boolean flags.

    Returns:
      var_to_flag: {PYTHON_VAR: "--cli-flag"} for every key/value pair appended to cmd.
      bool_flags: list of {"var", "true_flag", "false_flag"} for conditional appends.
    """
    tree = ast.parse(source)
    var_to_flag: dict[str, str] = {}
    bool_flags: list[dict] = []

    func = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "build_bootstrap_command"
        ),
        None,
    )
    if func is None:
        raise RuntimeError("build_bootstrap_command not found")

    for stmt in func.body:
        # cmd = [ ... ]
        if isinstance(stmt, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "cmd" for t in stmt.targets
        ):
            if isinstance(stmt.value, ast.List):
                elts = stmt.value.elts
                i = 0
                while i < len(elts) - 1:
                    flag_node = elts[i]
                    val_node = elts[i + 1]
                    if (
                        isinstance(flag_node, ast.Constant)
                        and isinstance(flag_node.value, str)
                        and flag_node.value.startswith("--")
                    ):
                        var = _extract_var(val_node)
                        if var:
                            var_to_flag[var] = flag_node.value
                            i += 2
                            continue
                    i += 1

        # if SOME_FLAG: cmd.append("--some-flag")
        elif isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Name):
            var = stmt.test.id
            for inner in stmt.body:
                flag = _extract_append_constant(inner)
                if flag is not None:
                    bool_flags.append(
                        {"var": var, "true_flag": flag, "false_flag": None}
                    )

        # cmd.append("--expose-public-url" if EXPOSE_PUBLIC_URL else "--no-expose-public-url")
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "append"
                and len(call.args) == 1
                and isinstance(call.args[0], ast.IfExp)
            ):
                ifexp = call.args[0]
                if (
                    isinstance(ifexp.test, ast.Name)
                    and isinstance(ifexp.body, ast.Constant)
                    and isinstance(ifexp.orelse, ast.Constant)
                ):
                    bool_flags.append(
                        {
                            "var": ifexp.test.id,
                            "true_flag": ifexp.body.value,
                            "false_flag": ifexp.orelse.value,
                        }
                    )

    return var_to_flag, bool_flags


def _extract_var(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "str"
        and node.args
        and isinstance(node.args[0], ast.Name)
    ):
        return node.args[0].id
    return None


def _extract_append_constant(stmt: ast.AST) -> str | None:
    if not isinstance(stmt, ast.Expr):
        return None
    call = stmt.value
    if not isinstance(call, ast.Call):
        return None
    if not (isinstance(call.func, ast.Attribute) and call.func.attr == "append"):
        return None
    if len(call.args) != 1 or not isinstance(call.args[0], ast.Constant):
        return None
    return call.args[0].value


def parse_readme_status(readme_text: str) -> dict[str, dict]:
    """Extract {engine_name: {status, languages}} from the 'Supported engines' table."""
    result: dict[str, dict] = {}
    in_table = False
    for line in readme_text.splitlines():
        if "| Engine | Colab Status | Languages |" in line:
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            if line.startswith("|---"):
                continue
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) >= 3:
                engine, status, languages = parts[0], parts[1], parts[2]
                result[engine] = {"status": status, "languages": languages}
    return result


def build_engines_json() -> dict:
    source = COLAB_SCRIPT.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")

    sections = parse_sections(source)
    var_to_flag, bool_flags = parse_cmd_mapping(source)
    status_map = parse_readme_status(readme)

    # Section 0 holds REPO_URL/REPO_REF/WORKDIR + ENGINE + EXPOSE_PUBLIC_URL +
    # TEST_TEXT/TEST_SPEED/TEST_VOICE/OPENAI_MODEL_ID.
    common_section = sections[0]

    repo_settings: list[dict] = []
    engine_selector: dict | None = None
    expose_setting: dict | None = None
    common_cli: list[dict] = []
    engine_list_ids: list[str] = []

    for param in common_section["params"]:
        name = param["name"]
        if name in {"REPO_URL", "REPO_REF", "WORKDIR"}:
            repo_settings.append(param)
        elif name == "ENGINE":
            engine_selector = param
            engine_list_ids = list(param.get("options", []))
        elif name == "EXPOSE_PUBLIC_URL":
            expose_setting = param
        else:
            entry = dict(param)
            entry["flag"] = var_to_flag.get(name)
            common_cli.append(entry)

    bool_flag_by_var = {bf["var"]: bf for bf in bool_flags}

    if expose_setting is not None and "EXPOSE_PUBLIC_URL" in bool_flag_by_var:
        bf = bool_flag_by_var["EXPOSE_PUBLIC_URL"]
        expose_setting["true_flag"] = bf["true_flag"]
        expose_setting["false_flag"] = bf["false_flag"]

    # Sections after [0] are per-engine.
    engines_by_id: dict[str, dict] = {}
    for section in sections[1:]:
        raw_title = section["raw_title"] or ""
        if not raw_title:
            continue
        normalized = normalize_section_title(raw_title)
        engine_id = SECTION_TO_ENGINE.get(normalized, normalized)
        if engine_id not in engine_list_ids:
            # Section without a corresponding entry in the ENGINE dropdown; skip.
            continue

        params = []
        for param in section["params"]:
            entry = dict(param)
            name = param["name"]
            if name in bool_flag_by_var:
                bf = bool_flag_by_var[name]
                entry["true_flag"] = bf["true_flag"]
                entry["false_flag"] = bf["false_flag"]
                entry["type"] = "boolean"
            else:
                entry["flag"] = var_to_flag.get(name)
            params.append(entry)

        status_info = status_map.get(engine_id, {})
        engines_by_id[engine_id] = {
            "id": engine_id,
            "title": raw_title,
            "status": status_info.get("status", ""),
            "languages": status_info.get("languages", ""),
            "notes": section["notes"],
            "params": params,
        }

    # Engines listed in ENGINE dropdown but with no #@markdown section
    # (no engine-specific params). Use a minimal placeholder entry.
    engines: list[dict] = []
    for engine_id in engine_list_ids:
        if engine_id in engines_by_id:
            engines.append(engines_by_id[engine_id])
        else:
            status_info = status_map.get(engine_id, {})
            engines.append(
                {
                    "id": engine_id,
                    "title": engine_id,
                    "status": status_info.get("status", ""),
                    "languages": status_info.get("languages", ""),
                    "notes": [],
                    "params": [],
                }
            )

    return {
        "repo_settings": repo_settings,
        "engine_selector": engine_selector,
        "expose_setting": expose_setting,
        "common_cli": common_cli,
        "engines": engines,
    }


def main() -> None:
    data = build_engines_json()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    n_engines = len(data["engines"])
    n_params = sum(len(e["params"]) for e in data["engines"]) + len(data["common_cli"])
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)} ({n_engines} engines, {n_params} params)")


if __name__ == "__main__":
    main()
