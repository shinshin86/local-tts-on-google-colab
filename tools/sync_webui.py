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
README_JA = REPO_ROOT / "README.ja.md"
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
    "Higgs Audio v3": "Higgs-Audio-v3",
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


def _common_python_prefix(names: list[str]) -> str:
    """Longest underscore-bounded prefix shared by every Python var name.

    Used to strip the engine prefix from labels in the WebUI (e.g.
    KOKORO_DEFAULT_VOICE -> "Voice", since both Kokoro params start with
    KOKORO_DEFAULT). Returns "" when fewer than 2 names share a meaningful
    prefix.
    """
    if not names:
        return ""
    if len(names) == 1:
        # Single param: strip everything up to the last underscore.
        i = names[0].rfind("_")
        return names[0][:i] if i > 0 else ""
    prefix = names[0]
    for n in names[1:]:
        while not n.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    # Trim to the last underscore boundary so we never cut a word mid-character.
    if not prefix.endswith("_"):
        i = prefix.rfind("_")
        return prefix[:i] if i > 0 else ""
    return prefix.rstrip("_")


def parse_readme_status(readme_text: str) -> dict[str, dict]:
    """Extract {engine_name: {status, languages}} from the 'Supported engines' table."""
    result: dict[str, dict] = {}
    in_table = False
    for line in readme_text.splitlines():
        if (
            "| Engine | Colab Status | Languages |" in line
            or "| エンジン | Colab 動作確認 | 言語 |" in line
        ):
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


# Heading → engine id overrides for cases where the README heading text
# diverges from the INSTALLERS engine id beyond a trailing parenthetical.
HEADING_TO_ENGINE = {
    "CSM-1B": "CSM-1B",
}


def parse_readme_descriptions(readme_text: str) -> dict[str, str]:
    """Extract {engine_id: first paragraph} from README's per-engine '### ...' headings.

    Used to give the WebUI's "Status & license details" panel some context for
    engines whose multi_tts_openai_colab.py #@markdown block is empty or terse.
    The first paragraph after the heading (until the first blank line) is used.
    """
    result: dict[str, str] = {}
    current_engine: str | None = None
    para_lines: list[str] = []

    def flush():
        nonlocal current_engine, para_lines
        if current_engine and para_lines and current_engine not in result:
            text = " ".join(p.strip() for p in para_lines).strip()
            if text:
                result[current_engine] = text
        para_lines = []

    for line in readme_text.splitlines():
        if line.startswith("### "):
            flush()
            heading = line[4:].strip()
            normalized = re.sub(r"\s*\([^)]*\)\s*$", "", heading).strip()
            current_engine = HEADING_TO_ENGINE.get(normalized, normalized)
            para_lines = []
            continue
        if current_engine is None:
            continue
        if line.strip() == "":
            if para_lines:
                flush()
                # Skip subsequent paragraphs under this heading.
                current_engine = None
            continue
        para_lines.append(line)

    flush()
    return result


def parse_readme_license_notes(readme_text: str) -> dict[str, list[str]]:
    """Extract concise Japanese license notes from README.ja.md's license table."""
    result: dict[str, list[str]] = {}
    in_table = False
    for line in readme_text.splitlines():
        if "| エンジン | コード | モデル重み | 商用利用 | 備考 |" in line:
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            break
        if line.startswith("|---"):
            continue

        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) < 5:
            continue
        engine, code, weights, commercial, note = parts[:5]
        if not engine:
            continue
        text = f"- ライセンス: コードは {code}、モデル重みは {weights}。商用利用: {commercial}。"
        if note:
            text += f" 備考: {note}"
        result.setdefault(engine, []).append(text)
    return result


def build_engines_json() -> dict:
    source = COLAB_SCRIPT.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    readme_ja = README_JA.read_text(encoding="utf-8")

    sections = parse_sections(source)
    var_to_flag, bool_flags = parse_cmd_mapping(source)
    status_map = parse_readme_status(readme)
    status_map_ja = parse_readme_status(readme_ja)
    description_map = parse_readme_descriptions(readme)
    description_map_ja = parse_readme_descriptions(readme_ja)
    license_notes_ja = parse_readme_license_notes(readme_ja)

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
        status_info_ja = status_map_ja.get(engine_id, {})
        engines_by_id[engine_id] = {
            "id": engine_id,
            "title": raw_title,
            "status": status_info.get("status", ""),
            "status_ja": status_info_ja.get("status", ""),
            "languages": status_info.get("languages", ""),
            "languages_ja": status_info_ja.get("languages", ""),
            "description": description_map.get(engine_id, ""),
            "description_ja": description_map_ja.get(engine_id, ""),
            "notes": section["notes"],
            "notes_ja": license_notes_ja.get(engine_id, []),
            "prefix": _common_python_prefix([p["name"] for p in params]),
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
            status_info_ja = status_map_ja.get(engine_id, {})
            engines.append(
                {
                    "id": engine_id,
                    "title": engine_id,
                    "status": status_info.get("status", ""),
                    "status_ja": status_info_ja.get("status", ""),
                    "languages": status_info.get("languages", ""),
                    "languages_ja": status_info_ja.get("languages", ""),
                    "description": description_map.get(engine_id, ""),
                    "description_ja": description_map_ja.get(engine_id, ""),
                    "notes": [],
                    "notes_ja": license_notes_ja.get(engine_id, []),
                    "prefix": "",
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
