# CLAUDE.md

Repository-specific operational rules for Claude Code. These take precedence
over default behavior.

## Repository at a glance

This repo wraps several local TTS engines as an OpenAI-compatible
`/v1/audio/speech` endpoint, intended to be launched on Google Colab. The
codebase is organized as:

- `colab/bootstrap.py` — entrypoint invoked from a Colab cell. Parses CLI flags
  and dispatches to the selected engine.
- `multi_tts_openai_colab.py` — canonical Colab cell. **Source of truth** for
  the embedded cells in `README.md` / `README.ja.md`.
- `src/config.py` — `Settings` dataclass holding every engine's tunables.
- `src/installers/<engine>.py` — clones / pip-installs the upstream engine,
  spawns its uvicorn process, returns a state dict.
- `src/installers/__init__.py` — `INSTALLERS` registry. Engine names here are
  what the `--engine` flag accepts.
- `src/apps/<engine>_app.py` — FastAPI app implementing `/v1/audio/speech`.
- `src/launcher.py` — orchestrates install → wait_http → smoke test → optional
  cloudflared. Also owns the per-engine voice resolution + hint text.
- `src/runtime.py` — shared shell / venv / cloudflared helpers.

## Mandatory procedure when adding (or changing) an engine

When the user asks to add a new TTS engine, or modify an existing one, you
**must** complete every step below before declaring the work done. Skipping
the Colab verification step is not acceptable, because nothing in this repo is
exercised on the developer's local machine.

1. **Verify the upstream license** — fetch the GitHub LICENSE file *and* every
   Hugging Face model card you intend to ship as a default. Code license and
   weights license can differ; weights license can differ between model sizes
   (this bit us with OuteTTS: 0.6B is Apache 2.0 but the 1B Llama-based
   variant is CC-BY-NC-SA-4.0 + Llama Community License). Record both in the
   README license table.
2. **Implement the engine** in the standard layout:
   - `src/installers/<engine>.py`
   - `src/apps/<engine>_app.py`
   - Register in `src/installers/__init__.py`
   - Add tunables to `src/config.py` (`Settings` dataclass)
   - Add CLI flags + Settings-kwarg passthrough in `colab/bootstrap.py`
   - Add `resolve_selected_voice` branch + `print_engine_voice_hints` block in
     `src/launcher.py`
   - Add form params + cmd args in `multi_tts_openai_colab.py`
3. **Update the READMEs** in *both* `README.md` and `README.ja.md`:
   - Supported-engines table row (Colab status + languages)
   - Engine-specific notes section
   - License table row (separate rows per model variant if licenses differ)
   - References list link at the bottom
4. **Sync the embedded Colab cells**. `multi_tts_openai_colab.py` is the
   source of truth; the cells inside `README.md` and `README.ja.md` must
   match it byte-for-byte. The mechanical sync looks like:

   ```bash
   python3 - <<'PY'
   import pathlib
   src = pathlib.Path("multi_tts_openai_colab.py").read_text(encoding="utf-8")
   cell = "".join(src.splitlines(keepends=True)[8:])  # drop the 7-line docstring + blank line
   if not cell.endswith("\n"):
       cell += "\n"
   for r in ["README.md", "README.ja.md"]:
       p = pathlib.Path(r)
       text = p.read_text(encoding="utf-8")
       s = text.find("#@title Local TTS on Google Colab")
       e = text.find("\nmain()\n", s) + len("\nmain()\n")
       p.write_text(text[:s] + cell + text[e:], encoding="utf-8")
   PY
   ```

   Run this any time you touch `multi_tts_openai_colab.py`.
5. **Commit per engine**, with English commit messages. Mention the license
   findings explicitly when they differ between code and weights, or between
   model sizes.
6. **Push** the feature branch (`git push -u origin <branch>`) before testing.
   Colab pulls from the remote, so verification can't run against unpushed
   commits.
7. **Verify on Google Colab** using the Colab MCP. Do *not* attempt to run
   `bootstrap.py` or any engine locally — the engines target Linux + CUDA +
   Colab's preinstalled toolchain. Instead:
   - Call `mcp__colab-mcp__open_colab_browser_connection` to attach.
   - Open or create a notebook, set `REPO_REF` to the feature branch name.
   - Run the cell, confirm the trycloudflare public URL appears, and exercise
     `/v1/audio/speech` against that public URL (this is the contract the repo
     is built around — local-only validation does not count).
   - Capture failures (logs, stack traces, OOM messages) before iterating.
8. **Open a PR** with an English title and body summarizing the new engine,
   its license, and the Colab verification result. Always link the trycloudflare
   verification.

## Conventions

- **PR titles and descriptions are written in English** for this repo.
- **Never run engines or `colab/bootstrap.py` locally.** It will not work and
  it is not the supported environment. All functional verification happens on
  Google Colab.
- Engine names use upper camel / hyphenated form (`Chatterbox`, `OpenVoice-V2`,
  `MOSS-TTS-Nano`). Match what's already in the `INSTALLERS` registry.
- Voice presets are exposed as the OpenAI `voice` parameter. The convention is
  `default` (no reference / built-in speaker) and `clone` (uses
  `--<engine>-prompt-wav` and possibly `--<engine>-prompt-text`). If `clone`
  is unavailable because the user did not configure a prompt wav, return a
  4xx with a clear message rather than silently falling back.
- Watermarks and inaudible markers (e.g. SilentCipher in Sarashina-TTS) must
  not be removed — surface the requirement in the engine's README section.
- Every engine must work behind a `trycloudflare` public URL — this is the
  product's primary contract, not a nice-to-have.
