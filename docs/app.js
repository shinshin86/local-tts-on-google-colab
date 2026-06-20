const { createApp, reactive, computed, watch, onMounted, ref } = Vue;

const REPO_URL = "https://github.com/shinshin86/local-tts-on-google-colab";

const ACRONYMS = {
  hf: "HF",
  vllm: "vLLM",
  wav: "(.wav)",
  id: "ID",
  lang: "language",
  cfg: "CFG",
  stg: "STG",
  vc: "VC",
  ddpm: "DDPM",
  sfx: "SFX",
  ckpt: "checkpoint",
  sovits: "SoVITS",
  url: "URL",
  api: "API",
  openai: "OpenAI",
  fp32: "fp32",
  bf16: "bf16",
  fp16: "fp16",
  int4: "int4",
  int8: "int8",
  bnb: "bnb",
  dtype: "dtype",
};

const I18N = {
  en: {
    pageTitle: "Local TTS on Google Colab — Cell Generator",
    pageDescription:
      "Generate a ready-to-paste Google Colab cell for the local-tts-on-google-colab project. Pick a TTS engine and copy a cell — that's it.",
    taglineBefore: "Pick a TTS engine, click ",
    taglineMiddle: ", paste into a ",
    taglineAfter:
      ", and run it. Scratchpad notebooks are ephemeral — nothing is written to your Drive until you explicitly save.",
    ttsEngine: "TTS engine",
    aboutEngine: "About this engine",
    colabStatus: "Colab status:",
    licenseMissingTitle: "License info not surfaced here",
    licenseMissingBody: "— check the upstream repository link above for the engine's code & weight licenses.",
    advancedOptions: "Advanced {engine} options",
    enable: "Enable",
    sampleRequest: "Sample request",
    copyCell: "Copy cell",
    copied: "Copied ✓",
    pasteIntoColab: "Paste into Colab & run",
    openColab: "Open Colab scratchpad ↗",
    scratchpadHint: "Scratchpad notebooks aren't saved to Drive unless you explicitly save them.",
    downloadPy: "Download as .py",
    resetOptions: "Reset all options",
    copyAgentPrompt: "Ask an AI agent instead",
    agentPromptHint: "Use this instead of copying and running the Colab cell manually.",
    copyPrompt: "Copy prompt",
    loading: "Loading engines.json…",
    githubRepository: "GitHub repository",
    generatedFrom: "Generated from",
    linesMeta: "{count} lines · {kind}",
    copyCellFallback: "Could not copy automatically — please select & copy the cell preview below.",
    copyPromptFallback: "Could not copy automatically — please select & copy the prompt preview below.",
    voiceSpeaker: "Voice / speaker",
    speaker: "Speaker",
    language: "Language",
    model: "Model",
    statusUnknown: "Unknown",
    statusWorks: "Works",
    statusNotWorking: "Not working",
    commonCli: {
      TEST_TEXT: "Sample text to synthesize",
      TEST_SPEED: "Playback speed",
      TEST_VOICE: "Voice override (optional)",
      OPENAI_MODEL_ID: "OpenAI model ID (optional)",
    },
  },
  ja: {
    pageTitle: "Local TTS on Google Colab — セル生成ツール",
    pageDescription:
      "local-tts-on-google-colab 用のGoogle Colabセルを生成します。TTSエンジンを選んでコピーするだけで使えます。",
    taglineBefore: "TTSエンジンを選び、「",
    taglineMiddle: "」をクリックして",
    taglineAfter:
      "に貼り付けて実行します。Scratchpadのノートブックは一時的なもので、明示的に保存しない限りDriveには書き込まれません。",
    ttsEngine: "TTSエンジン",
    aboutEngine: "このエンジンについて",
    colabStatus: "Colabでの状態:",
    licenseMissingTitle: "ライセンス情報はここには表示されていません",
    licenseMissingBody: "— エンジンのコードと重みのライセンスは上流リポジトリで確認してください。",
    advancedOptions: "{engine} の詳細オプション",
    enable: "有効化",
    sampleRequest: "サンプルリクエスト",
    copyCell: "セルをコピー",
    copied: "コピーしました ✓",
    pasteIntoColab: "Colabに貼り付けて実行",
    openColab: "Colab scratchpadを開く ↗",
    scratchpadHint: "Scratchpadのノートブックは、明示的に保存しない限りDriveには保存されません。",
    downloadPy: ".pyとしてダウンロード",
    resetOptions: "すべて初期値に戻す",
    copyAgentPrompt: "AIエージェントに依頼する",
    agentPromptHint: "手動でColabセルをコピーして実行する代わりに使えます。",
    copyPrompt: "プロンプトをコピー",
    loading: "engines.jsonを読み込み中…",
    githubRepository: "GitHubリポジトリ",
    generatedFrom: "生成元",
    linesMeta: "{count}行 · {kind}",
    copyCellFallback: "自動コピーできませんでした。下のセルプレビューを選択してコピーしてください。",
    copyPromptFallback: "自動コピーできませんでした。下のプロンプトプレビューを選択してコピーしてください。",
    voiceSpeaker: "声 / 話者",
    speaker: "話者",
    language: "言語",
    model: "モデル",
    statusUnknown: "不明",
    statusWorks: "動作OK",
    statusNotWorking: "未動作",
    commonCli: {
      TEST_TEXT: "読み上げるサンプルテキスト",
      TEST_SPEED: "再生速度",
      TEST_VOICE: "声の上書き指定（任意）",
      OPENAI_MODEL_ID: "OpenAIモデルID（任意）",
    },
  },
};

function getInitialLang() {
  try {
    const saved = localStorage.getItem("local-tts-lang");
    if (saved === "en" || saved === "ja") return saved;
  } catch (e) {
    // Ignore blocked localStorage.
  }
  return (navigator.language || "").toLowerCase().startsWith("ja") ? "ja" : "en";
}

function formatText(template, vars = {}) {
  return String(template).replace(/\{(\w+)\}/g, (_, key) =>
    Object.prototype.hasOwnProperty.call(vars, key) ? String(vars[key]) : `{${key}}`
  );
}

function niceLabel(name, prefix) {
  let s = name;
  if (prefix && s.startsWith(prefix + "_")) s = s.slice(prefix.length + 1);
  if (!s) return name;
  const words = s.toLowerCase().split("_").filter(Boolean);
  if (!words.length) return name;
  const labeled = words.map((w) => (ACRONYMS[w] ? ACRONYMS[w] : w));
  let label = labeled.join(" ");
  // Capitalize the first letter unless that letter belongs to an acronym mapping.
  const firstWord = labeled[0];
  if (!/^[A-Z]/.test(firstWord) && !/^\(/.test(firstWord)) {
    label = label.charAt(0).toUpperCase() + label.slice(1);
  }
  return label;
}

function commonCliLabel(name, lang = "en") {
  return I18N[lang].commonCli[name] || niceLabel(name);
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderInlineMd(text) {
  if (!text) return "";
  let s = escapeHtml(text);
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(
    /\[([^\]]+)\]\(([^)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener">$1</a>'
  );
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  return s;
}

function statusShort(status) {
  if (!status) return "Unknown";
  const lower = status.toLowerCase();
  if (lower.startsWith("works") || lower.startsWith("working")) return "Works";
  if (lower.startsWith("not working") || lower.startsWith("not verified")) return "Not working";
  return status.split(/[(–—]/)[0].trim() || "Unknown";
}

function statusShortLabel(status, lang = "en") {
  const short = statusShort(status);
  if (short === "Works") return I18N[lang].statusWorks;
  if (short === "Not working") return I18N[lang].statusNotWorking;
  if (short === "Unknown") return I18N[lang].statusUnknown;
  return short;
}

function statusClass(status) {
  const short = statusShort(status).toLowerCase();
  if (short === "works") return "badge-ok";
  if (short === "not working") return "badge-error";
  return "badge-warn";
}

function pythonRepr(value) {
  if (value === null || value === undefined) return '""';
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "number") return String(value);
  const str = String(value);
  if (!/[\\"'\n\r\t]/.test(str)) return `"${str}"`;
  // Escape for Python double-quoted string.
  const escaped = str
    .replace(/\\/g, "\\\\")
    .replace(/"/g, '\\"')
    .replace(/\n/g, "\\n")
    .replace(/\r/g, "\\r")
    .replace(/\t/g, "\\t");
  return `"${escaped}"`;
}

function pythonStrLiteral(value) {
  if (value === null || value === undefined) return '""';
  return pythonRepr(value);
}

function buildCmdArgs(data, engine, values) {
  const args = [];
  args.push("--engine", engine.id);
  args.push("--root-dir", "/content/openai-compatible-local-tts");

  for (const p of data.common_cli) {
    if (!p.flag) continue;
    const val = values[p.name];
    args.push(p.flag, val === "" || val === undefined || val === null ? "" : String(val));
  }

  for (const p of engine.params) {
    const val = values[p.name];
    if (p.type === "boolean") {
      if (val && p.true_flag) {
        args.push(p.true_flag);
      } else if (!val && p.false_flag) {
        args.push(p.false_flag);
      }
    } else if (p.flag) {
      args.push(p.flag, val === "" || val === undefined || val === null ? "" : String(val));
    }
  }

  const expose = data.expose_setting;
  if (expose) {
    const exposeVal = values[expose.name];
    if (exposeVal && expose.true_flag) args.push(expose.true_flag);
    else if (!exposeVal && expose.false_flag) args.push(expose.false_flag);
  }

  return args;
}

function generateCell(data, engine, values) {
  if (!engine) return "";

  const argLines = buildCmdArgs(data, engine, values).map((a) => `    ${pythonStrLiteral(a)},`);
  const repoUrl = values.REPO_URL || REPO_URL + ".git";
  const repoRef = values.REPO_REF || "main";
  const workdir = values.WORKDIR || "/content/local-tts-on-google-colab";

  return [
    `#@title Local TTS on Google Colab -> OpenAI Compatible /v1/audio/speech (${engine.id})`,
    `# Generated by docs/index.html (see ${REPO_URL})`,
    "",
    `REPO_URL = ${pythonStrLiteral(repoUrl)}`,
    `REPO_REF = ${pythonStrLiteral(repoRef)}`,
    `WORKDIR  = ${pythonStrLiteral(workdir)}`,
    "",
    "import shlex",
    "import subprocess",
    "from pathlib import Path",
    "",
    "",
    "def run(cmd, *, cwd=None):",
    '    print("$", shlex.join(cmd))',
    "    proc = subprocess.Popen(",
    "        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True",
    "    )",
    "    for line in proc.stdout:",
    '        print(line, end="", flush=True)',
    "    ret = proc.wait()",
    "    if ret != 0:",
    "        raise subprocess.CalledProcessError(ret, cmd)",
    "",
    "",
    "workdir = Path(WORKDIR)",
    "if not workdir.exists():",
    '    run(["git", "clone", REPO_URL, str(workdir)])',
    "else:",
    '    print(f"reuse: {workdir}")',
    "",
    'run(["git", "fetch", "--all", "--tags", "--prune"], cwd=str(workdir))',
    'run(["git", "checkout", REPO_REF], cwd=str(workdir))',
    'if REPO_REF in {"main", "master"}:',
    '    run(["git", "pull", "--ff-only", "origin", REPO_REF], cwd=str(workdir))',
    "",
    "cmd = [",
    '    "python",',
    '    "colab/bootstrap.py",',
    ...argLines,
    "]",
    "run(cmd, cwd=str(workdir))",
    "",
  ].join("\n");
}

function colabConnectionGuidance(lang = "en") {
  if (lang === "ja") {
    return [
      "まず、現在のエージェント環境でColab MCPサーバー、またはColab操作用ツールが利用できるか確認してください。",
      "利用可能なColab接続がある場合は、ユーザーに実装方式を選ばせず、その接続を使って進めてください。",
      "利用可能なColab接続がない場合は、ローカル設定を変更する前に、ユーザーがどのColab接続を用意できるか確認してください。",
      "このリポジトリには、Codex向けに扱いやすい既定として colab-mcp-go 用の .mcp.json が含まれています。",
      "公式の googlecolab/colab-mcp は、それに対応したクライアントでは選択肢になります。",
      "ツールのインストール、MCP設定の編集、MCP実装の切り替えは、ユーザーの承認なしに行わないでください。",
    ];
  }
  return [
    "First check whether the current agent environment already has a Colab MCP server or any Colab control tool available.",
    "If a usable Colab connection is already available, use it without asking the user to choose an implementation.",
    "If no usable Colab connection is available, ask the user what Colab connection they can provide before changing local setup.",
    "This repository includes .mcp.json configured for colab-mcp-go, which is a practical default for Codex-oriented usage.",
    "The official googlecolab/colab-mcp may be appropriate in clients that support it.",
    "Do not install tools, edit MCP config, or switch MCP implementations without user approval.",
  ];
}

function generateAgentPrompt(engine, values, generatedCell, lang = "en") {
  if (!engine) return "";
  const guidance = colabConnectionGuidance(lang).map((line) => `- ${line}`).join("\n");
  const sampleText = values.TEST_TEXT || "";
  if (lang === "ja") {
    const publicUrlNote = values.EXPOSE_PUBLIC_URL
      ? "セットアップコマンドが公開URLを生成する場合は、そのURLを提示し、OpenAI互換エンドポイントとして報告してください。"
      : "ユーザーが明示的に依頼しない限り、公開URLは作成しないでください。";

    return [
      "Google Colab上でローカルのOpenAI互換TTSサーバーを起動してください。",
      "",
      "目的:",
      `- https://github.com/shinshin86/local-tts-on-google-colab から ${engine.id} エンジンをセットアップして実行する`,
      "- 可能であれば、Google ColabはMCP接続経由で操作する",
      "- 下にある生成済みのColab Pythonセルを、新しいColab scratchpad、または接続済みのColabノートブックで実行する",
      "- サーバーが正常に起動したことを確認し、簡単な /v1/audio/speech の疎通確認を実行、または手順を説明する",
      "- エンドポイントURL、モデルID、選択された声/話者、実行時の制限事項をまとめる",
      "",
      "Colab接続の扱い:",
      guidance,
      "",
      "実行時の注意:",
      "- 実行前に、無料GPUランタイム、有料ランタイム、GPUが使えない場合のCPUフォールバックのどれを使うかユーザーに確認してください。",
      "- Colabでブラウザログイン、ランタイム選択、ノートブックの信頼確認が必要な場合は、ユーザーに操作を依頼してください。",
      "- パッケージインストールやモデルダウンロードが失敗した場合は、エラーを確認し、必要最小限の修正で対応してください。",
      "- セットアップを一から書き直すのではなく、このリポジトリが生成したセルを優先して使ってください。",
      `- 疎通確認用のサンプルテキスト: ${sampleText}`,
      `- ${publicUrlNote}`,
      "",
      "実行する生成済みColabセル:",
      "```python",
      generatedCell.trimEnd(),
      "```",
      "",
    ].join("\n");
  }

  const publicUrlNote = values.EXPOSE_PUBLIC_URL
    ? "Expose a public URL if the setup command produces one, then report the OpenAI-compatible endpoint."
    : "Do not expose a public URL unless the user asks for one.";

  return [
    "You are helping me run a local OpenAI-compatible TTS server on Google Colab.",
    "",
    "Goal:",
    `- Set up and run the ${engine.id} engine from https://github.com/shinshin86/local-tts-on-google-colab`,
    "- Use Google Colab through an MCP connection when possible.",
    "- Run the generated Colab Python cell below in a fresh Colab scratchpad or an already connected Colab notebook.",
    "- Confirm the server starts successfully, then run or describe a simple /v1/audio/speech smoke test.",
    "- Summarize the endpoint URL, model ID, selected voice/speaker, and any runtime limitations.",
    "",
    "Colab connection handling:",
    guidance,
    "",
    "Operational constraints:",
    "- Before executing, confirm whether the user wants a free GPU runtime, paid runtime, or CPU fallback if GPU is unavailable.",
    "- If Colab prompts for browser sign-in, runtime selection, or notebook trust, ask the user to complete it.",
    "- If package installation or model download fails, inspect the error and make the smallest fix needed.",
    "- Prefer the repository's generated cell over rewriting the setup from scratch.",
    `- Sample text for the smoke test: ${sampleText}`,
    `- ${publicUrlNote}`,
    "",
    "Generated Colab cell to execute:",
    "```python",
    generatedCell.trimEnd(),
    "```",
    "",
  ].join("\n");
}

function defaultValues(data) {
  const values = {};
  for (const p of data.repo_settings) values[p.name] = p.default;
  values[data.engine_selector.name] = data.engine_selector.default;
  values[data.expose_setting.name] = data.expose_setting.default;
  for (const p of data.common_cli) values[p.name] = p.default;
  for (const e of data.engines) {
    for (const p of e.params) {
      if (!(p.name in values)) values[p.name] = p.default;
    }
  }
  return values;
}

createApp({
  setup() {
    const data = ref(null);
    const copyState = ref("");
    const agentCopyState = ref("");
    const state = reactive({ engineId: "Kokoro", lang: getInitialLang(), values: {} });

    function t(key, vars = {}) {
      const text = I18N[state.lang][key] ?? I18N.en[key] ?? key;
      return formatText(text, vars);
    }

    function setLang(lang) {
      if (lang !== "en" && lang !== "ja") return;
      state.lang = lang;
    }

    function syncDocumentLanguage(lang) {
      document.documentElement.lang = lang;
      document.title = I18N[lang].pageTitle;
      const meta = document.querySelector('meta[name="description"]');
      if (meta) meta.setAttribute("content", I18N[lang].pageDescription);
    }

    onMounted(async () => {
      syncDocumentLanguage(state.lang);
      const res = await fetch("./engines.json");
      if (!res.ok) {
        console.error("Failed to load engines.json:", res.status);
        return;
      }
      const json = await res.json();
      data.value = json;
      state.engineId = json.engine_selector.default;
      state.values = defaultValues(json);
    });

    watch(
      () => state.lang,
      (lang) => {
        syncDocumentLanguage(lang);
        try {
          localStorage.setItem("local-tts-lang", lang);
        } catch (e) {
          // Ignore blocked localStorage.
        }
      }
    );

    const selectedEngine = computed(() => {
      if (!data.value) return null;
      return data.value.engines.find((e) => e.id === state.engineId) || null;
    });

    const primaryParam = computed(() => {
      const e = selectedEngine.value;
      if (!e || !e.params.length) return null;
      // Heuristic: prefer the most user-facing voice/speaker/model selector.
      const selects = e.params.filter((p) => p.type === "select");
      if (!selects.length) return null;
      const preference = [
        (p) => /_DEFAULT_VOICE$/.test(p.name),
        (p) => /_DEFAULT_SPEAKER$/.test(p.name),
        (p) => /_VOICE$/.test(p.name),
        (p) => /_DEFAULT_LANG/.test(p.name),
        (p) => /_LANGUAGE$/.test(p.name),
        (p) => /_HF_CHECKPOINT$/.test(p.name),
        (p) => /_MODEL$/.test(p.name),
      ];
      for (const pred of preference) {
        const hit = selects.find(pred);
        if (hit) return hit;
      }
      return selects[0];
    });

    const primaryLabel = computed(() => {
      const p = primaryParam.value;
      if (!p) return "";
      if (/_VOICE$/.test(p.name)) return t("voiceSpeaker");
      if (/_SPEAKER$/.test(p.name)) return t("speaker");
      if (/_LANGUAGE$/.test(p.name) || /_LANG$/.test(p.name)) return t("language");
      if (/_CHECKPOINT$/.test(p.name) || /_MODEL$/.test(p.name)) return t("model");
      return niceLabel(p.name);
    });

    const advancedEngineParams = computed(() => {
      const e = selectedEngine.value;
      if (!e) return [];
      const primary = primaryParam.value;
      return e.params.filter((p) => !primary || p.name !== primary.name);
    });

    const hasEngineDetails = computed(() => {
      const e = selectedEngine.value;
      if (!e) return false;
      return Boolean(e.status) || Boolean(e.description) || (e.notes && e.notes.length > 0);
    });

    const hasLicenseInNotes = computed(() => {
      const e = selectedEngine.value;
      if (!e) return false;
      const blob = (e.notes || []).join(" ") + " " + (e.description || "");
      return /licens|apache|mit\b|cc[-\s]?by|community license|openrail|noncommercial|non[-\s]?commercial/i.test(
        blob
      );
    });

    const generatedCell = computed(() =>
      data.value ? generateCell(data.value, selectedEngine.value, state.values) : ""
    );

    const generatedLines = computed(() => generatedCell.value.split("\n").length);

    const generatedAgentPrompt = computed(() =>
      data.value
        ? generateAgentPrompt(selectedEngine.value, state.values, generatedCell.value, state.lang)
        : ""
    );

    const generatedAgentPromptLines = computed(() => generatedAgentPrompt.value.split("\n").length);

    async function copyText(text, copyStateRef, fallbackMessage) {
      try {
        await navigator.clipboard.writeText(text);
        copyStateRef.value = "copied";
        setTimeout(() => (copyStateRef.value = ""), 1800);
      } catch (e) {
        // Fallback: select-and-copy via a temporary textarea.
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        try {
          document.execCommand("copy");
          copyStateRef.value = "copied";
          setTimeout(() => (copyStateRef.value = ""), 1800);
        } catch (err) {
          alert(fallbackMessage);
        }
        document.body.removeChild(ta);
      }
    }

    function copyCell() {
      return copyText(
        generatedCell.value,
        copyState,
        t("copyCellFallback")
      );
    }

    function copyAgentPrompt() {
      return copyText(
        generatedAgentPrompt.value,
        agentCopyState,
        t("copyPromptFallback")
      );
    }

    function downloadCell() {
      const engine = selectedEngine.value;
      const filename = `colab_cell_${(engine ? engine.id : "tts")}.py`.toLowerCase().replace(/[^a-z0-9._-]+/g, "_");
      const blob = new Blob([generatedCell.value], { type: "text/x-python" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }

    function resetDefaults() {
      if (!data.value) return;
      const fresh = defaultValues(data.value);
      Object.keys(state.values).forEach((k) => delete state.values[k]);
      Object.assign(state.values, fresh);
      state.engineId = data.value.engine_selector.default;
    }

    function engineLabel(engine) {
      const tag = statusShort(engine.status);
      return tag && tag !== "Works" ? `${engine.id} — ${statusShortLabel(engine.status, state.lang)}` : engine.id;
    }

    function lineMeta(count, kind) {
      const label = state.lang === "ja" && kind === "Prompt" ? "プロンプト" : kind;
      return t("linesMeta", { count, kind: label });
    }

    return {
      data,
      state,
      selectedEngine,
      primaryParam,
      primaryLabel,
      advancedEngineParams,
      hasEngineDetails,
      hasLicenseInNotes,
      generatedCell,
      generatedLines,
      generatedAgentPrompt,
      generatedAgentPromptLines,
      copyState,
      agentCopyState,
      repoUrl: REPO_URL,
      copyCell,
      copyAgentPrompt,
      downloadCell,
      resetDefaults,
      setLang,
      t,
      lineMeta,
      niceLabel,
      commonCliLabel,
      renderInlineMd,
      statusShort,
      statusShortLabel: (status) => statusShortLabel(status, state.lang),
      statusClass,
      engineLabel,
    };
  },
}).mount("#app");
