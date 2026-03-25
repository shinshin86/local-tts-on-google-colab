# Colab MCP セットアップ

このリポジトリでは、[googlecolab/colab-mcp](https://github.com/googlecolab/colab-mcp) を Claude Code の project-scoped MCP として設定しています。

これにより、Claude Code から Google Colab のノートブックを操作（セルの読み書き・実行）できます。

## 仕組み

`.mcp.json` に以下が定義されています。

```json
{
  "mcpServers": {
    "colab-mcp": {
      "command": "uvx",
      "args": ["--python", "3.13", "git+https://github.com/googlecolab/colab-mcp"]
    }
  }
}
```

Claude Code がこのリポジトリで起動すると、`uvx` 経由で `colab-mcp` サーバーが自動的に立ち上がります。

## 必要な前提

- `uv` がローカルに入っていること（`uvx` コマンドが使える）
- `git` が使えること
- Google Colab をブラウザで開けること
- Google アカウントで Colab にログイン済みであること

## 接続フロー

`colab-mcp` は、単に Colab の URL を渡す方式ではありません。以下の流れで接続します。

1. Claude Code が `colab-mcp` を起動する
2. `colab-mcp` がローカルに一時 WebSocket サーバーを立てる（一時トークンとポートが発行される）
3. `open_colab_browser_connection` ツールが呼ばれると、ブラウザで接続用の Colab URL が開く
4. Colab フロントエンドがローカル WebSocket サーバーへ接続する
5. 接続が成立すると、ノートブック操作ツール（セル読み書き・実行・削除）が使えるようになる

## 使い方

1. Colab でノートブックを開く
2. [multi_tts_openai_colab.py](multi_tts_openai_colab.py) の内容を 1 セルに貼る
3. Claude Code をこのリポジトリで起動する
4. project-scoped MCP の利用承認を求められたら承認する
5. Claude Code 内で `/mcp` を開き、`colab-mcp` が接続済みか確認する
6. 接続できたら、以降の Colab 操作はエージェント側で進められる

## proxy tools と runtime tools

`colab-mcp` には 2 つのモードがあります。

| モード | デフォルト | 機能 |
|---|---|---|
| proxy tools | 有効 | Colab ブラウザ UI 経由でノートブックを操作 |
| runtime tools | 無効 | Colab ランタイムに直接コードを実行（`--enable-runtime` + OAuth 認証が必要） |

このリポジトリでは proxy tools のみを使っています。

## 補足

- `.python-version` は `uv` が使う Python バージョン（3.13）を明示するために置いています。Colab ランタイムの Python とは別です
- ローカルに Python 3.13 が未導入でも、`uv` が自動で解決します
- MCP の承認状態をリセットしたいときは `claude mcp reset-project-choices` を使います
- `open_colab_browser_connection` は内部で `webbrowser.open_new(...)` を呼ぶため、ブラウザが開けるローカル環境で使う前提です
