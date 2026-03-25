# Colab 実機テスト結果

Google Colab 上で `multi_tts_openai_colab.py` を使い、各エンジンの動作確認を行った結果です。

テスト日: 2026-03-25
Colab ランタイム: Python 3.12 (CPU)

## 結果一覧

| エンジン | 結果 | 公開 URL テスト | WAV 形式 |
|---|---|---|---|
| Kokoro | 成功 | 200 OK, 226,844 bytes | PCM 16bit, mono, 24000Hz |
| Irodori-TTS | 成功 | 200 OK, 572,238 bytes | PCM 16bit, mono, 48000Hz |
| MeloTTS | 動作不可 | - | - |
| Style-Bert-VITS2 | 動作不可 | - | - |
| Piper | 成功 | 200 OK, 146,476 bytes | PCM 16bit, mono, 22050Hz |

## 動作確認済みエンジン

### Kokoro

- モデル: `kokoro-82m` (hexgrad/Kokoro-82M)
- デフォルト voice: `jf_alpha` (日本語女性)
- セットアップ時間: 数分（pip install + モデルダウンロード）
- 補足: `unidic` と `unidic download`（`misaki[ja]` の日本語処理に必要）、`espeak-ng` の apt インストールは、スクリプトの `install_kokoro()` に含まれており、自動でセットアップされます。

### Irodori-TTS

- モデル: `Aratako/Irodori-TTS-500M`
- voice 切り替え: なし（参照音声なしモード）
- セットアップ時間: 数分（git clone + uv sync + dacvae + モデルダウンロード）
- 補足: `uv sync` でプロジェクトの依存を解決するため、比較的スムーズにセットアップできます。出力は 48kHz で他エンジンより高音質です。

### Piper

- モデル/Voice: `en_US-lessac-medium` (英語)
- セットアップ時間: 1-2分（pip install + voice ダウンロード）
- 補足: Piper は内蔵 HTTP サーバー（Flask）をバックエンドとして起動し、その前段に OpenAI 互換プロキシを載せる構成です。依存が軽く、セットアップが安定しています。日本語モデルを使いたい場合は `PIPER_VOICE` を変更してください。

## 動作しなかったエンジン

### MeloTTS

セットアップ段階で失敗します。

原因:
1. MeloTTS が `transformers==4.27.4` に依存 → `tokenizers==0.13.3` が要求される
2. `tokenizers==0.13.3` にはビルド済みホイールがなく、ソースビルドに **Rust コンパイラ**が必要
3. `fugashi` のビルドに `libmecab-dev` (apt) が必要
4. `--no-deps` で回避を試みても `pykakasi` 等の未宣言依存が多数不足

MeloTTS 側の依存管理（特に transformers のバージョン固定）が改善されるまで、Colab + uv + venv 環境では動作しません。

### Style-Bert-VITS2

セットアップは通るがサーバー起動後に音声合成が失敗します。

原因:
1. `pyopenjtalk` が `pkg_resources` に依存 → `setuptools>=82` で削除済み → `setuptools<81` が必要
2. `-e .` インストールで `torch` が依存に含まれない → `transformers` が PyTorch を認識できない
3. `scipy` も別途インストールが必要

手動で個別に修正しても、次々と新しい依存問題が出るモグラ叩き状態になります。

## テストで使用した構成

- `multi_tts_openai_colab.py` を Colab の1セルに貼り付けて実行
- `EXPOSE_PUBLIC_URL = True` で `cloudflared` による公開 URL を取得
- 公開 URL に対して `POST /v1/audio/speech` で WAV が返ることを確認
- 各エンジンは別ランタイムで個別にテスト（1ランタイム1エンジン）

## テスト済みエンドポイント

動作確認済みの全エンジンで以下が正常に応答しました:

- `GET /` - エンジン情報
- `GET /v1/models` - モデル一覧
- `GET /v1/voices` - voice 一覧
- `POST /v1/audio/speech` - 音声合成（WAV）

## 新しくこのリポジトリを使う方へ

1. Google Colab でノートブックを開きます
2. `multi_tts_openai_colab.py` の内容を1つのコードセルに貼り付けます
3. フォームで `ENGINE` を選びます（まずは **Kokoro** か **Piper** が安定しています）
4. セルを実行します
5. セットアップが完了すると、ローカル URL と公開 URL が表示されます
6. 公開 URL に対して `curl` で音声合成を試せます

推奨の試行順:
1. **Kokoro** - 日本語 TTS、複数 voice 対応、安定動作
2. **Irodori-TTS** - 日本語 TTS、高音質（48kHz）、voice 切り替えなし
3. **Piper** - 英語 TTS がデフォルト、軽量で安定

別エンジンに切り替えるときは、**ランタイムを再起動してからセルを再実行**してください。
