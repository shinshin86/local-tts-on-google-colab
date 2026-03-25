# local-tts-on-google-colab

Google Colab 上で選択したローカル TTS を一時的に OpenAI 互換 `/v1/audio/speech` として起動し、動作確認できるようにするためのサンプルです。

対象エンジン:

| エンジン | Colab 動作確認 | 言語 |
|---|---|---|
| Kokoro | 動作OK | 日本語 / 英語 / 中国語 他 |
| Irodori-TTS | 動作OK | 日本語 |
| Piper | 動作OK | 英語（デフォルト）/ 多言語 |
| MeloTTS | 動作不可 | - |
| Style-Bert-VITS2 | 動作不可 | - |

`MeloTTS` と `Style-Bert-VITS2` は Colab の uv + venv 環境で依存解決に問題があり、現時点では動作しません。詳細は [COLAB_TEST_RESULTS.md](COLAB_TEST_RESULTS.md) を参照してください。

`VOICEVOX` は含めていません。

## 使い方

1. [multi_tts_openai_colab.py](multi_tts_openai_colab.py) の内容を Colab の1つのコードセルに貼り付けます。
2. フォーム上で `ENGINE` を選びます。
3. 必要なら各エンジンのパラメータを調整します。
4. `TEST_VOICE` を空にすると、そのエンジンの既定 voice を使ってテストします。
5. 起動後に `/v1/voices` の一覧が自動表示されるので、必要ならそれを見て `voice` を選び直します。
6. セルを実行します。
7. ローカル URL と、必要なら `trycloudflare` の公開 URL が表示されます。

この実装は「1ランタイムで1エンジンずつ」の運用を前提にしています。別エンジンを試すときは、ランタイムを再起動してから再実行する想定です。

## OpenAI 互換の範囲

対応エンドポイント:

- `GET /`
- `GET /v1/models`
- `GET /v1/voices`
- `POST /v1/audio/speech`

互換対象の主な入力:

- `model`
- `input`
- `voice`
- `speed`
- `response_format`

このサンプルは `wav` 固定です。`mp3` などへの変換は行っていません。

## エンジンごとの補足

### Kokoro

[hexgrad/kokoro](https://github.com/hexgrad/kokoro) を使った日本語・英語・中国語対応の軽量 TTS です。デフォルト voice は日本語の `jf_alpha` で、フォームから 9 種類の voice を選べます。

### Irodori-TTS

[Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) を使った日本語 TTS です。デフォルトで Hugging Face の `Aratako/Irodori-TTS-500M` モデルを使用します。出力は 48kHz で高音質ですが、voice の切り替え機能はありません。

### Piper

[piper-tts](https://github.com/OHF-Voice/piper1-gpl) の内蔵 HTTP サーバーをバックエンドとして起動し、その前段に OpenAI 互換ラッパーを載せています。デフォルトは英語の `en_US-lessac-medium` です。依存が軽く、セットアップが安定しています。

### MeloTTS (現在動作不可)

[myshell-ai/MeloTTS](https://github.com/myshell-ai/MeloTTS) を使う構成ですが、依存パッケージ `tokenizers` のビルドに Rust コンパイラが必要なため、現在の Colab 環境ではインストールに失敗します。

### Style-Bert-VITS2 (現在動作不可)

[litagin02/Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) を使う構成ですが、`setuptools` / `torch` / `scipy` の依存整合性が取れず、現在の Colab 環境では音声合成まで到達できません。

## 注意点

- Colab の管理ランタイムでは、外部公開やプロキシ利用は恒常運用向きではありません。このリポジトリは短時間の動作確認用です。
- エンジンごとに依存が重いため、別エンジンへの切り替えはランタイム再起動前提にしています。
- `Irodori-TTS` の倫理制限、各音声モデルのライセンスは個別に確認してください。

## Colab 実機テスト結果

詳細は [COLAB_TEST_RESULTS.md](COLAB_TEST_RESULTS.md) を参照してください。

## 参考

- OpenAI Audio Speech API
  https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create
- Irodori-TTS
  https://github.com/Aratako/Irodori-TTS
- Kokoro
  https://github.com/hexgrad/kokoro
- MeloTTS
  https://github.com/myshell-ai/MeloTTS
- Style-Bert-VITS2
  https://github.com/litagin02/Style-Bert-VITS2
- Piper
  https://github.com/OHF-Voice/piper1-gpl
