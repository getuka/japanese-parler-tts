# Japanese Parler-TTS

Japanese Parler-TTS用のOpenAI互換APIサーバーを構築するリポジトリです。

## Dockerを使用した構築手順 (推奨)

1. **リポジトリのクローン**

   ```bash
   git clone https://github.com/getuka/japanese-parler-tts.git
   cd japanese-parler-tts
   ```

3. **コンテナの起動**

   ```bash
   docker compose up -d
   ```

   サーバーは`http://localhost:8000`でアクセス可能です。


## ローカル環境での構築手順

1. **リポジトリのクローン**

   ```bash
   git clone https://github.com/getuka/japanese-parler-tts.git
   cd japanese-parler-tts
   ```

2. **依存関係のインストール**

   ```bash
   pip3 install torch torchvision torchaudio
   pip install packaging ninja
   pip install -r requirements.txt
   ```

3. **サーバーの起動**

   ```bash
   uvicorn server.main:app --host 0.0.0.0 --port 8000
   ```

   サーバーは`http://localhost:8000`でアクセス可能です。


## 使用方法

1. **APIエンドポイント**

   サーバーは以下のエンドポイントを提供します：

   - **`/v1/models`**：使用できるmodelの一覧を返します
   - **`/v1/audio/speech`**：テキストを音声に変換します。

2. **音声合成のリクエスト**

   以下のようなJSONペイロードを使用して、`/synthesize`エンドポイントにPOSTリクエストを送信します：

   ```cmd
   curl http://0.0.0.0:8000/v1/audio/speech \
      -H "Authorization: Bearer token-123" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "2121-8/japanese-parler-tts-mini-bate",
        "input": "こんにちは。お元気ですか？",
        "voice": "A female speaker with a slightly high-pitched voice delivers her words at a moderate speed with a quite monotone tone in a confined environment, resulting in a quite clear audio recording.",
        "response_format": "wav"
      }' \
    --output speech.wav
   ```
   ```python
   from pathlib import Path
   from openai import OpenAI
   client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="token-123")
    
   speech_file_path = "speech.wav"
   response = client.audio.speech.create(
       model="2121-8/japanese-parler-tts-mini-bate",
       voice="A female speaker with a slightly high-pitched voice delivers her words at a moderate speed with a quite monotone tone in a confined environment, resulting in a quite clear audio recording.",
       input="こんにちは、今日はどのようにお過ごしですか？",
       response_format="wav",
       speed=1,
   )
    
   response.stream_to_file(speech_file_path)
   ```

   - **`text`**：生成する日本語テキスト
   - **`model`**：使用するモデル
   - **`voice`**：読み上げ制御用のプロンプト
   - **`response_format`**：レスポンスの音声フォーマット (オプション)

3. **レスポンス**

   リクエストが成功すると、指定した音声フォーマットでレスポンスを返します。

## 注意事項

- このプロジェクトは、Parler-TTSモデルを日本語で利用するための実験的な実装です。
- 本リポジトリおよびモデルの作成者は、著作権侵害やその他の法的問題に関する責任を一切負いません。
- 本モデルの利用により得られる結果の正確性、合法性、または適切性について、作成者は一切保証しません。

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は`LICENSE`ファイルをご参照ください。
