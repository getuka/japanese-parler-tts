from fastapi import Header, FastAPI, HTTPException, Response, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from enum import Enum
import soundfile as sf
import torch
from server.tts import TTS
from typing import Optional
from openai.types import Model
import os
import yaml


# 設定クラス
# 設定クラス
class Config:
    # デフォルト設定
    MODEL_NAME = None
    VALID_API_KEYS = None
    TORCH_DTYPE = None

    DESCRIPTION_PROMPT = "A female speaker with a slightly high-pitched voice delivers her words at a moderate speed with a quite monotone tone in a confined environment, resulting in a quite clear audio recording."
    ATTN_IMPLEMENTATION = "eager"
    COMPILE_MODE = "default"
    OUTPUT_FILE = "out.wav"

    # 音声フォーマット設定
    class ResponseFormat(str, Enum):
        OPUS = "opus"
        AAC = "aac"
        FLAC = "flac"
        WAV = "wav"
        PCM = "pcm"
        MP3 = "mp3"

    @classmethod
    def load_from_yaml(cls, yaml_path: str):
        """
        YAML ファイルから設定を読み込むメソッド。

        Args:
            yaml_path (str): 設定ファイルのパス
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {yaml_path}")

        with open(yaml_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)

        # 設定を反映
        cls.MODEL_NAME = config_data.get('model_name', cls.MODEL_NAME)
        cls.TORCH_DTYPE = getattr(torch, config_data.get('torch_dtype', 'float16'))
        cls.VALID_API_KEYS = config_data.get('api_keys', cls.VALID_API_KEYS)

# 音声生成リクエスト
class AudioRequest(BaseModel):
    input: str = Query(None)
    model: str = Config.MODEL_NAME
    voice: str = Config.DESCRIPTION_PROMPT
    response_format: Config.ResponseFormat = Config.ResponseFormat.WAV
    speed: float = 1.0


# FastAPI アプリの初期化
app = FastAPI()

# YAML 設定ファイルのパス
CONFIG_FILE_PATH = "config.yaml"

# 設定をロード
Config.load_from_yaml(CONFIG_FILE_PATH)
print(Config.MODEL_NAME,
      Config.TORCH_DTYPE,
      Config.ATTN_IMPLEMENTATION,
      Config.COMPILE_MODE,)

# TTS モジュールの初期化
tts = TTS(
    model_name=Config.MODEL_NAME,
    torch_dtype=Config.TORCH_DTYPE,
    attn_implementation=Config.ATTN_IMPLEMENTATION,
    compile_mode=Config.COMPILE_MODE,
)
print("TTS モジュールの準備完了！")


# 資格認証用の関数
def authentication(audio_request: AudioRequest, authorization: Optional[str]):
    """
    API認証を行う関数。

    Args:
        audio_request (AudioRequest): リクエストに含まれるモデル情報
        authorization (Optional[str]): 認証用のAuthorizationヘッダー

    Raises:
        HTTPException: 認証に失敗した場合に対応したエラーを送信
    """
    # Authorizationヘッダーから"Bearer "を取り除いてAPIキーを取得
    # API Keyが取得できなかったら401を返す
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid Authentication: API key is missing."
        )
    api_key = authorization.replace("Bearer ", "").strip()

    # API Keyが登録済みで一致しなかった場合、401を返す
    if (Config.VALID_API_KEYS is not None) and (api_key not in Config.VALID_API_KEYS):
        raise HTTPException(
            status_code=401,
            detail="Incorrect API key provided: Ensure the API key is correct."
        )

    # モデルを一致しなかった場合、401を返す
    if audio_request.model != Config.MODEL_NAME:
        raise HTTPException(
            status_code=401,
            detail="You must use a valid model to access the API."
        )


@app.get("/health")
async def health() -> Response:
    """
    システムのヘルスチェック
    """
    return Response(status_code=200, content="OK")


@app.get("/v1/models", response_model=list[Model])
def get_models() -> Response:

    return Response(status_code=200, content={
        "data": {
            "id": Config.MODEL_NAME,
            "object": "model",
        }
    })


@app.post("/v1/audio/speech", summary="TTSを使用して音声合成をする")
async def generate_speech(audio_request: AudioRequest, authorization: Optional[str] = Header(None)) -> FileResponse:
    """
    入力文章を音声に変換して返すエンドポイント

    Args:
        audio_request (AudioRequest): 音声リクエストの内容
        authorization (Optional[str]): 認証情報 (Bearer トークン)

    Returns:
        FileResponse: 生成された音声ファイルを返却

    Raises:
        HTTPException: 認証または処理に失敗した場合のエラー
    """
    try:
        print(f"認証情報: {authorization}")
        print(f"リクエスト受信: {audio_request}")

        authentication(audio_request, authorization)

        # 音声生成
        sampling_rate, audio = tts.generate(
            text=audio_request.input,
            description=audio_request.voice,
        )

        # 音声ファイルに書き込み
        # 音声ファイルに書き込み
        sf.write(Config.OUTPUT_FILE, audio, sampling_rate, format=audio_request.response_format)

        # 音声ファイルをレスポンスとして返す
        return FileResponse(
            path=Config.OUTPUT_FILE,
            media_type=f"audio/{audio_request.response_format}",
            filename=f"out.{audio_request.response_format}",
        )

    except HTTPException as e:
        # エラーハンドリング
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"音声生成エラー: {str(e)}")
