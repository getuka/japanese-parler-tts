import torch
from torch import dtype
from numpy import ndarray
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer
from threading import Thread
from rubyinserter import add_ruby
from typing import Iterator, Tuple


class TTS:
    """
    ParlerTTS を利用した TTS クラス。
    テキストを音声として生成する。
    """

    def __init__(
        self,
        model_name: str = "2121-8/japanese-parler-tts-mini-bate",
        torch_dtype: dtype = torch.bfloat16,
        attn_implementation: str = "eager",
        compile_mode: str = "default"
    ) -> None:
        """
        コンストラクタ。モデル、トークナイザーを初期化する

        Args:
            model_name (str): 使用するモデル名
            torch_dtype (dtype): モデルのデータ型
            attn_implementation (str): 注意メカニズムの実装方法
            compile_mode (str): コンパイルモード
        """
        self.torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # モデルとトークナイザーのロード
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name, attn_implementation=attn_implementation
        ).to(self.torch_device, dtype=torch_dtype)

        # モデルのコンパイル
        self.model.forward = torch.compile(
            self.model.forward, mode=compile_mode
        )

        # オーディオ設定
        self.sampling_rate = self.model.audio_encoder.config.sampling_rate
        self.frame_rate = self.model.audio_encoder.config.frame_rate

    def _prepare_inputs(self, text: str, description: str) -> dict:
        """
        テキストとプロンプトをトークナイズして入力データを準備する

        Args:
            text (str): 入力テキスト
            description (str): プロンプト

        Returns:
            dict: モデルに渡す入力データ
        """
        description_tokens = self.tokenizer(
            description, return_tensors="pt").to(self.torch_device)
        text_tokens = self.tokenizer(
            add_ruby(text), return_tensors="pt").to(self.torch_device)

        return {
            "input_ids": description_tokens.input_ids,
            "prompt_input_ids": text_tokens.input_ids,
            "attention_mask": description_tokens.attention_mask,
            "prompt_attention_mask": text_tokens.attention_mask,
        }

    def generate(self, text: str, description: str) -> Tuple[int, ndarray]:
        """
        テキストを音声データとして生成する

        Args:
            text (str): 入力テキスト
            description (str): プロンプト

        Returns:
            Tuple[int, ndarray]: サンプリングレートと生成された音声データ
        """
        inputs = self._prepare_inputs(text, description)
        generation = self.model.generate(
            **inputs, do_sample=True, temperature=1.0, min_new_tokens=10
        )
        audio_data = generation.to(torch.float32).cpu().numpy().squeeze()
        return self.sampling_rate, audio_data

    def stream(self, text: str, description: str, play_steps_in_s: float = 0.5) -> Iterator[Tuple[int, ndarray]]:
        """
        テキストを音声ストリームとして生成する

        Args:
            text (str): 入力テキスト
            description (str): プロンプト
            play_steps_in_s (float): 再生ステップの長さ（秒単位）

        Yields:
            Tuple[int, ndarray]: サンプリングレートと生成された音声データ
        """
        inputs = self._prepare_inputs(text, description)
        play_steps = int(self.frame_rate * play_steps_in_s)
        streamer = ParlerTTSStreamer(
            self.model, device=self.torch_device, play_steps=play_steps
        )

        thread = Thread(target=self.model.generate, kwargs={
                        **inputs, "streamer": streamer})
        thread.start()

        for new_audio in streamer:
            if new_audio.shape[0] == 0:
                break
            yield self.sampling_rate, new_audio