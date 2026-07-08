# -*- coding: utf-8 -*-
"""
Qwen-Image-Edit パイプラインの構築・生成処理。

poc/generate_qwen_edit.py を関数化したもの。モジュールレベルのシングルトンとして
遅延ロード(初回生成リクエスト時にロードし、以後プロセス内に常駐)する。
"""
import threading
import time

import torch
from accelerate import init_empty_weights
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImageTransformer2DModel,
)
# 注意: 2509 以降の Edit モデル + Multiple-angles LoRA は Plus 条件付けで学習されている。
# QwenImageEditPipeline(旧形式)だとキャラクター同一性が崩れる(別人が生成される)。
# ComfyUI ワークフローの TextEncodeQwenImageEditPlus に対応するのはこちら。
from diffusers import QwenImageEditPlusPipeline
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from prompts import NEGATIVE_PROMPT

TRANSFORMER_PATH = "/home/animede/ComfyUI/models/diffusion_models/qwen_image_edit_2511_bf16.safetensors"
LORA_LIGHTNING_PATH = "/home/animede/ComfyUI/models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
LORA_ANGLES_PATH = "/home/animede/ComfyUI/models/loras/Qwen-Edit-2509-Multiple-angles.safetensors"

# フォールバック用(2511 transformer と Multiple-angles LoRA が非互換だった場合)
FALLBACK_TRANSFORMER_PATH = "/home/animede/ComfyUI/models/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors"
FALLBACK_PREFIX = "model.diffusion_model."

SHIFT = 3.0
NUM_INFERENCE_STEPS = 4
TRUE_CFG_SCALE = 1.0
TARGET_PIXELS = 1024 * 1024  # ComfyUI ImageScaleToTotalPixels(1.0 megapixel) 相当
VRAM_FREE_THRESHOLD_GB = 65.0

_pipe = None
_pipe_lock = threading.Lock()
_load_info = {"cpu_offload": False, "fallback_transformer": False, "load_time_s": None}


def _free_vram_gb() -> float:
    """現在の空き VRAM(GB)を取得。CUDA が使えない場合は 0 を返す。"""
    if not torch.cuda.is_available():
        return 0.0
    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    return free_bytes / (1024 ** 3)


def _load_transformer():
    """transformer をローカル ComfyUI ファイルから読み込む。
    まず 2511 bf16 を試し、LoRA 適用時に非互換エラーが出た場合は
    呼び出し側で 2509 fp8 にフォールバックする。
    """
    config = QwenImageTransformer2DModel.load_config("Qwen/Qwen-Image-Edit-2509", subfolder="transformer")
    with init_empty_weights():
        transformer = QwenImageTransformer2DModel.from_config(config)
    raw = load_file(TRANSFORMER_PATH, device="cuda:0" if torch.cuda.is_available() else "cpu")
    raw.pop("__index_timestep_zero__", None)
    raw = {k: v.to(torch.bfloat16) for k, v in raw.items()}
    transformer.load_state_dict(raw, strict=True, assign=True)
    del raw
    transformer.eval()
    return transformer


def _load_transformer_fallback():
    """2509 fp8 transformer をフォールバックとして読み込む(prefix 除去 + bf16 キャスト)。"""
    config = QwenImageTransformer2DModel.load_config("Qwen/Qwen-Image-Edit-2509", subfolder="transformer")
    with init_empty_weights():
        transformer = QwenImageTransformer2DModel.from_config(config)
    raw = load_file(FALLBACK_TRANSFORMER_PATH, device="cuda:0" if torch.cuda.is_available() else "cpu")
    raw.pop("__index_timestep_zero__", None)
    stripped = {
        (k[len(FALLBACK_PREFIX):] if k.startswith(FALLBACK_PREFIX) else k): v.to(torch.bfloat16)
        for k, v in raw.items()
    }
    del raw
    transformer.load_state_dict(stripped, strict=True, assign=True)
    del stripped
    transformer.eval()
    return transformer


def _build_pipeline(transformer):
    proc_dir = snapshot_download(repo_id="Qwen/Qwen-Image-Edit-2509", allow_patterns=["processor/*"])
    processor = AutoProcessor.from_pretrained(proc_dir, subfolder="processor")

    vae = AutoencoderKLQwenImage.from_pretrained("Qwen/Qwen-Image", subfolder="vae", torch_dtype=torch.bfloat16)
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen-Image", subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen-Image", subfolder="tokenizer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("Qwen/Qwen-Image", subfolder="scheduler")

    pipe = QwenImageEditPlusPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
        transformer=transformer,
    )
    return pipe


def _apply_loras(pipe):
    """Lightning + Multiple-angles LoRA を両方適用して有効化する。"""
    pipe.load_lora_weights(LORA_LIGHTNING_PATH, adapter_name="lightning")
    pipe.load_lora_weights(LORA_ANGLES_PATH, adapter_name="angles")
    pipe.set_adapters(["lightning", "angles"], adapter_weights=[1.0, 1.0])


def _load_pipeline_locked():
    """pipe をロードする(呼び出し側で _pipe_lock を保持していること)。"""
    global _pipe
    t0 = time.time()

    free_gb = _free_vram_gb()
    use_cpu_offload = free_gb < VRAM_FREE_THRESHOLD_GB
    print(f"[pipeline] free VRAM: {free_gb:.1f} GB -> cpu_offload={use_cpu_offload}")

    fallback_used = False
    try:
        transformer = _load_transformer()
        pipe = _build_pipeline(transformer)
        if use_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        _apply_loras(pipe)
    except Exception as exc:  # noqa: BLE001 - 2511 + angles LoRA 非互換時のフォールバック
        print(f"[pipeline] primary transformer/LoRA failed ({exc!r}); falling back to 2509 fp8")
        fallback_used = True
        transformer = _load_transformer_fallback()
        pipe = _build_pipeline(transformer)
        if use_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        _apply_loras(pipe)

    pipe.scheduler.config["shift"] = SHIFT

    _load_info["cpu_offload"] = use_cpu_offload
    _load_info["fallback_transformer"] = fallback_used
    _load_info["load_time_s"] = time.time() - t0
    print(f"[pipeline] loaded in {_load_info['load_time_s']:.1f}s (fallback={fallback_used})")

    _pipe = pipe
    return _pipe


def get_pipeline():
    """シングルトンパイプラインを取得。未ロードならスレッドセーフにロードする。"""
    global _pipe
    if _pipe is not None:
        return _pipe
    with _pipe_lock:
        if _pipe is None:
            _load_pipeline_locked()
    return _pipe


def get_load_info() -> dict:
    return dict(_load_info)


def preprocess_image(image: Image.Image, target_pixels: int = TARGET_PIXELS) -> Image.Image:
    """ComfyUI の ImageScaleToTotalPixels(1.0 megapixel) 相当。
    アスペクト比を維持したまま総画素数 ≒ target_pixels になるようリサイズし、
    幅・高さを16の倍数に丸める。
    """
    image = image.convert("RGB")
    w, h = image.size
    if w <= 0 or h <= 0:
        raise ValueError("invalid image size")

    scale = (target_pixels / (w * h)) ** 0.5
    new_w = max(16, round(w * scale / 16) * 16)
    new_h = max(16, round(h * scale / 16) * 16)

    if (new_w, new_h) != (w, h):
        image = image.resize((new_w, new_h), Image.LANCZOS)
    return image


def generate_view(
    image: Image.Image,
    prompt: str,
    seed: int = 0,
    negative_prompt: str = NEGATIVE_PROMPT,
) -> Image.Image:
    """1方向分の画像を生成する。"""
    pipe = get_pipeline()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
        image=[image],
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        true_cfg_scale=TRUE_CFG_SCALE,
        generator=generator,
    )
    return result.images[0]
