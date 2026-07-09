# -*- coding: utf-8 -*-
"""
Qwen-Image-Edit パイプラインの構築・生成処理。

poc/generate_qwen_edit.py を関数化したもの。モジュールレベルのシングルトンとして
遅延ロード(初回生成リクエスト時にロードし、以後プロセス内に常駐)する。
"""
import os
import threading
import time

import torch
from accelerate import cpu_offload_with_hook, init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImageTransformer2DModel,
)
# 注意: 2509 以降の Edit モデル + Multiple-angles LoRA は Plus 条件付けで学習されている。
# QwenImageEditPipeline(旧形式)だとキャラクター同一性が崩れる(別人が生成される)。
# ComfyUI ワークフローの TextEncodeQwenImageEditPlus に対応するのはこちら。
from diffusers import QwenImageEditPlusPipeline
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from safetensors.torch import safe_open
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from prompts import NEGATIVE_PROMPT

# ローカル(ComfyUI)優先パス。存在しない場合は隣の HF リポジトリから自動ダウンロードする
# (_resolve_model_path 参照。ダウンロード先は通常の HF キャッシュ、~/.cache/huggingface)。
# ComfyUI のインストール先は環境変数 COMFYUI_DIR で上書き可能(デフォルトは ~/ComfyUI)。
# 特定ユーザー名を含む絶対パスをハードコードしないため、ホームディレクトリ相対で解決する。
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI"))
COMFYUI_MODELS_DIR = os.path.join(COMFYUI_DIR, "models")

TRANSFORMER_PATH = os.path.join(
    COMFYUI_MODELS_DIR, "diffusion_models", "qwen_image_edit_2511_bf16.safetensors"
)
TRANSFORMER_HF_REPO = "Comfy-Org/Qwen-Image-Edit_ComfyUI"
TRANSFORMER_HF_FILE = "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors"

LORA_LIGHTNING_PATH = os.path.join(
    COMFYUI_MODELS_DIR, "loras", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
)
LORA_LIGHTNING_HF_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LORA_LIGHTNING_HF_FILE = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

LORA_ANGLES_PATH = os.path.join(
    COMFYUI_MODELS_DIR, "loras", "Qwen-Edit-2509-Multiple-angles.safetensors"
)
LORA_ANGLES_HF_REPO = "Comfy-Org/Qwen-Image-Edit_ComfyUI"
LORA_ANGLES_HF_FILE = "split_files/loras/Qwen-Edit-2509-Multiple-angles.safetensors"

# フォールバック用(2511 transformer と Multiple-angles LoRA が非互換だった場合)
FALLBACK_TRANSFORMER_PATH = os.path.join(
    COMFYUI_MODELS_DIR, "diffusion_models", "qwen_image_edit_2509_fp8_e4m3fn.safetensors"
)
FALLBACK_TRANSFORMER_HF_REPO = "Comfy-Org/Qwen-Image-Edit_ComfyUI"
FALLBACK_TRANSFORMER_HF_FILE = "split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors"
FALLBACK_PREFIX = "model.diffusion_model."


def _resolve_model_path(local_path: str, repo_id: str, repo_filename: str) -> str:
    """ローカル(ComfyUI)にファイルがあればそれを使い、無ければ HF Hub から自動ダウンロードして
    そのキャッシュパスを返す(2 回目以降はキャッシュヒットで再ダウンロードしない)。
    """
    if os.path.exists(local_path):
        return local_path
    print(f"[pipeline] {local_path} が見つからないため HF Hub からダウンロードします: "
          f"{repo_id}/{repo_filename}")
    downloaded = hf_hub_download(repo_id=repo_id, filename=repo_filename)
    print(f"[pipeline] ダウンロード完了: {downloaded}")
    return downloaded


SHIFT = 3.0
NUM_INFERENCE_STEPS = 4
TRUE_CFG_SCALE = 1.0
TARGET_PIXELS = 1024 * 1024  # ComfyUI ImageScaleToTotalPixels(1.0 megapixel) 相当
VRAM_FREE_THRESHOLD_GB = 65.0
# text_encoder(bf16で~16GB)を常時GPU常駐にしたままだと、group offload(transformerのみ
# ブロックオフロード)でも実測で追加 ~22GB 必要になる(tools/test_group_offload.py で検証済み)。
# 24GB級カード(RTX 4090等)ではこれが厳しいため、この閾値未満では text_encoder も
# 使用直前/直後で GPU<->CPU を入れ替える(group_lowvram モード)。
VRAM_LOW_THRESHOLD_GB = 28.0
OFFLOAD_MODE_ENV = "CHARSHEET_OFFLOAD_MODE"
GROUP_OFFLOAD_BLOCKS_ENV = "CHARSHEET_GROUP_OFFLOAD_BLOCKS"
GROUP_OFFLOAD_USE_STREAM_ENV = "CHARSHEET_GROUP_OFFLOAD_USE_STREAM"
GROUP_OFFLOAD_NON_BLOCKING_ENV = "CHARSHEET_GROUP_OFFLOAD_NON_BLOCKING"
GROUP_OFFLOAD_RECORD_STREAM_ENV = "CHARSHEET_GROUP_OFFLOAD_RECORD_STREAM"
GROUP_OFFLOAD_LOW_CPU_MEM_ENV = "CHARSHEET_GROUP_OFFLOAD_LOW_CPU_MEM"
GROUP_OFFLOAD_DISK_PATH_ENV = "CHARSHEET_GROUP_OFFLOAD_DISK_PATH"

_pipe = None
_pipe_lock = threading.Lock()
_load_info = {
    "cpu_offload": False,
    "offload_mode": None,
    "fallback_transformer": False,
    "load_time_s": None,
    "group_offload": None,
}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _auto_offload_mode(free_gb: float) -> str:
    if free_gb >= VRAM_FREE_THRESHOLD_GB:
        return "none"
    if free_gb >= VRAM_LOW_THRESHOLD_GB:
        return "group"
    return "group_lowvram"


def _resolve_offload_mode(free_gb: float) -> str:
    """実行時の offload 戦略を決める。

    既定(auto)は空き VRAM に応じて3段階:
    - VRAM_FREE_THRESHOLD_GB 以上: none(全常駐、最速)
    - VRAM_LOW_THRESHOLD_GB 以上・VRAM_FREE_THRESHOLD_GB 未満: group
      (transformer のみブロック単位で GPU/CPU 入れ替え、text_encoder/vae は GPU 常駐)
    - VRAM_LOW_THRESHOLD_GB 未満: group_lowvram
      (group に加えて text_encoder も使用直前/直後で GPU<->CPU を入れ替える。
       24GB級カード(RTX 4090等)向け。text_encoder 常駐だけで ~16GB 使うため、
       group のままだと実測で追加 ~22GB 必要になり 24GB カードでは厳しい)

    32GB級カード(RTX 5090等)で model_cpu(丸ごとオフロード)を使うと
    transformer 本体(bf16で約40GB)だけで空きVRAMを超えてOOMするため、auto では
    model_cpu を選ばない(ダミーVRAM確保で30GB空きを再現した検証で group は正常
    動作・none と bit-for-bit 同一出力を確認済み。tools/test_group_offload.py 参照)。
    model_cpu は比較検証用に CHARSHEET_OFFLOAD_MODE=model_cpu で明示指定した場合のみ使う。
    """
    requested = os.environ.get(OFFLOAD_MODE_ENV, "auto").strip().lower().replace("-", "_")
    aliases = {
        "cpu": "model_cpu",
        "cpu_offload": "model_cpu",
        "model": "model_cpu",
        "model_cpu_offload": "model_cpu",
        "group_offload": "group",
        "group_low": "group_lowvram",
        "group_low_vram": "group_lowvram",
        "lowvram": "group_lowvram",
        "low_vram": "group_lowvram",
        "off": "none",
        "cuda": "none",
        "full_cuda": "none",
        "disable": "none",
        "disabled": "none",
    }
    requested = aliases.get(requested, requested)
    valid_modes = {"model_cpu", "group", "group_lowvram", "none"}
    if requested == "auto":
        return _auto_offload_mode(free_gb)
    if requested not in valid_modes:
        print(f"[pipeline] unknown {OFFLOAD_MODE_ENV}={requested!r}; falling back to auto")
        return _auto_offload_mode(free_gb)
    if requested in {"group", "group_lowvram", "none"} and not torch.cuda.is_available():
        print(f"[pipeline] CUDA unavailable; {requested} mode falls back to CPU execution")
        return "none"
    return requested


def _free_vram_gb() -> float:
    """現在の空き VRAM(GB)を取得。CUDA が使えない場合は 0 を返す。"""
    if not torch.cuda.is_available():
        return 0.0
    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    return free_bytes / (1024 ** 3)


def _load_safetensors_streaming(
    transformer,
    path: str,
    load_device: str | torch.device,
    strip_prefix: str = "",
    dtype: torch.dtype = torch.bfloat16,
):
    """safetensors を 1 tensor ずつ meta model へ流し込み、CPU peak を抑える。

    注意: init_empty_weights() で作った meta モデルのパラメータ dtype は既定で float32。
    set_module_tensor_to_device() は dtype= を渡さないと value を「meta 側の既存 dtype」
    (= float32)へ再キャストしてしまうため、bf16(や fp8)のつもりが実質 fp32 でロードされ、
    ホストRAM/VRAMを本来の2〜4倍消費するバグになる。必ず dtype= を明示すること。
    """
    expected_keys = set(transformer.state_dict().keys())
    loaded_keys = set()
    unexpected_keys = []
    load_device = str(load_device)

    with safe_open(path, framework="pt", device=load_device) as f:
        for raw_key in f.keys():
            if raw_key == "__index_timestep_zero__":
                continue
            key = raw_key[len(strip_prefix):] if strip_prefix and raw_key.startswith(strip_prefix) else raw_key
            if key not in expected_keys:
                unexpected_keys.append(key)
                continue
            tensor = f.get_tensor(raw_key)
            set_module_tensor_to_device(transformer, key, load_device, value=tensor, dtype=dtype, clear_cache=False)
            loaded_keys.add(key)
            del tensor

    missing_keys = sorted(expected_keys - loaded_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"transformer state_dict mismatch: missing={len(missing_keys)}, "
            f"unexpected={len(unexpected_keys)}"
        )


def _load_transformer(load_device: str | torch.device | None = None):
    """transformer をローカル ComfyUI ファイルから読み込む。
    まず 2511 bf16 を試し、LoRA 適用時に非互換エラーが出た場合は
    呼び出し側で 2509 fp8 にフォールバックする。
    """
    path = _resolve_model_path(TRANSFORMER_PATH, TRANSFORMER_HF_REPO, TRANSFORMER_HF_FILE)
    config = QwenImageTransformer2DModel.load_config("Qwen/Qwen-Image-Edit-2509", subfolder="transformer")
    with init_empty_weights():
        transformer = QwenImageTransformer2DModel.from_config(config)
    if load_device is None:
        load_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _load_safetensors_streaming(transformer, path, load_device)
    transformer.eval()
    return transformer


def _load_transformer_fallback(load_device: str | torch.device | None = None):
    """2509 fp8 transformer をフォールバックとして読み込む(prefix 除去 + bf16 キャスト)。"""
    path = _resolve_model_path(
        FALLBACK_TRANSFORMER_PATH, FALLBACK_TRANSFORMER_HF_REPO, FALLBACK_TRANSFORMER_HF_FILE
    )
    config = QwenImageTransformer2DModel.load_config("Qwen/Qwen-Image-Edit-2509", subfolder="transformer")
    with init_empty_weights():
        transformer = QwenImageTransformer2DModel.from_config(config)
    if load_device is None:
        load_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _load_safetensors_streaming(transformer, path, load_device, strip_prefix=FALLBACK_PREFIX)
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
    lightning_path = _resolve_model_path(
        LORA_LIGHTNING_PATH, LORA_LIGHTNING_HF_REPO, LORA_LIGHTNING_HF_FILE
    )
    angles_path = _resolve_model_path(LORA_ANGLES_PATH, LORA_ANGLES_HF_REPO, LORA_ANGLES_HF_FILE)
    pipe.load_lora_weights(lightning_path, adapter_name="lightning")
    pipe.load_lora_weights(angles_path, adapter_name="angles")
    pipe.set_adapters(["lightning", "angles"], adapter_weights=[1.0, 1.0])


def _apply_group_offload_to_transformer(pipe, offload_all_components: bool) -> dict:
    """transformer を block-level group offloading で GPU/CPU 入れ替える。

    transformer 本体(bf16で約40GB)だけが VRAM に対して大きすぎるため、これを
    ブロック単位(既定 num_blocks_per_group=1)+ CUDA stream プリフェッチで GPU/CPU
    入れ替える。vae は十分小さいので常時 GPU 常駐のままにする。
    offload_all_components=True の場合は text_encoder(~16GB)も丸ごとスワップし、
    24GB級カード向けにさらに VRAM を削る(group_lowvram モード)。
    """
    num_blocks = int(os.environ.get(GROUP_OFFLOAD_BLOCKS_ENV, "1"))
    use_stream = _env_bool(GROUP_OFFLOAD_USE_STREAM_ENV, True)
    non_blocking = _env_bool(GROUP_OFFLOAD_NON_BLOCKING_ENV, use_stream)
    record_stream = _env_bool(GROUP_OFFLOAD_RECORD_STREAM_ENV, False)
    low_cpu_mem_usage = _env_bool(GROUP_OFFLOAD_LOW_CPU_MEM_ENV, True)
    offload_to_disk_path = os.environ.get(GROUP_OFFLOAD_DISK_PATH_ENV)
    if offload_to_disk_path:
        os.makedirs(offload_to_disk_path, exist_ok=True)

    config = {
        "component": "transformer",
        "offload_type": "block_level",
        "num_blocks_per_group": num_blocks,
        "use_stream": use_stream,
        "non_blocking": non_blocking,
        "record_stream": record_stream,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "offload_to_disk_path": offload_to_disk_path,
    }
    print(f"[pipeline] enabling transformer group offload: {config}")
    pipe.transformer.enable_group_offload(
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=num_blocks,
        non_blocking=non_blocking,
        use_stream=use_stream,
        record_stream=record_stream,
        low_cpu_mem_usage=low_cpu_mem_usage,
        offload_to_disk_path=offload_to_disk_path,
    )

    # transformer は group hook が CUDA へ onload する。vae は毎回丸ごと入れ替える
    # ほど大きくないので常時 CUDA に置いたままにする。text_encoder は呼び出し側
    # (offload_all_components フラグ)次第で常駐 or 丸ごとスワップを切り替える。
    components = ("vae",) if offload_all_components else ("vae", "text_encoder")
    for component_name in components:
        component = getattr(pipe, component_name, None)
        if component is not None:
            component.to("cuda")

    if offload_all_components:
        config["text_encoder"] = _apply_text_encoder_swap_offload(pipe)
    return config


def _apply_text_encoder_swap_offload(pipe) -> dict:
    """text_encoder(bf16で~16GB)を使用直前にGPUへ、使用直後にCPUへ丸ごと入れ替える。

    text_encoder はブロック構造が vision塔+言語モデルの二重構成で transformer ほど
    単純ではないため、ブロック単位オフロードではなく accelerate.cpu_offload_with_hook
    (diffusers の enable_model_cpu_offload() が内部で使うのと同じ実績のある部品)で
    丸ごとスワップする。true_cfg_scale<=1 で CFG 無効のため、1回の生成で text_encoder
    の forward は1回しか呼ばれず、ブロック単位にする利点が薄いことも理由。
    24GB級カード(RTX 4090等)で text_encoder の常時GPU常駐(~16GB)が VRAM を圧迫する
    場合向け。
    """
    text_encoder = pipe.text_encoder
    _, hook = cpu_offload_with_hook(text_encoder, execution_device=torch.device("cuda"))

    def _offload_after_forward(module, args, output):
        hook.offload()
        return output

    text_encoder.register_forward_hook(_offload_after_forward)
    return {"component": "text_encoder", "offload_type": "whole_module_swap"}


def _configure_pipeline_runtime(pipe, offload_mode: str) -> dict | None:
    """offload 戦略を pipeline に適用する。"""
    if offload_mode == "model_cpu":
        pipe.enable_model_cpu_offload()
        return None
    if offload_mode == "group":
        return _apply_group_offload_to_transformer(pipe, offload_all_components=False)
    if offload_mode == "group_lowvram":
        return _apply_group_offload_to_transformer(pipe, offload_all_components=True)
    if torch.cuda.is_available():
        pipe.to("cuda")
    return None


def _load_pipeline_locked():
    """pipe をロードする(呼び出し側で _pipe_lock を保持していること)。"""
    global _pipe
    t0 = time.time()

    free_gb = _free_vram_gb()
    offload_mode = _resolve_offload_mode(free_gb)
    use_cpu_offload = offload_mode == "model_cpu"
    # "none"(VRAM に全常駐)以外は raw state_dict を先に CPU へロードする。
    # cuda:0 に直接ロードすると、enable_model_cpu_offload の hook が効く前に
    # transformer 本体(bf16 換算で ~40GB)だけで空きVRAMを使い切って OOM することがある
    # (32GB級カードで実際に発生することを確認済み)。
    transformer_load_device = "cpu" if offload_mode != "none" else None
    print(f"[pipeline] free VRAM: {free_gb:.1f} GB -> offload_mode={offload_mode}")

    fallback_used = False
    group_offload_config = None
    try:
        transformer = _load_transformer(load_device=transformer_load_device)
        pipe = _build_pipeline(transformer)
        if offload_mode in {"group", "group_lowvram"}:
            _apply_loras(pipe)
            group_offload_config = _configure_pipeline_runtime(pipe, offload_mode)
        else:
            _configure_pipeline_runtime(pipe, offload_mode)
            _apply_loras(pipe)
    except Exception as exc:  # noqa: BLE001 - 2511 + angles LoRA 非互換時のフォールバック
        print(f"[pipeline] primary transformer/LoRA failed ({exc!r}); falling back to 2509 fp8")
        fallback_used = True
        transformer = _load_transformer_fallback(load_device=transformer_load_device)
        pipe = _build_pipeline(transformer)
        if offload_mode in {"group", "group_lowvram"}:
            _apply_loras(pipe)
            group_offload_config = _configure_pipeline_runtime(pipe, offload_mode)
        else:
            _configure_pipeline_runtime(pipe, offload_mode)
            _apply_loras(pipe)

    pipe.scheduler.config["shift"] = SHIFT

    _load_info["cpu_offload"] = use_cpu_offload
    _load_info["offload_mode"] = offload_mode
    _load_info["fallback_transformer"] = fallback_used
    _load_info["load_time_s"] = time.time() - t0
    _load_info["group_offload"] = group_offload_config
    print(
        f"[pipeline] loaded in {_load_info['load_time_s']:.1f}s "
        f"(fallback={fallback_used}, offload_mode={offload_mode})"
    )

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
