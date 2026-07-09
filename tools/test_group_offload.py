# -*- coding: utf-8 -*-
"""Diffusers group offloading の単発比較テスト。

アプリ本体の通常動作は変えず、サブプロセスごとに offload mode を切り替えて
同じ入力画像・同じプロンプトで 1 方向だけ生成時間を測る。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "_group_offload_test"


def _reserve_cuda_memory(target_free_gb: float | None, chunk_mb: int = 512) -> list[torch.Tensor]:
    """CUDA の空きメモリが target_free_gb 程度になるまでダミー確保する。

    実GPUの容量を厳密に変えるものではないが、48GB/80GB等のGPU上で
    「24GB級GPUで残りメモリが少ない」状況のスモークテストに使える。
    """
    if target_free_gb is None or not torch.cuda.is_available():
        return []

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    target_free_bytes = int(target_free_gb * (1024**3))
    reserve_bytes = max(0, free_bytes - target_free_bytes)
    if reserve_bytes == 0:
        print(
            f"[vram-limit] free={free_bytes / (1024**3):.2f}GB <= "
            f"target={target_free_gb:.2f}GB; no reservation",
            flush=True,
        )
        return []

    tensors = []
    chunk_bytes = max(1, chunk_mb) * 1024 * 1024
    remaining = reserve_bytes
    print(
        f"[vram-limit] total={total_bytes / (1024**3):.2f}GB, "
        f"free={free_bytes / (1024**3):.2f}GB -> reserving "
        f"{reserve_bytes / (1024**3):.2f}GB to leave ~{target_free_gb:.2f}GB",
        flush=True,
    )
    while remaining > 0:
        alloc_bytes = min(chunk_bytes, remaining)
        tensors.append(torch.empty((alloc_bytes,), dtype=torch.uint8, device="cuda"))
        remaining -= alloc_bytes
    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info()
    print(f"[vram-limit] free after reservation: {free_after / (1024**3):.2f}GB", flush=True)
    return tensors


def _cuda_snapshot() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False}
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "cuda": True,
        "free_gb": free_bytes / (1024**3),
        "total_gb": total_bytes / (1024**3),
        "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "max_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
    }


def _run_worker(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(ROOT))
    os.environ["CHARSHEET_OFFLOAD_MODE"] = args.mode

    import pipeline  # noqa: PLC0415
    from prompts import NEGATIVE_PROMPT, VIEW_BY_KEY  # noqa: PLC0415

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = Path(args.result_json)
    view = VIEW_BY_KEY[args.view]

    vram_reservation = _reserve_cuda_memory(args.limit_free_vram_gb, args.vram_chunk_mb)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    before = _cuda_snapshot()

    image = Image.open(args.image).convert("RGB")
    processed = pipeline.preprocess_image(image, target_pixels=args.target_pixels)

    t0 = time.perf_counter()
    pipe = pipeline.get_pipeline()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_elapsed_s = time.perf_counter() - t0

    generation_times = []
    output_paths = []
    for index in range(args.repeats):
        t1 = time.perf_counter()
        out_image = pipeline.generate_view(
            processed,
            prompt=view["prompt"],
            seed=args.seed + index,
            negative_prompt=NEGATIVE_PROMPT,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t1
        generation_times.append(elapsed)

        output_path = output_dir / f"{args.mode}_{args.view}_{index + 1}.png"
        out_image.save(output_path)
        output_paths.append(str(output_path))

    after = _cuda_snapshot()
    result = {
        "mode": args.mode,
        "view": args.view,
        "seed": args.seed,
        "target_pixels": args.target_pixels,
        "repeats": args.repeats,
        "load_elapsed_s": load_elapsed_s,
        "pipeline_load_info": pipeline.get_load_info(),
        "limit_free_vram_gb": args.limit_free_vram_gb,
        "vram_reserved_gb": sum(t.numel() for t in vram_reservation) / (1024**3),
        "generation_times_s": generation_times,
        "generation_mean_s": mean(generation_times),
        "cuda_before": before,
        "cuda_after": after,
        "output_paths": output_paths,
        "execution_device": str(pipe._execution_device),
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_parent(args: argparse.Namespace) -> int:
    image_path = Path(args.image).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for mode in args.modes:
        result_json = output_dir / f"result_{mode}.json"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--image",
            str(image_path),
            "--mode",
            mode,
            "--view",
            args.view,
            "--seed",
            str(args.seed),
            "--repeats",
            str(args.repeats),
            "--target-pixels",
            str(args.target_pixels),
            "--output-dir",
            str(output_dir),
            *(
                ["--limit-free-vram-gb", str(args.limit_free_vram_gb)]
                if args.limit_free_vram_gb is not None
                else []
            ),
            "--vram-chunk-mb",
            str(args.vram_chunk_mb),
            "--result-json",
            str(result_json),
        ]
        env = os.environ.copy()
        env["CHARSHEET_OFFLOAD_MODE"] = mode
        print(f"\n=== {mode} をテスト中 ===", flush=True)
        completed = subprocess.run(cmd, cwd=ROOT, env=env, check=False)  # noqa: S603
        if completed.returncode != 0:
            print(f"{mode}: failed with exit code {completed.returncode}", file=sys.stderr)
            return completed.returncode
        summary.append(json.loads(result_json.read_text(encoding="utf-8")))

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    for result in summary:
        cuda = result["cuda_after"]
        max_reserved = cuda.get("max_reserved_gb") if cuda.get("cuda") else None
        reserved_text = f", max_reserved={max_reserved:.2f}GB" if max_reserved is not None else ""
        print(
            f"{result['mode']}: load={result['load_elapsed_s']:.2f}s, "
            f"gen_mean={result['generation_mean_s']:.2f}s{reserved_text}, "
            f"device={result['execution_device']}"
        )
    print(f"詳細JSON: {summary_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen pipeline group offload smoke/benchmark test")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--image", required=True, help="入力キャラクター画像")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["model_cpu", "group"],
        choices=["auto", "model_cpu", "group", "group_lowvram", "none"],
        help="親プロセスで順に比較する offload mode",
    )
    parser.add_argument(
        "--mode",
        default="group",
        choices=["auto", "model_cpu", "group", "group_lowvram", "none"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--view", default="front", help="prompts.py の view key")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1, help="各 mode の生成回数")
    parser.add_argument("--target-pixels", type=int, default=1024 * 1024)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--limit-free-vram-gb",
        type=float,
        default=None,
        help="各テストプロセス内で空きVRAMが指定GB程度になるようダミー確保する。例: 24",
    )
    parser.add_argument("--vram-chunk-mb", type=int, default=512, help="VRAMダミー確保のチャンクサイズMB")
    parser.add_argument("--result-json", default=str(DEFAULT_OUTPUT_DIR / "result.json"), help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.view not in {
        "front",
        "back",
        "left",
        "right",
        "front_left_45",
        "front_right_45",
        "back_left_45",
        "back_right_45",
    }:
        raise SystemExit(f"unknown view: {args.view}")
    if args.worker:
        _run_worker(args)
        return 0
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
