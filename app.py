# -*- coding: utf-8 -*-
"""
キャラクターシート作成 Web アプリ本体(FastAPI)。
仕様書 docs/character_sheet_spec.md §4, §5 準拠。
"""
import os
import shutil
import threading
import traceback
import uuid
from datetime import datetime

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

import bg
import pipeline
import sheet
import split as splitmod
from prompts import NEGATIVE_PROMPT, VIEW_BY_KEY, VIEWS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SPLITS_DIR = os.path.join(OUTPUTS_DIR, "_splits")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)

app = FastAPI(title="Character Sheet Generator")

# --- ジョブ管理(メモリ内) ---
jobs = {}  # job_id -> dict
jobs_lock = threading.Lock()
current_job_id = None  # 実行中ジョブ(同時1件)
current_job_lock = threading.Lock()


def _job_dir(job_id: str) -> str:
    d = os.path.join(OUTPUTS_DIR, job_id)
    os.makedirs(d, exist_ok=True)
    return d


def _init_job(job_id: str, seed: int):
    views = [
        {
            "key": v["key"],
            "label_ja": v["label_ja"],
            "label_en": v["label_en"],
            "status": "queued",
            "url": None,
            "has_prev": False,
        }
        for v in VIEWS
    ]
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "total": len(VIEWS),
            "seed": seed,
            "views": views,
            "sheet_url": None,
            "zip_url": None,
            "error": None,
            "refine_error": None,
            "created_at": datetime.now().isoformat(),
            "load_info": None,
        }


def _has_prev(job_id: str, key: str) -> bool:
    return os.path.exists(os.path.join(OUTPUTS_DIR, job_id, f"{key}_prev.png"))


def _refresh_has_prev(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job:
            for v in job["views"]:
                v["has_prev"] = _has_prev(job_id, v["key"])


def _restore_job_from_disk(job_id: str):
    """メモリに無いジョブを outputs/{job_id}/ から再構築する。
    8方向すべての画像が揃っている場合のみ復元(それ以外は None)。
    サーバー再起動後でも既存ジョブに対して refine / undo / 参照 ができるようにする。
    """
    # パストラバーサル防止
    if not job_id or "/" in job_id or "\\" in job_id or ".." in job_id:
        return None
    job_dir = os.path.join(OUTPUTS_DIR, job_id)
    if not os.path.isdir(job_dir):
        return None
    for v in VIEWS:
        if not os.path.exists(os.path.join(job_dir, f"{v['key']}.png")):
            return None

    views = [
        {
            "key": v["key"],
            "label_ja": v["label_ja"],
            "label_en": v["label_en"],
            "status": "done",
            "url": f"/api/jobs/{job_id}/images/{v['key']}.png",
            "has_prev": _has_prev(job_id, v["key"]),
        }
        for v in VIEWS
    ]
    sheet_exists = os.path.exists(os.path.join(job_dir, "sheet.png"))
    zip_exists = os.path.exists(os.path.join(job_dir, "download.zip"))
    job = {
        "job_id": job_id,
        "status": "done",
        "progress": len(VIEWS),
        "total": len(VIEWS),
        "seed": 0,
        "views": views,
        "sheet_url": f"/api/jobs/{job_id}/sheet.png" if sheet_exists else None,
        "zip_url": f"/api/jobs/{job_id}/download.zip" if zip_exists else None,
        "error": None,
        "refine_error": None,
        "created_at": datetime.fromtimestamp(os.path.getmtime(job_dir)).isoformat(),
        "load_info": None,
        "restored_from_disk": True,
    }
    with jobs_lock:
        # 競合時は既存を優先
        if job_id not in jobs:
            jobs[job_id] = job
        return jobs[job_id]


def _get_or_restore_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is not None:
        return job
    return _restore_job_from_disk(job_id)


def _update_job(job_id: str, **kwargs):
    with jobs_lock:
        jobs[job_id].update(kwargs)


def _update_view(job_id: str, key: str, **kwargs):
    with jobs_lock:
        for v in jobs[job_id]["views"]:
            if v["key"] == key:
                v.update(kwargs)
                break


def _rebuild_sheet_and_zip(job_id: str):
    """8方向画像から sheet.png と download.zip を再生成する。"""
    job_dir = os.path.join(OUTPUTS_DIR, job_id)
    image_paths = {v["key"]: os.path.join(job_dir, f"{v['key']}.png") for v in VIEWS}
    sheet_img = sheet.build_character_sheet(image_paths, title="Character Sheet")
    sheet_path = os.path.join(job_dir, "sheet.png")
    sheet_img.save(sheet_path)

    zip_bytes = sheet.build_zip(job_dir, image_paths, sheet_path)
    zip_path = os.path.join(job_dir, "download.zip")
    with open(zip_path, "wb") as f:
        f.write(zip_bytes)

    _update_job(
        job_id,
        sheet_url=f"/api/jobs/{job_id}/sheet.png",
        zip_url=f"/api/jobs/{job_id}/download.zip",
    )


def _run_job(job_id: str, input_path: str, seed: int):
    global current_job_id
    job_dir = _job_dir(job_id)
    try:
        _update_job(job_id, status="running")

        input_image = Image.open(input_path).convert("RGB")
        processed = pipeline.preprocess_image(input_image)
        processed.save(os.path.join(job_dir, "input.png"))

        image_paths = {}
        for view in VIEWS:
            key = view["key"]
            _update_view(job_id, key, status="running")
            try:
                out_image = pipeline.generate_view(
                    processed,
                    prompt=view["prompt"],
                    seed=seed,
                    negative_prompt=NEGATIVE_PROMPT,
                )
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                _update_view(job_id, key, status="error")
                _update_job(job_id, status="error", error=f"{key}: {exc}")
                return

            out_path = os.path.join(job_dir, f"{key}.png")
            out_image.save(out_path)
            image_paths[key] = out_path

            with jobs_lock:
                jobs[job_id]["progress"] += 1
                progress = jobs[job_id]["progress"]
            _update_view(
                job_id, key, status="done", url=f"/api/jobs/{job_id}/images/{key}.png"
            )
            _update_job(job_id, progress=progress)

        # 全方向完了 -> シート合成 + zip
        sheet_img = sheet.build_character_sheet(image_paths, title="Character Sheet")
        sheet_path = os.path.join(job_dir, "sheet.png")
        sheet_img.save(sheet_path)

        zip_bytes = sheet.build_zip(job_dir, image_paths, sheet_path)
        zip_path = os.path.join(job_dir, "download.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)

        _update_job(
            job_id,
            status="done",
            sheet_url=f"/api/jobs/{job_id}/sheet.png",
            zip_url=f"/api/jobs/{job_id}/download.zip",
            load_info=pipeline.get_load_info(),
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        _update_job(job_id, status="error", error=str(exc))
    finally:
        with current_job_lock:
            current_job_id = None


def _safe_split_id(split_id: str) -> bool:
    """split_id の妥当性検証(パストラバーサル対策)。"""
    return bool(split_id) and split_id.isalnum() and len(split_id) <= 32


@app.post("/api/split")
async def split_image(image: UploadFile = File(...)):
    """アップロード画像から複数キャラクターを検出・分離する(GPU 不使用、排他不要)。"""
    split_id = uuid.uuid4().hex[:12]
    split_dir = os.path.join(SPLITS_DIR, split_id)
    os.makedirs(split_dir, exist_ok=True)

    source_path = os.path.join(split_dir, "source.png")
    try:
        contents = await image.read()
        with open(source_path, "wb") as f:
            f.write(contents)
        src = Image.open(source_path).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")

    try:
        crops = splitmod.detect_figures(src)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"キャラクター検出に失敗しました: {exc}")

    figures = []
    for i, crop in enumerate(crops):
        crop.save(os.path.join(split_dir, f"figure_{i}.png"))
        figures.append(
            {
                "index": i,
                "url": f"/api/splits/{split_id}/figure_{i}.png",
                "width": crop.width,
                "height": crop.height,
            }
        )

    return {"split_id": split_id, "count": len(figures), "figures": figures}


@app.get("/api/splits/{split_id}/{filename}")
async def get_split_file(split_id: str, filename: str):
    if not _safe_split_id(split_id):
        raise HTTPException(status_code=404, detail="不正な split_id です")
    # source.png または figure_{i}.png のみ許可(パストラバーサル対策)
    ok = filename == "source.png"
    if not ok and filename.startswith("figure_") and filename.endswith(".png"):
        idx = filename[len("figure_"):-len(".png")]
        ok = idx.isdigit()
    if not ok:
        raise HTTPException(status_code=404, detail="不正なファイル名です")
    path = os.path.join(SPLITS_DIR, split_id, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="ファイルが見つかりません")
    return FileResponse(path, media_type="image/png")


@app.post("/api/generate")
async def generate(
    image: UploadFile = File(None),
    seed: int = Form(0),
    split_id: str = Form(None),
    figure_index: int = Form(None),
):
    global current_job_id

    # 入力の決定: multipart image か、split の保存済み crop のどちらか
    crop_path = None
    if image is None or not (image.filename or "").strip():
        if split_id is None or figure_index is None:
            raise HTTPException(
                status_code=400,
                detail="image ファイル、または split_id + figure_index のどちらかを指定してください",
            )
        if not _safe_split_id(split_id):
            raise HTTPException(status_code=400, detail="不正な split_id です")
        if figure_index < 0:
            raise HTTPException(status_code=400, detail="figure_index が不正です")
        crop_path = os.path.join(SPLITS_DIR, split_id, f"figure_{figure_index}.png")
        if not os.path.exists(crop_path):
            raise HTTPException(status_code=400, detail="指定された分割画像が見つかりません")

    with current_job_lock:
        if current_job_id is not None:
            raise HTTPException(status_code=409, detail="別のジョブが実行中です。しばらく待ってから再試行してください。")
        job_id = uuid.uuid4().hex[:12]
        current_job_id = job_id

    job_dir = _job_dir(job_id)
    input_path = os.path.join(job_dir, "upload_raw.png")

    try:
        if crop_path is not None:
            shutil.copy2(crop_path, input_path)
        else:
            contents = await image.read()
            with open(input_path, "wb") as f:
                f.write(contents)
        # 検証: 開けるか確認
        with Image.open(input_path) as im:
            im.verify()
    except HTTPException:
        with current_job_lock:
            current_job_id = None
        raise
    except Exception as exc:
        with current_job_lock:
            current_job_id = None
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")

    _init_job(job_id, seed)

    thread = threading.Thread(target=_run_job, args=(job_id, input_path, seed), daemon=True)
    thread.start()

    return {"job_id": job_id}


def _run_refine(job_id: str, key: str, instruction: str, seed: int):
    global current_job_id
    job_dir = _job_dir(job_id)
    img_path = os.path.join(job_dir, f"{key}.png")
    try:
        _update_job(job_id, status="refining", refine_error=None)
        _update_view(job_id, key, status="refining")

        current_image = Image.open(img_path)
        if current_image.mode in ("RGBA", "LA"):
            # 背景削除済み画像は白背景に合成してから編集(透過→黒化を防ぐ)
            base = Image.new("RGBA", current_image.size, (255, 255, 255, 255))
            base.paste(current_image, (0, 0), current_image.convert("RGBA"))
            current_image = base.convert("RGB")
        else:
            current_image = current_image.convert("RGB")
        prompt = instruction.strip() + " Keep everything else exactly the same."

        out_image = pipeline.generate_view(
            current_image,
            prompt=prompt,
            seed=seed,
            negative_prompt=NEGATIVE_PROMPT,
        )
        out_image.save(img_path)

        _update_view(job_id, key, status="done")
        _rebuild_sheet_and_zip(job_id)
        _refresh_has_prev(job_id)
        _update_job(job_id, status="done")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        # 失敗時: 画像は上書きされていないか、バックアップ(_prev)が残っている
        _update_view(job_id, key, status="done")
        _refresh_has_prev(job_id)
        _update_job(job_id, status="done", refine_error=f"{key}: {exc}")
    finally:
        with current_job_lock:
            current_job_id = None


def _run_remove_bg(job_id: str, keys: list):
    """背景削除をバックグラウンドで実行する(keys は方向キーのリスト)。"""
    global current_job_id
    job_dir = _job_dir(job_id)
    try:
        _update_job(job_id, status="removing_bg", refine_error=None)
        errors = []
        for key in keys:
            img_path = os.path.join(job_dir, f"{key}.png")
            if not os.path.exists(img_path):
                continue
            _update_view(job_id, key, status="removing_bg")
            try:
                # 実行前にバックアップ(refine と同じ1世代仕組み)
                shutil.copy2(img_path, os.path.join(job_dir, f"{key}_prev.png"))
                out = bg.remove_background(Image.open(img_path))
                out.save(img_path)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                errors.append(f"{key}: {exc}")
            _update_view(job_id, key, status="done")

        _rebuild_sheet_and_zip(job_id)
        _refresh_has_prev(job_id)
        _update_job(
            job_id,
            status="done",
            refine_error=("背景削除: " + "; ".join(errors)) if errors else None,
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        for key in keys:
            _update_view(job_id, key, status="done")
        _refresh_has_prev(job_id)
        _update_job(job_id, status="done", refine_error=f"背景削除: {exc}")
    finally:
        with current_job_lock:
            current_job_id = None


class RefineRequest(BaseModel):
    key: str
    instruction: str
    seed: int = 0


class UndoRequest(BaseModel):
    key: str


class RemoveBgRequest(BaseModel):
    key: str  # 方向キー または "all"


@app.post("/api/jobs/{job_id}/refine")
async def refine(job_id: str, req: RefineRequest):
    global current_job_id

    job = _get_or_restore_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")
    if req.key not in VIEW_BY_KEY:
        raise HTTPException(status_code=400, detail=f"不正な方向キーです: {req.key}")
    if not req.instruction or not req.instruction.strip():
        raise HTTPException(status_code=400, detail="修正指示を入力してください")

    img_path = os.path.join(OUTPUTS_DIR, job_id, f"{req.key}.png")
    if not os.path.exists(img_path):
        raise HTTPException(status_code=409, detail="この方向の画像はまだ生成されていません")

    with current_job_lock:
        if current_job_id is not None:
            raise HTTPException(status_code=409, detail="別の生成/修正が実行中です。しばらく待ってから再試行してください。")
        current_job_id = job_id

    try:
        # 現画像をバックアップ(1世代のみ、上書き)
        prev_path = os.path.join(OUTPUTS_DIR, job_id, f"{req.key}_prev.png")
        shutil.copy2(img_path, prev_path)
        _refresh_has_prev(job_id)
    except Exception as exc:
        with current_job_lock:
            current_job_id = None
        raise HTTPException(status_code=500, detail=f"バックアップに失敗しました: {exc}")

    thread = threading.Thread(
        target=_run_refine, args=(job_id, req.key, req.instruction, req.seed), daemon=True
    )
    thread.start()

    return {"job_id": job_id, "key": req.key, "status": "refining"}


@app.post("/api/jobs/{job_id}/remove_bg")
async def remove_bg(job_id: str, req: RemoveBgRequest):
    global current_job_id

    job = _get_or_restore_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")
    if req.key != "all" and req.key not in VIEW_BY_KEY:
        raise HTTPException(status_code=400, detail=f"不正な方向キーです: {req.key}")

    keys = [v["key"] for v in VIEWS] if req.key == "all" else [req.key]
    job_dir = os.path.join(OUTPUTS_DIR, job_id)
    existing = [k for k in keys if os.path.exists(os.path.join(job_dir, f"{k}.png"))]
    if not existing:
        raise HTTPException(status_code=409, detail="対象の画像がまだ生成されていません")

    with current_job_lock:
        if current_job_id is not None:
            raise HTTPException(status_code=409, detail="別の処理が実行中です。しばらく待ってから再試行してください。")
        current_job_id = job_id

    thread = threading.Thread(target=_run_remove_bg, args=(job_id, existing), daemon=True)
    thread.start()

    return {"job_id": job_id, "keys": existing, "status": "removing_bg"}


@app.post("/api/jobs/{job_id}/undo")
def undo(job_id: str, req: UndoRequest):
    global current_job_id

    job = _get_or_restore_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")
    if req.key not in VIEW_BY_KEY:
        raise HTTPException(status_code=400, detail=f"不正な方向キーです: {req.key}")

    job_dir = os.path.join(OUTPUTS_DIR, job_id)
    img_path = os.path.join(job_dir, f"{req.key}.png")
    prev_path = os.path.join(job_dir, f"{req.key}_prev.png")
    if not os.path.exists(prev_path):
        raise HTTPException(status_code=404, detail="復元できるバックアップがありません")
    if not os.path.exists(img_path):
        raise HTTPException(status_code=409, detail="現在の画像が存在しません")

    with current_job_lock:
        if current_job_id is not None:
            raise HTTPException(status_code=409, detail="別の生成/修正が実行中です。しばらく待ってから再試行してください。")
        current_job_id = job_id

    try:
        # 入れ替え(トグル動作: 復元後は今の画像が _prev になる)
        tmp_path = os.path.join(job_dir, f"{req.key}_swap_tmp.png")
        os.replace(img_path, tmp_path)
        os.replace(prev_path, img_path)
        os.replace(tmp_path, prev_path)

        _rebuild_sheet_and_zip(job_id)
        _refresh_has_prev(job_id)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"復元に失敗しました: {exc}")
    finally:
        with current_job_lock:
            current_job_id = None

    with jobs_lock:
        return JSONResponse(dict(jobs[job_id]))


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = _get_or_restore_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")
    _refresh_has_prev(job_id)
    with jobs_lock:
        return JSONResponse(dict(jobs[job_id]))


@app.get("/api/jobs/{job_id}/images/{key}.png")
async def get_view_image(job_id: str, key: str):
    path = os.path.join(OUTPUTS_DIR, job_id, f"{key}.png")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="画像が見つかりません")
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/input.png")
async def get_input_image(job_id: str):
    path = os.path.join(OUTPUTS_DIR, job_id, "input.png")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="画像が見つかりません")
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/sheet.png")
async def get_sheet_image(job_id: str):
    path = os.path.join(OUTPUTS_DIR, job_id, "sheet.png")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="シートが見つかりません")
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/download.zip")
async def get_download_zip(job_id: str):
    path = os.path.join(OUTPUTS_DIR, job_id, "download.zip")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="ZIP が見つかりません")
    return FileResponse(path, media_type="application/zip", filename=f"character_sheet_{job_id}.zip")


# --- 静的ファイル配信 ---
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
