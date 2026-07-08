# -*- coding: utf-8 -*-
"""
キャラクターシート合成(4列×2行グリッド)・ZIP 作成。
仕様書 §7 準拠。
"""
import datetime
import io
import os
import zipfile

from PIL import Image, ImageDraw, ImageFont

from prompts import SHEET_LAYOUT, VIEW_BY_KEY

CELL_W = 400
CELL_H = 500
IMAGE_AREA_H = 440
MARGIN = 20
TITLE_H = 90
LABEL_FONT_SIZE = 18
TITLE_FONT_SIZE = 34
BG_COLOR = (255, 255, 255)
CELL_BORDER_COLOR = (220, 220, 220)
TEXT_COLOR = (30, 30, 30)
SUBTEXT_COLOR = (100, 100, 100)

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansMonoCJK-Regular.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _fit_image(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    """アスペクト比を維持して box にフィット(縮小のみ)させる。"""
    w, h = img.size
    scale = min(box_w / w, box_h / h, 1.0)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def build_character_sheet(image_paths: dict, title: str = "Character Sheet") -> Image.Image:
    """
    image_paths: {key: path_to_png} の dict (8方向すべて揃っている想定)
    並び順・ラベルは prompts.SHEET_LAYOUT / VIEW_BY_KEY に準拠。
    """
    cols = len(SHEET_LAYOUT[0])
    rows = len(SHEET_LAYOUT)

    sheet_w = MARGIN * (cols + 1) + CELL_W * cols
    sheet_h = TITLE_H + MARGIN * (rows + 1) + CELL_H * rows

    sheet = Image.new("RGB", (sheet_w, sheet_h), BG_COLOR)
    draw = ImageDraw.Draw(sheet)

    title_font = _load_font(TITLE_FONT_SIZE)
    label_font = _load_font(LABEL_FONT_SIZE)
    sub_font = _load_font(14)

    # タイトル
    draw.text((MARGIN, 20), title, fill=TEXT_COLOR, font=title_font)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    draw.text((MARGIN, 60), timestamp, fill=SUBTEXT_COLOR, font=sub_font)

    for row_idx, row_keys in enumerate(SHEET_LAYOUT):
        for col_idx, key in enumerate(row_keys):
            cell_x = MARGIN + col_idx * (CELL_W + MARGIN)
            cell_y = TITLE_H + MARGIN + row_idx * (CELL_H + MARGIN)

            # セル枠
            draw.rectangle(
                [cell_x, cell_y, cell_x + CELL_W, cell_y + CELL_H],
                outline=CELL_BORDER_COLOR,
                width=1,
            )

            path = image_paths.get(key)
            if path and os.path.exists(path):
                img = Image.open(path)
                # 透過画像(背景削除済み)は alpha を mask にして白背景へ合成
                if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                    img = img.convert("RGBA")
                else:
                    img = img.convert("RGB")
                fitted = _fit_image(img, CELL_W - 20, IMAGE_AREA_H - 10)
                fw, fh = fitted.size
                paste_x = cell_x + (CELL_W - fw) // 2
                paste_y = cell_y + 10 + (IMAGE_AREA_H - 10 - fh) // 2
                if fitted.mode == "RGBA":
                    sheet.paste(fitted, (paste_x, paste_y), fitted)
                else:
                    sheet.paste(fitted, (paste_x, paste_y))

            view = VIEW_BY_KEY.get(key, {})
            label_ja = view.get("label_ja", key)
            label_en = view.get("label_en", key)
            label_text = f"{label_ja} / {label_en}"

            bbox = draw.textbbox((0, 0), label_text, font=label_font)
            text_w = bbox[2] - bbox[0]
            label_x = cell_x + (CELL_W - text_w) // 2
            label_y = cell_y + IMAGE_AREA_H + 12
            draw.text((label_x, label_y), label_text, fill=TEXT_COLOR, font=label_font)

    return sheet


def build_zip(job_dir: str, image_paths: dict, sheet_path: str) -> bytes:
    """8方向画像 + sheet.png を ZIP にまとめてバイト列で返す。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for key, path in image_paths.items():
            if path and os.path.exists(path):
                zf.write(path, arcname=f"{key}.png")
        if sheet_path and os.path.exists(sheet_path):
            zf.write(sheet_path, arcname="sheet.png")
    buf.seek(0)
    return buf.read()
