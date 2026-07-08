# -*- coding: utf-8 -*-
"""
シート分解取り込み(split)。

複数体のキャラクターが1枚に描かれたキャラクターシート画像から、
各キャラクターを背景色ベースの前景抽出 + 連結成分解析で検出・分離する。
GPU は使用しない(OpenCV のみ)。
"""
import cv2
import numpy as np
from PIL import Image

# 調整可能パラメータ
WORK_LONG_SIDE = 1600        # 作業用縮小の長辺
BORDER_FRAC = 0.015          # 背景色推定に使う外周の割合(1〜2%)
BG_DIST_THRESHOLD = 30.0     # 背景色とのユークリッド距離の閾値(/255)
MORPH_KERNEL = 7             # モルフォロジーのカーネルサイズ(5〜9px)
MIN_AREA_FRAC = 0.005        # 面積が全体の 0.5% 未満の成分は除外
MERGE_DIST_FRAC = 0.02       # ボックス統合の距離閾値(画像幅の 2%)
PAD_FRAC = 0.02              # 切り出し時のパディング(ボックス長辺の 2%)
ROW_CLUSTER_FRAC = 0.5       # 行クラスタリング: y中心差 < 平均ボックス高 × この係数で同一行


def _estimate_background_color(work: np.ndarray) -> np.ndarray:
    """画像の外周 1〜2% の画素から背景色を推定(中央値)。"""
    h, w = work.shape[:2]
    b = max(2, int(round(min(w, h) * BORDER_FRAC)))
    border = np.concatenate([
        work[:b].reshape(-1, 3),
        work[-b:].reshape(-1, 3),
        work[:, :b].reshape(-1, 3),
        work[:, -b:].reshape(-1, 3),
    ])
    return np.median(border, axis=0)


def _merge_close_boxes(boxes: list, merge_dist: float) -> list:
    """近接/重複するバウンディングボックスを統合する。"""

    def close(a, b):
        gap_x = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
        gap_y = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
        return gap_x <= merge_dist and gap_y <= merge_dist

    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        while boxes:
            cur = boxes.pop(0)
            i = 0
            while i < len(boxes):
                if close(cur, boxes[i]):
                    o = boxes.pop(i)
                    cur = [
                        min(cur[0], o[0]), min(cur[1], o[1]),
                        max(cur[2], o[2]), max(cur[3], o[3]),
                    ]
                    changed = True
                else:
                    i += 1
            out.append(cur)
        boxes = out
    return boxes


def _sort_reading_order(boxes: list) -> list:
    """行クラスタリング(y中心)→ 上の行から、行内は左→右の順にソート。"""
    if not boxes:
        return []
    mean_h = float(np.mean([b[3] - b[1] for b in boxes]))
    items = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2.0)

    rows = []  # [{"cy": float, "boxes": [...]}]
    for box in items:
        cy = (box[1] + box[3]) / 2.0
        placed = False
        for row in rows:
            if abs(cy - row["cy"]) < mean_h * ROW_CLUSTER_FRAC:
                row["boxes"].append(box)
                row["cy"] = float(np.mean([(b[1] + b[3]) / 2.0 for b in row["boxes"]]))
                placed = True
                break
        if not placed:
            rows.append({"cy": cy, "boxes": [box]})

    rows.sort(key=lambda r: r["cy"])
    ordered = []
    for row in rows:
        row["boxes"].sort(key=lambda b: b[0])
        ordered.extend(row["boxes"])
    return ordered


def detect_figures(
    image: Image.Image,
    bg_threshold: float = BG_DIST_THRESHOLD,
) -> list:
    """キャラクターシート画像から各キャラクターの切り出し画像リストを返す。

    検出数が 0 の場合は空リスト(呼び出し側で「1体扱い」にフォールバック)。
    """
    rgb = image.convert("RGB")
    orig_w, orig_h = rgb.size
    if orig_w < 16 or orig_h < 16:
        return []

    # 1. 作業用に長辺 1600px へ縮小(座標は元解像度へ逆変換して切り出し)
    scale = min(1.0, WORK_LONG_SIDE / max(orig_w, orig_h))
    if scale < 1.0:
        work_img = rgb.resize(
            (max(1, int(round(orig_w * scale))), max(1, int(round(orig_h * scale)))),
            Image.LANCZOS,
        )
    else:
        work_img = rgb
    work = np.asarray(work_img, dtype=np.uint8)
    wh, ww = work.shape[:2]

    # 2. 背景色推定(外周の中央値)
    bg_color = _estimate_background_color(work)

    # 3. 背景色とのユークリッド距離 > 閾値 を前景マスクに
    dist = np.linalg.norm(work.astype(np.float32) - bg_color.astype(np.float32), axis=2)
    mask = (dist > bg_threshold).astype(np.uint8) * 255

    # 4. close → open でノイズ除去
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 5. 連結成分抽出、面積 0.5% 未満は除外
    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    total_area = float(ww * wh)
    boxes = []
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        if area < total_area * MIN_AREA_FRAC:
            continue
        boxes.append([float(x), float(y), float(x + w), float(y + h)])

    if not boxes:
        return []

    # 6. 近接/重複ボックスの統合(距離: 画像幅の 2% 以内)
    boxes = _merge_close_boxes(boxes, merge_dist=ww * MERGE_DIST_FRAC)

    # 7. 行クラスタリング → 上の行から、行内は左→右
    boxes = _sort_reading_order(boxes)

    # 8. 2% パディングを付けて元解像度から切り出し
    crops = []
    inv = 1.0 / scale
    for x0, y0, x1, y1 in boxes:
        x0, y0, x1, y1 = x0 * inv, y0 * inv, x1 * inv, y1 * inv
        pad = PAD_FRAC * max(x1 - x0, y1 - y0)
        cx0 = max(0, int(round(x0 - pad)))
        cy0 = max(0, int(round(y0 - pad)))
        cx1 = min(orig_w, int(round(x1 + pad)))
        cy1 = min(orig_h, int(round(y1 + pad)))
        if cx1 - cx0 < 8 or cy1 - cy0 < 8:
            continue
        crops.append(rgb.crop((cx0, cy0, cx1, cy1)))

    return crops
