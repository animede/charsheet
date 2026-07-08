# -*- coding: utf-8 -*-
"""
8方向の視点定義。
各エントリ: key, label_ja, label_en, prompt
仕様書 docs/character_sheet_spec.md §2 準拠。
"""

NEGATIVE_PROMPT = "低分辨率，低画质，肢体畸形，手指畸形"

VIEWS = [
    {
        "key": "front",
        "label_ja": "前",
        "label_en": "Front",
        "prompt": (
            "Show the character in a full body front view, facing directly "
            "toward the camera, neutral standing A-pose with arms slightly "
            "away from the body, full figure visible from head to toe"
        ),
    },
    {
        "key": "back",
        "label_ja": "後ろ",
        "label_en": "Back",
        "prompt": (
            "Show the character from behind in a full body back view, rear "
            "facing the camera, neutral standing pose, showing all back "
            "details, hair, costume and accessories from behind, full "
            "figure head to toe"
        ),
    },
    {
        "key": "left",
        "label_ja": "左",
        "label_en": "Left",
        "prompt": (
            "Show the character in a full body left side profile view, "
            "character facing to the left, neutral standing pose, showing "
            "the complete left side silhouette from head to toe"
        ),
    },
    {
        "key": "right",
        "label_ja": "右",
        "label_en": "Right",
        "prompt": (
            "Show the character in a full body right side profile view, "
            "character facing to the right, neutral standing pose, showing "
            "the complete right side silhouette from head to toe"
        ),
    },
    {
        "key": "front_left_45",
        "label_ja": "左前45度",
        "label_en": "Front-Left 45°",
        "prompt": (
            "Show the character from a 3/4 front-left angle, neutral "
            "standing pose, showing both front and left side details, full "
            "body visible from head to toe"
        ),
    },
    {
        "key": "front_right_45",
        "label_ja": "右前45度",
        "label_en": "Front-Right 45°",
        "prompt": (
            "Show the character from a 3/4 front-right angle, neutral "
            "standing pose, showing both front and right side details, "
            "full body visible from head to toe"
        ),
    },
    {
        "key": "back_left_45",
        "label_ja": "左後ろ45度",
        "label_en": "Back-Left 45°",
        "prompt": (
            "Show the character from a 3/4 back-left angle, neutral "
            "standing pose, showing back and left side details, full body "
            "visible from head to toe"
        ),
    },
    {
        "key": "back_right_45",
        "label_ja": "右後ろ45度",
        "label_en": "Back-Right 45°",
        "prompt": (
            "Show the character from a 3/4 back-right angle, neutral "
            "standing pose, showing back and right side details, full body "
            "visible from head to toe"
        ),
    },
]

VIEW_BY_KEY = {v["key"]: v for v in VIEWS}

# キャラクターシート合成時の並び順(仕様書 §7)
# 1行目: 前系, 2行目: 後ろ系
SHEET_LAYOUT = [
    ["front", "front_left_45", "front_right_45", "left"],
    ["right", "back_left_45", "back_right_45", "back"],
]
