# charsheet — キャラクターシート作成 Web アプリ

1枚のキャラクター画像から Qwen-Image-Edit (Diffusers) の I2I 編集で8方向ビューを生成し、
シート合成・一括DLできる FastAPI アプリ。詳細仕様は docs/character_sheet_spec.md、
API 一覧と使い方は README.md を参照。

## 起動

```bash
./venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8600
```

## 構成

- `pipeline.py` — 生成パイプライン(シングルトン遅延ロード)。
  **必ず `QwenImageEditPlusPipeline` を使うこと**(旧 `QwenImageEditPipeline` に戻すと
  Multiple-angles LoRA が機能せずキャラ同一性が崩れる。`image=[img]` とリストで渡す)
- `prompts.py` — 8方向の定義(キー・ラベル・プロンプト)とシート並び順
- `app.py` — FastAPI 本体。ジョブはメモリ内 dict + ディスク復元、GPU 処理は同時1件(409 排他)
- `sheet.py` — 4×2 グリッド合成(RGBA は白背景に alpha 合成)・ZIP
- `split.py` — 複数体シートのキャラクター自動検出・分離(OpenCV、背景色ベース)
- `bg.py` — 背景除去(rembg / isnet-general-use)。RMBG-1.4 は transformers 5.x 非互換で不採用
- `static/` — 素の HTML/JS/CSS(日本語 UI)

## 外部依存

- モデル重み: `$COMFYUI_DIR/models/`(既定 `~/ComfyUI`、環境変数で上書き可)配下を優先参照
  (pipeline.py 冒頭の定数)。無ければ `_resolve_model_path()` が Comfy-Org / lightx2v の
  HF リポジトリから自動ダウンロードし、通常の HF キャッシュに保存する(README.md の対応表を参照)
- HF キャッシュ: `Qwen/Qwen-Image`(vae/text_encoder 等)、`Qwen/Qwen-Image-Edit-2509`(processor)
- GPU: 空き VRAM 65GB 未満だと自動で `enable_model_cpu_offload()`(遅くなる)

## 注意

- `outputs/` はジョブ成果物(git 管理外)。`{key}_prev.png` は refine/背景削除の1世代バックアップ
- 生成パラメータ(steps=4, true_cfg_scale=1.0, shift=3.0, 入力1MP)は ComfyUI の
  character_sheet_v1.0 ワークフロー準拠。変更時は docs/character_sheet_spec.md も更新すること
