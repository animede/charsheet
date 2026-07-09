# キャラクターシート作成 Web アプリ

アップロードされたキャラクター画像 1 枚から、Qwen-Image-Edit (Diffusers) の I2I 編集で
8 方向のビューを生成し、キャラクターシートとしてまとめて表示・一括ダウンロードできる
Web アプリケーションです。

仕様書: `docs/character_sheet_spec.md`
使い方マニュアル(エンドユーザー向け): `docs/user_manual.md`

## セットアップ

```bash
cd charsheet
python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

## 起動方法

```bash
cd charsheet
./venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8600
```

ブラウザで `http://localhost:8600/` を開くと UI が表示されます。

## 外部依存(モデルファイル)

以下は ComfyUI のモデルディレクトリを優先的に参照する(pipeline.py 冒頭の定数)。
**すでに ComfyUI で同じモデル(下表)をお使いの場合は、そのファイルがそのまま
再利用されるため、あらためてダウンロードする必要はありません。**
ComfyUI のインストール先は既定で `~/ComfyUI`(実行ユーザーのホームディレクトリ配下)を
自動参照する。別の場所にある場合は環境変数 `COMFYUI_DIR` で上書きできる:

```bash
export COMFYUI_DIR=/path/to/ComfyUI
```

**該当パスにファイルが無い場合のみ、自動で Hugging Face Hub からダウンロードして
通常の HF キャッシュ(`~/.cache/huggingface/hub/`)に保存する**(2 回目以降は
再ダウンロードしない)。ComfyUI が無い環境でも、この自動ダウンロードだけで単独で動作する。

| ローカル優先パス(`$COMFYUI_DIR/models/...`) | 無い場合の自動ダウンロード元 |
|---|---|
| `diffusion_models/qwen_image_edit_2511_bf16.safetensors` | `Comfy-Org/Qwen-Image-Edit_ComfyUI`(split_files/diffusion_models/) |
| `loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` | `lightx2v/Qwen-Image-Edit-2511-Lightning` |
| `loras/Qwen-Edit-2509-Multiple-angles.safetensors` | `Comfy-Org/Qwen-Image-Edit_ComfyUI`(split_files/loras/) |
| フォールバック用 `diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors` | `Comfy-Org/Qwen-Image-Edit_ComfyUI`(split_files/diffusion_models/) |

vae / text_encoder / tokenizer / scheduler は HuggingFace キャッシュ(`Qwen/Qwen-Image`)、
processor は `Qwen/Qwen-Image-Edit-2509` から取得(こちらは元々 HF から取得する設計で変更なし)。
背景除去モデル(isnet-general-use, ~179MB)は初回実行時に `~/.u2net/` へ自動ダウンロードされる。

transformer は数 GB〜20GB 程度あるため、ComfyUI 側にファイルが無い環境での初回ロードは
ダウンロードに時間がかかる(回線速度に依存)。ダウンロード中はサーバーログ
(`server.log`)に進捗が出力される。

## API 一覧

| メソッド / パス | 内容 |
|---|---|
| `POST /api/generate` | multipart で `image` + `seed`(任意、デフォルト 0)を受け取りジョブ開始。`{"job_id": "..."}` を返す。実行中ジョブがあれば **409**。`image` の代わりに `split_id` + `figure_index` フォームフィールドでも受け付け(分割済み crop を入力に使う)。両方無ければ **400** |
| `POST /api/split` | **シート分解**。multipart で `image` を受け取り、複数キャラクターを自動検出・分離して `outputs/_splits/{split_id}/` に保存。レスポンス: `{"split_id", "count", "figures": [{"index", "url", "width", "height"}]}`。GPU 不使用のため排他なし |
| `GET /api/splits/{split_id}/figure_{i}.png` | 分割された各キャラクターの切り出し画像 |
| `GET /api/splits/{split_id}/source.png` | 分割元画像 |
| `GET /api/jobs/{job_id}` | ジョブ状態 JSON。`status` (queued / running / done / error)、`progress`(完了方向数)、`total`、各方向の `{key, label_ja, label_en, status, url}`、`sheet_url`、`zip_url`、`error` |
| `GET /api/jobs/{job_id}/images/{key}.png` | 各方向の生成画像 |
| `GET /api/jobs/{job_id}/input.png` | アップロード画像(前処理後: 総画素数 ≒ 1MP、16 の倍数) |
| `GET /api/jobs/{job_id}/sheet.png` | キャラクターシート合成画像(4 列 × 2 行) |
| `GET /api/jobs/{job_id}/download.zip` | 8 方向画像 + sheet.png の一括 ZIP |
| `POST /api/jobs/{job_id}/refine` | **個別ビューの修正**。JSON `{"key": "back", "instruction": "髪の毛を削除", "seed": 0}`(seed は任意)。指定方向の現画像を入力に修正指示を I2I で 1 回適用して差し替え、sheet.png / download.zip を再生成。実行中の生成/修正があれば **409**。実行前に現画像を `{key}_prev.png` にバックアップ(1 世代のみ) |
| `POST /api/jobs/{job_id}/undo` | **修正の取り消し**。JSON `{"key": "back"}`。`{key}_prev.png` と現画像を入れ替えて復元し、sheet/zip を再生成。**トグル動作**(もう一度呼ぶと再度修正版に戻る)。更新後のジョブ状態 JSON を返す |
| `POST /api/jobs/{job_id}/remove_bg` | **背景削除**。JSON `{"key": "front"}` または `{"key": "all"}`(全方向一括)。rembg (ISNet) で背景を除去し透過 PNG に差し替え、sheet.png(白背景に合成)/ download.zip(透過のまま)を再生成。実行前に `{key}_prev.png` にバックアップされるので undo で復元可。実行中の処理があれば **409** |
| `GET /` | UI (`static/index.html`) |

- `GET /api/jobs/{job_id}` のジョブ状態 JSON: `status` に `refining` / `removing_bg` が追加。各方向に
  `has_prev`(bool、バックアップの有無)が含まれる。修正/背景削除の失敗時は `refine_error` に理由が入る
- **背景削除モデル**: rembg パッケージの `isnet-general-use`(ONNX)。初回実行時に
  ~179MB を `~/.u2net/` にダウンロードする。briaai/RMBG-1.4 の transformers remote code は
  transformers 5.x と非互換のため不採用(bg.py のコメント参照)
- 背景削除済み(透過)の画像に refine をかける場合は、白背景に合成してから編集される
- **サーバー再起動後**: ジョブ情報はメモリ内管理だが、`outputs/{job_id}/` に 8 方向の画像が
  揃っていればディスクからジョブを自動復元するため、過去ジョブへの参照 / refine / undo が可能

方向キー(8 方向): `front`, `back`, `left`, `right`, `front_left_45`,
`front_right_45`, `back_left_45`, `back_right_45`

curl での使用例:

```bash
# ジョブ開始
curl -X POST http://localhost:8600/api/generate \
  -F "image=@character.png" -F "seed=0"
# -> {"job_id": "xxxxxxxxxxxx"}

# 進捗確認(1〜2 秒間隔でポーリング)
curl http://localhost:8600/api/jobs/xxxxxxxxxxxx

# 完了後に ZIP ダウンロード
curl -O http://localhost:8600/api/jobs/xxxxxxxxxxxx/download.zip

# 個別ビューの修正(例: back の黒い髪を削除)
curl -X POST http://localhost:8600/api/jobs/xxxxxxxxxxxx/refine \
  -H "Content-Type: application/json" \
  -d '{"key": "back", "instruction": "頭の黒い髪の毛を削除して", "seed": 0}'

# 修正の取り消し(トグル)
curl -X POST http://localhost:8600/api/jobs/xxxxxxxxxxxx/undo \
  -H "Content-Type: application/json" \
  -d '{"key": "back"}'

# シート分解(複数キャラクターの検出)
curl -X POST http://localhost:8600/api/split -F "image=@multi_char_sheet.png"
# -> {"split_id": "yyyy", "count": 8, "figures": [...]}

# 分割 crop を入力に 8 方向生成
curl -X POST http://localhost:8600/api/generate \
  -F "split_id=yyyy" -F "figure_index=0" -F "seed=0"
```

## 構成

```
charsheet/
├── app.py           # FastAPI アプリ本体(API + 静的ファイル配信)
├── pipeline.py      # パイプライン構築・生成処理(遅延ロード、シングルトン)
├── prompts.py       # 8 方向の定義(キー、日本語/英語ラベル、プロンプト)
├── sheet.py         # キャラクターシート合成(PIL)・ZIP 作成
├── split.py         # シート分解: 複数キャラクターの検出・分離(OpenCV、GPU 不使用)
├── static/          # UI(素の HTML/JS/CSS、日本語)
└── outputs/         # ジョブごとの生成結果 outputs/{job_id}/、分割結果 outputs/_splits/{split_id}/
```

## 使用モデル

- transformer: `$COMFYUI_DIR/models/diffusion_models/qwen_image_edit_2511_bf16.safetensors`
- LoRA(両方 weight 1.0 で適用):
  - `$COMFYUI_DIR/models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`
  - `$COMFYUI_DIR/models/loras/Qwen-Edit-2509-Multiple-angles.safetensors`
- vae / text_encoder / tokenizer / scheduler: `Qwen/Qwen-Image` の HF キャッシュを再利用
- processor: `Qwen/Qwen-Image-Edit-2509` の `processor/` のみ取得
- サンプリング: steps=4, true_cfg_scale=1.0, shift=3.0(Lightning 4steps LoRA 用)
- パイプライン: `QwenImageEditPlusPipeline`(2509 以降の Edit モデル + Multiple-angles LoRA
  は Plus 条件付けで学習されているため。旧 `QwenImageEditPipeline` だとキャラクター同一性が崩れる)

## UI での修正機能

生成完了後、各タイルの「修正」ボタンで自由テキストの修正指示
(例:「頭の黒い髪の毛を削除して」)を入力し「適用」すると、その方向の画像だけを
I2I 編集で差し替えます。修正後は「元に戻す」ボタンで修正前と入れ替えできます
(トグル動作)。修正中は他タイルの操作は無効化されます。修正指示には
` Keep everything else exactly the same.` が自動付加され、指示以外の部分は極力保持されます。

## シート分解取り込み(split)

複数体(6〜8 体など)のキャラクターが 1 枚に描かれたキャラクターシート画像を
アップロードすると、自動で各キャラクターを検出・分離します(`POST /api/split` が
ファイル選択/D&D 直後に自動実行されます)。

- **2 体以上検出時**: 「N体のキャラクターを検出しました。8方向生成の元にする1体を
  選んでください(通常は正面向き)」と表示され、検出 crop のサムネイルがグリッド表示
  されます。クリックで 1 体を選択(選択枠ハイライト)してから生成すると、その crop
  だけを入力に 8 方向生成します。「シート全体をそのまま使う」リンクで従来動作も選べます
- **0〜1 体検出時**: 従来どおり画像全体を入力に使います(UI 挙動も従来のまま)
- 検出アルゴリズム: 外周画素から背景色を推定 → 背景色との距離で前景マスク →
  モルフォロジーでノイズ除去 → 連結成分抽出(面積 0.5% 未満除外)→ 近接ボックス統合 →
  行クラスタリングで上の行から左→右に並べ、2% パディング付きで元解像度から切り出し
- 依存: `opencv-python-headless`(未インストールなら
  `venv/bin/python -m pip install opencv-python-headless`)

## 注意事項

- **初回生成リクエスト時にモデルを遅延ロード**します(30 秒〜数分)。以後はプロセス内に常駐します
- **同時実行は 1 ジョブのみ**(生成・修正共通)。実行中に `POST /api/generate` や
  `POST .../refine` を呼ぶと 409 が返ります
- **VRAM 対策**: モデルロード時に空き VRAM を確認し、空きが 65GB 未満の場合は
  自動的に `enable_model_cpu_offload()` を使用します(生成は遅くなります)。
  ComfyUI 等の他プロセスが VRAM を大量使用している場合は事前に停止を推奨
- ジョブ情報はメモリ内管理のため、**サーバー再起動で消えます**(`outputs/` のファイルは残る)
- 生成時間の目安: モデルロード後、1 方向あたり約 4〜5 秒 × 8 方向 ≒ 40 秒
  (RTX PRO 6000, cpu_offload なしの場合)
- 万一 Multiple-angles LoRA が 2511 transformer と非互換になった場合は、自動的に
  `qwen_image_edit_2509_fp8_e4m3fn.safetensors` にフォールバックします
  (ログに `falling back to 2509 fp8` と出力される)

## ライセンス

このリポジトリのコード(アプリ本体)は [MIT License](LICENSE) です。

利用する各モデル(Qwen-Image-Edit、Lightning LoRA、Multiple-angles LoRA、
背景除去モデルなど)は、それぞれの配布元が定めるライセンス・利用規約に従います。
本アプリはそれらのモデルを同梱しておらず、実行時に参照・ダウンロードするのみです。
商用利用や再配布を行う場合は、各モデルのライセンスを個別にご確認ください。
