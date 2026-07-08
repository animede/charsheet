# キャラクターシート作成アプリ 仕様書・実装計画

作成日: 2026-07-08

## 1. 概要

アップロードされたキャラクター画像 1 枚から、Diffusers (Qwen-Image-Edit) の I2I 編集で
8 方向のビューを生成し、キャラクターシートとしてまとめて表示・一括ダウンロードできる
Web アプリケーション。

ComfyUI のワークフロー `/home/animede/ComfyUI/workflow/character_sheet_v1.0.json` と同等の
処理を、実証済み PoC `/home/animede/diffusers-server/poc/generate_qwen_edit.py` の方式で
Diffusers に移植する。

## 2. 生成する 8 方向

| # | 方向 | キー | プロンプト |
|---|------|-----|-----------|
| 1 | 前 | front | Show the character in a full body front view, facing directly toward the camera, neutral standing A-pose with arms slightly away from the body, full figure visible from head to toe |
| 2 | 後ろ | back | Show the character from behind in a full body back view, rear facing the camera, neutral standing pose, showing all back details, hair, costume and accessories from behind, full figure head to toe |
| 3 | 左 | left | Show the character in a full body left side profile view, character facing to the left, neutral standing pose, showing the complete left side silhouette from head to toe |
| 4 | 右 | right | Show the character in a full body right side profile view, character facing to the right, neutral standing pose, showing the complete right side silhouette from head to toe |
| 5 | 左前45度 | front_left_45 | Show the character from a 3/4 front-left angle, neutral standing pose, showing both front and left side details, full body visible from head to toe |
| 6 | 右前45度 | front_right_45 | Show the character from a 3/4 front-right angle, neutral standing pose, showing both front and right side details, full body visible from head to toe |
| 7 | 左後ろ45度 | back_left_45 | Show the character from a 3/4 back-left angle, neutral standing pose, showing back and left side details, full body visible from head to toe |
| 8 | 右後ろ45度 | back_right_45 | Show the character from a 3/4 back-right angle, neutral standing pose, showing back and right side details, full body visible from head to toe |

(1〜7 は ComfyUI ワークフローから抽出。8 は同形式で新規作成。)

ネガティブプロンプト(全方向共通、PoC と同じ):
`低分辨率，低画质，肢体畸形，手指畸形`

## 3. 生成パイプライン仕様

PoC `poc/generate_qwen_edit.py` の構成をベースに、Multiple-angles LoRA を追加する。

- **パイプライン**: `QwenImageEditPlusPipeline` (diffusers 0.39)
  - **重要**: 旧 `QwenImageEditPipeline` は不可。2509 以降の Edit モデルと Multiple-angles LoRA は
    Plus 条件付け(ComfyUI の `TextEncodeQwenImageEditPlus` 相当)で学習されており、
    旧パイプラインではキャラクター同一性が崩れ別人が生成される(2026-07-08 実測で確認済み)。
    `image` 引数はリストで渡す(`image=[img]`)
- **transformer**: `/home/animede/ComfyUI/models/diffusion_models/qwen_image_edit_2511_bf16.safetensors`
  を `Qwen/Qwen-Image-Edit-2509` の transformer config で `init_empty_weights` +
  `load_state_dict(strict=True, assign=True)` 読み込み(prefix 除去不要、bf16 キャスト、
  `__index_timestep_zero__` キーは pop)
- **processor**: `Qwen/Qwen-Image-Edit-2509` の `processor/` のみ snapshot_download
- **vae / text_encoder / tokenizer / scheduler**: `Qwen/Qwen-Image` の HF キャッシュから再利用
- **LoRA 2 本**(`load_lora_weights` を adapter_name 付きで 2 回 + `set_adapters` で両方有効化、weight 各 1.0):
  1. `/home/animede/ComfyUI/models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`
  2. `/home/animede/ComfyUI/models/loras/Qwen-Edit-2509-Multiple-angles.safetensors`
- **サンプリング設定**(ComfyUI ワークフロー準拠):
  - `num_inference_steps=4`, `true_cfg_scale=1.0`
  - `scheduler.config["shift"] = 3.0`(ModelSamplingAuraFlow shift=3 相当)
  - seed は UI から指定可(デフォルト 0、固定 seed で 8 方向生成)
- **入力前処理**: ComfyUI の `ImageScaleToTotalPixels`(1.0 megapixel) 相当。
  アスペクト比を維持したまま総画素数 ≒ 1,048,576 になるようリサイズし、幅・高さを 16 の倍数に丸める
- **実行**: 8 方向を同一入力画像に対して順次生成(直列、ジョブは同時 1 件のみ)

### VRAM 対策

GPU は RTX PRO 6000 (96GB) だが他プロセス(ComfyUI 等)が VRAM を使用中の場合がある。
モデルロード前に空き VRAM を確認し、**空きが 65GB 未満なら `pipe.enable_model_cpu_offload()`
を使う**(それ以外は `pipe.to("cuda")`)。ロードは初回生成リクエスト時の遅延ロードとし、
以後プロセス内に常駐させる。

## 4. アプリケーション構成

```
/home/animede/diffusers-server/charsheet/
├── app.py           # FastAPI アプリ本体(API + 静的ファイル配信)
├── pipeline.py      # パイプライン構築・生成処理(遅延ロード、シングルトン)
├── prompts.py       # 8 方向の定義(キー、日本語ラベル、英語ラベル、プロンプト)
├── sheet.py         # キャラクターシート合成(PIL)・ZIP 作成
├── static/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── outputs/         # ジョブごとの生成結果 outputs/{job_id}/
└── README.md        # 起動方法・API・注意事項
```

- サーバー: FastAPI + uvicorn、ポート **8600**、venv は `/home/animede/diffusers-server/venv`
- 追加インストール: `fastapi`, `uvicorn`, `python-multipart`

## 5. API 仕様

| メソッド/パス | 内容 |
|---|---|
| `POST /api/generate` | multipart で画像 (`image`) + 任意 `seed` を受け取りジョブ開始。`{job_id}` を返す。実行中ジョブがあれば 409 |
| `GET /api/jobs/{job_id}` | ジョブ状態 JSON: `status` (queued/running/done/error), `progress`(完了方向数/8), 各方向の `{key, label_ja, status, url}`, `sheet_url`, `zip_url`, `error` |
| `GET /api/jobs/{job_id}/images/{key}.png` | 各方向画像 |
| `GET /api/jobs/{job_id}/input.png` | アップロード画像(前処理後) |
| `GET /api/jobs/{job_id}/sheet.png` | キャラクターシート合成画像 |
| `GET /api/jobs/{job_id}/download.zip` | 8 方向画像 + sheet.png の一括 ZIP |
| `GET /` | `static/index.html` |

ジョブはバックグラウンドスレッドで実行し、1 方向完了するごとに `outputs/{job_id}/{key}.png`
に保存して進捗を更新する。8 方向完了後に sheet.png と download.zip を生成して `done` にする。

## 6. UI 仕様(static/index.html、素の HTML/JS/CSS)

1. **アップロード**: ファイル選択 + ドラッグ&ドロップ。選択画像をプレビュー表示。seed 入力欄(任意)。「キャラクターシート生成」ボタン
2. **各画像の表示**: 8 方向のタイルグリッド(4列×2行)。日本語ラベル(前/後ろ/左/右/左前45°/右前45°/左後ろ45°/右後ろ45°)付き。生成中はスピナー、完了したタイルから順次画像表示(1〜2 秒間隔のポーリングで `GET /api/jobs/{id}` を監視)。クリックで拡大表示
3. **キャラクターシート表示**: 全方向完了後、合成された sheet.png を大きく表示
4. **一括ダウンロード**: 「一括ダウンロード (ZIP)」ボタン → `download.zip`。sheet.png 単体ダウンロードボタンも併設
5. エラー時はメッセージ表示。生成中は再実行ボタンを無効化

## 7. キャラクターシート合成仕様 (sheet.py)

- 4 列 × 2 行のグリッド。並び順: 前 / 左前45 / 右前45 / 左 || 右 / 左後ろ45 / 右後ろ45 / 後ろ
  (1 行目が前系、2 行目が後ろ系)
- 各セル: 生成画像を等比縮小(セル内フィット)+ 下部にラベル(日本語 + 英語)
- 白背景、セル間マージン、上部にタイトル "Character Sheet" と生成日時
- PNG で保存

## 8. 実装計画(手順)

1. `charsheet/` ディレクトリ作成、`prompts.py`(8 方向定義)
2. `pipeline.py`: PoC を関数化。遅延ロード + スレッドロック。`generate_view(image, prompt, seed) -> PIL.Image`
3. `sheet.py`: グリッド合成 + ZIP
4. `app.py`: FastAPI。ジョブ管理(メモリ内 dict + スレッド)、API、静的配信
5. `static/`: UI 実装
6. 依存追加: `venv/bin/pip install fastapi uvicorn python-multipart`
7. **動作検証**:
   - `poc/output_sample.png` を入力にした CLI スモークテスト(pipeline.py 単体)で 1 方向生成
   - サーバー起動 → curl で `POST /api/generate` → ポーリング → 8 方向 + sheet + zip 生成確認
   - UI をブラウザ相当(curl / preview)で確認
8. README.md 作成(起動: `venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8600 --app-dir charsheet` 等)

## 9. 個別ビューの修正機能(refine)【追加実装済み】

生成完了後、各方向の画像に対して自由テキストの修正指示(例:「頭の黒い髪の毛を削除して」)を
出すと、その画像を入力に I2I 編集を 1 回実行して差し替える。シートと ZIP も再合成する。

### API

| メソッド/パス | 内容 |
|---|---|
| `POST /api/jobs/{job_id}/refine` | JSON: `{"key": "back", "instruction": "髪の毛を削除", "seed": 0(任意)}`。実行中の生成/refine があれば 409。プロンプトはユーザー指示に ` Keep everything else exactly the same.` を付加。実行前に現画像を `outputs/{job_id}/{key}_prev.png` にバックアップ(1 世代のみ)。完了後 `{key}.png` を上書きし sheet.png / download.zip を再生成 |
| `POST /api/jobs/{job_id}/undo` | JSON: `{"key": "back"}`。`{key}_prev.png` があれば現画像と入れ替えて復元(トグル動作: 復元後は今の画像が `_prev` になる)、sheet/zip 再生成。更新後のジョブ状態 JSON を返す |

- ジョブ状態(`GET /api/jobs/{job_id}`)に方向ごとの `refining` ステータスと
  各ビューの `has_prev`(bool)を追加。修正失敗時は `refine_error` に理由
- refine 中はジョブ全体の `status` も `refining` になり、既存のポーリングで進捗が見える

### サーバー再起動後の対応

ジョブはメモリ内管理だが、メモリに無い job_id でも `outputs/{job_id}/` に 8 方向の画像が
揃っていればディスクからジョブ情報を再構築し、refine / undo / 参照ができる。

### UI

- 完了した各タイルに「修正」ボタン。押すとタイル下にテキスト入力+「適用」ボタンを表示
  (プレースホルダー例:「例: 頭の黒い髪の毛を削除して」)
- refine 中はそのタイルにスピナー(現画像を薄く表示)、他タイルの修正ボタンは無効化
- 完了したら画像を再読み込み(キャッシュ回避に `?t=` タイムスタンプ付与)し、シート表示も更新
- `has_prev` が true のタイルには「元に戻す」ボタンを表示(undo API を呼ぶ)

### パイプラインに関する注記

pipeline.py は `QwenImageEditPlusPipeline` を使用する(`image=[image]` とリストで渡す)。
2509 以降の Edit モデル + Multiple-angles LoRA は Plus 条件付けで学習されているため、
旧 `QwenImageEditPipeline` だとキャラクター同一性が崩れる(別人が生成される)。

## 10. シート分解取り込み(split)【追加実装済み】

複数体(6〜8 体)のキャラクターが 1 枚に描かれたキャラクターシート画像を入力に使う場合、
シート全体を 1 キャラとして I2I にかけると生成結果にも複数体が写り込んで破綻する。
そこで、アップロード画像から各キャラクターを自動検出・分離し、ユーザーが 1 体を選んで
それを入力に 8 方向生成する。

### モジュール charsheet/split.py

`detect_figures(image: PIL.Image) -> list[PIL.Image]`(GPU 不使用、OpenCV のみ)。

1. RGB 変換。作業用に長辺 1600px へ縮小(座標は元解像度へ逆変換して切り出し)
2. 画像の外周 1〜2% の画素から背景色を推定(中央値)
3. 背景色とのユークリッド距離 > 閾値(30/255、調整可)を前景マスクに
4. `cv2.morphologyEx` で close → open(カーネル 7px)してノイズ除去
5. `cv2.connectedComponentsWithStats` で連結成分抽出。面積が全体の 0.5% 未満は除外
6. バウンディングボックス同士が近接/重複するものは統合(距離が画像幅の 2% 以内)
7. 行クラスタリング(y 中心でグルーピング)→ 上の行から、行内は左→右の順にソート
8. 各ボックスに 2% パディングを付けて元解像度から切り出し

検出数が 0 なら空リスト(呼び出し側で「1体扱い」にフォールバック)。

### API

| メソッド/パス | 内容 |
|---|---|
| `POST /api/split` | multipart で画像を受け取り、figure 検出して `outputs/_splits/{split_id}/figure_{i}.png` と `source.png` に保存。レスポンス: `{"split_id", "count", "figures": [{"index", "url", "width", "height"}]}`。GPU 不使用なので排他不要 |
| `GET /api/splits/{split_id}/figure_{i}.png` / `source.png` | 分割画像の配信(split_id は英数字のみ、ファイル名は `source.png` / `figure_{数字}.png` のみ許可 = パストラバーサル対策) |
| `POST /api/generate`(拡張) | 従来の multipart `image` に加え、`split_id` + `figure_index` のフォームフィールドでも受け付ける(保存済み crop を入力に使う)。両方無ければ 400 |

### UI

- ファイル選択/D&D 後、自動で `POST /api/split` を呼ぶ
- 検出数が 2 体以上: 「N体のキャラクターを検出しました。8方向生成の元にする1体を
  選んでください(通常は正面向き)」と表示し、検出 crop のサムネイルをグリッド表示。
  クリックで選択(選択枠ハイライト)。選択後「キャラクターシート生成」ボタン有効化 →
  split_id + figure_index で generate
- 検出数が 0〜1 体: 従来どおり画像全体を入力に(UI 挙動も従来のまま)
- 「シート全体をそのまま使う」リンクで、複数検出時でも従来動作を選べる

### 依存

`opencv-python-headless`(`venv/bin/python -m pip install opencv-python-headless`)

## 11. 制約・注意

- 生成・修正は 1 ジョブずつ直列(GPU 1 枚のため)。1 方向あたり数秒〜十数秒 × 8
- Multiple-angles LoRA は 2509 用だが 2511 transformer と互換(アーキテクチャ同一)。
  万一適用エラーになる場合は transformer を `qwen_image_edit_2509_fp8_e4m3fn.safetensors`
  (prefix `model.diffusion_model.` 除去 + bf16 キャスト)に切り替える
- ジョブ情報はメモリ内管理(サーバー再起動で消える)。outputs/ のファイルは残り、
  8 方向が揃っているジョブはディスクから自動復元される

## 12. 背景削除機能(remove_bg)【追加実装済み 2026-07-08】

生成済みの各方向画像から背景を除去して透過 PNG にする。方向ごと、または全方向一括で
実行でき、refine と同じ `_prev` バックアップ / undo(トグル)で元に戻せる。

### 背景除去モデル (charsheet/bg.py)

- **rembg パッケージの `isnet-general-use`(ONNX)** を使用。遅延ロードのシングルトン
- 初回実行時に ~179MB を `~/.u2net/` に自動ダウンロード
- briaai/RMBG-1.4(ローカル `/home/animede/ComfyUI/models/rmbg/RMBG-1.4/model.pth` あり)は
  transformers 5.x と remote code が非互換(`all_tied_weights_keys` エラー)のため不採用

### API

| メソッド/パス | 内容 |
|---|---|
| `POST /api/jobs/{job_id}/remove_bg` | JSON: `{"key": "front"}` または `{"key": "all"}`。実行中の処理があれば 409。各方向とも実行前に `{key}_prev.png` へバックアップ → 透過PNGで上書き → sheet.png / download.zip を再生成。ビュー状態は `removing_bg` |

### 合成・関連仕様

- sheet.py は RGBA 画像を白背景に alpha 合成(ZIP には透過 PNG のまま格納)
- 背景削除済み(透過)画像への refine は、白背景に合成してから I2I 編集する(透過→黒化防止)
- UI: 各タイルに「背景削除」ボタン、シートパネルに「全方向の背景を削除」ボタン。
  透過部分はタイル/拡大表示上で市松模様の上に表示される
