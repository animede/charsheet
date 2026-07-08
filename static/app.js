(() => {
  "use strict";

  const VIEW_ORDER = [
    "front", "back", "left", "right",
    "front_left_45", "front_right_45", "back_left_45", "back_right_45",
  ];
  const VIEW_LABELS = {
    front: { ja: "前", en: "Front" },
    back: { ja: "後ろ", en: "Back" },
    left: { ja: "左", en: "Left" },
    right: { ja: "右", en: "Right" },
    front_left_45: { ja: "左前45度", en: "Front-Left 45°" },
    front_right_45: { ja: "右前45度", en: "Front-Right 45°" },
    back_left_45: { ja: "左後ろ45度", en: "Back-Left 45°" },
    back_right_45: { ja: "右後ろ45度", en: "Back-Right 45°" },
  };

  const dropzone = document.getElementById("dropzone");
  const dropzoneText = document.getElementById("dropzone-text");
  const fileInput = document.getElementById("file-input");
  const previewImage = document.getElementById("preview-image");
  const seedInput = document.getElementById("seed-input");
  const generateBtn = document.getElementById("generate-btn");
  const errorMessage = document.getElementById("error-message");
  const statusMessage = document.getElementById("status-message");
  const viewsGrid = document.getElementById("views-grid");
  const sheetPanel = document.getElementById("sheet-panel");
  const sheetImage = document.getElementById("sheet-image");
  const downloadZipBtn = document.getElementById("download-zip-btn");
  const downloadSheetBtn = document.getElementById("download-sheet-btn");
  const removeBgAllBtn = document.getElementById("remove-bg-all-btn");
  const lightbox = document.getElementById("lightbox");
  const lightboxImage = document.getElementById("lightbox-image");
  const lightboxClose = document.getElementById("lightbox-close");

  const splitSection = document.getElementById("split-section");
  const splitMessage = document.getElementById("split-message");
  const splitGrid = document.getElementById("split-grid");
  const useWholeLink = document.getElementById("use-whole-link");

  let selectedFile = null;
  let pollTimer = null;
  let currentJobId = null;
  let busy = false; // 生成 or refine 実行中

  // シート分解(split)状態
  let inputMode = "file"; // "file" = 画像全体 / "figure" = 分割 crop を入力に使う
  let currentSplitId = null;
  let selectedFigureIndex = null;

  function showError(msg) {
    errorMessage.textContent = msg;
    errorMessage.classList.remove("hidden");
  }

  function clearError() {
    errorMessage.textContent = "";
    errorMessage.classList.add("hidden");
  }

  function showStatus(msg) {
    statusMessage.textContent = msg;
    statusMessage.classList.remove("hidden");
  }

  function clearStatus() {
    statusMessage.textContent = "";
    statusMessage.classList.add("hidden");
  }

  function buildViewTiles() {
    viewsGrid.innerHTML = "";
    for (const key of VIEW_ORDER) {
      const label = VIEW_LABELS[key];
      const tile = document.createElement("div");
      tile.className = "view-tile";
      tile.id = `tile-${key}`;

      const imgWrap = document.createElement("div");
      imgWrap.className = "view-tile-image-wrap";

      const spinner = document.createElement("div");
      spinner.className = "spinner";
      spinner.id = `spinner-${key}`;

      const img = document.createElement("img");
      img.id = `img-${key}`;
      img.alt = label.ja;
      img.addEventListener("click", () => {
        if (img.src && img.style.display !== "none") {
          openLightbox(img.src);
        }
      });

      const statusBadge = document.createElement("div");
      statusBadge.className = "view-tile-status";
      statusBadge.id = `status-${key}`;
      statusBadge.textContent = "待機中";

      imgWrap.appendChild(spinner);
      imgWrap.appendChild(img);
      imgWrap.appendChild(statusBadge);

      const labelDiv = document.createElement("div");
      labelDiv.className = "view-tile-label";
      labelDiv.innerHTML = `<div class="ja">${label.ja}</div><div class="en">${label.en}</div>`;

      // --- 修正(refine)UI ---
      const actions = document.createElement("div");
      actions.className = "view-tile-actions";
      actions.id = `actions-${key}`;

      const refineToggleBtn = document.createElement("button");
      refineToggleBtn.className = "tile-btn";
      refineToggleBtn.id = `refine-toggle-${key}`;
      refineToggleBtn.textContent = "修正";
      refineToggleBtn.disabled = true;
      refineToggleBtn.addEventListener("click", () => toggleRefineForm(key));

      const removeBgBtn = document.createElement("button");
      removeBgBtn.className = "tile-btn secondary";
      removeBgBtn.id = `remove-bg-${key}`;
      removeBgBtn.textContent = "背景削除";
      removeBgBtn.disabled = true;
      removeBgBtn.addEventListener("click", () => doRemoveBg(key));

      const undoBtn = document.createElement("button");
      undoBtn.className = "tile-btn secondary hidden";
      undoBtn.id = `undo-${key}`;
      undoBtn.textContent = "元に戻す";
      undoBtn.addEventListener("click", () => doUndo(key));

      actions.appendChild(refineToggleBtn);
      actions.appendChild(removeBgBtn);
      actions.appendChild(undoBtn);

      const refineForm = document.createElement("div");
      refineForm.className = "refine-form hidden";
      refineForm.id = `refine-form-${key}`;

      const refineInput = document.createElement("input");
      refineInput.type = "text";
      refineInput.id = `refine-input-${key}`;
      refineInput.placeholder = "例: 頭の黒い髪の毛を削除して";
      refineInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") doRefine(key);
      });

      const refineApplyBtn = document.createElement("button");
      refineApplyBtn.className = "tile-btn";
      refineApplyBtn.id = `refine-apply-${key}`;
      refineApplyBtn.textContent = "適用";
      refineApplyBtn.addEventListener("click", () => doRefine(key));

      refineForm.appendChild(refineInput);
      refineForm.appendChild(refineApplyBtn);

      tile.appendChild(imgWrap);
      tile.appendChild(labelDiv);
      tile.appendChild(actions);
      tile.appendChild(refineForm);
      viewsGrid.appendChild(tile);
    }
  }

  function toggleRefineForm(key) {
    const form = document.getElementById(`refine-form-${key}`);
    if (!form) return;
    const willShow = form.classList.contains("hidden");
    // 他のフォームは閉じる
    for (const k of VIEW_ORDER) {
      const f = document.getElementById(`refine-form-${k}`);
      if (f) f.classList.add("hidden");
    }
    if (willShow) {
      form.classList.remove("hidden");
      const input = document.getElementById(`refine-input-${key}`);
      if (input) input.focus();
    }
  }

  function setRefineButtonsEnabled(enabled) {
    for (const key of VIEW_ORDER) {
      const toggleBtn = document.getElementById(`refine-toggle-${key}`);
      const applyBtn = document.getElementById(`refine-apply-${key}`);
      const undoBtn = document.getElementById(`undo-${key}`);
      const bgBtn = document.getElementById(`remove-bg-${key}`);
      const img = document.getElementById(`img-${key}`);
      const hasImage = img && img.style.display !== "none" && img.src;
      if (toggleBtn) toggleBtn.disabled = !enabled || !hasImage;
      if (applyBtn) applyBtn.disabled = !enabled;
      if (undoBtn) undoBtn.disabled = !enabled;
      if (bgBtn) bgBtn.disabled = !enabled || !hasImage;
    }
    if (removeBgAllBtn) removeBgAllBtn.disabled = !enabled;
  }

  function updateViewTile(view) {
    const key = view.key;
    const spinner = document.getElementById(`spinner-${key}`);
    const img = document.getElementById(`img-${key}`);
    const statusBadge = document.getElementById(`status-${key}`);
    const undoBtn = document.getElementById(`undo-${key}`);
    if (!spinner || !img || !statusBadge) return;

    const statusText = {
      queued: "待機中",
      running: "生成中",
      refining: "修正中",
      removing_bg: "背景削除中",
      done: "完了",
      error: "エラー",
    };
    statusBadge.textContent = statusText[view.status] || view.status;

    if (view.status === "done" && view.url) {
      spinner.style.display = "none";
      img.style.display = "block";
      img.style.opacity = "1";
      // ポーリングごとの無駄な再読込を避けつつ、refine/undo 後は
      // data-reload フラグでキャッシュバスティング付き再読込を行う
      if (!img.src || img.dataset.reload === "1") {
        img.src = view.url + "?t=" + Date.now();
        img.dataset.reload = "0";
      }
    } else if (view.status === "refining" || view.status === "removing_bg") {
      // refine / 背景削除中は現画像を薄く表示したままスピナーを重ねる
      spinner.style.display = "block";
      img.style.opacity = "0.35";
      img.dataset.reload = "1";
    } else if (view.status === "running") {
      spinner.style.display = "block";
      img.style.display = "none";
      img.dataset.reload = "1";
    } else if (view.status === "error") {
      spinner.style.display = "none";
      img.style.display = "none";
    } else {
      spinner.style.display = "none";
      img.style.display = "none";
    }

    if (undoBtn) {
      if (view.has_prev) {
        undoBtn.classList.remove("hidden");
      } else {
        undoBtn.classList.add("hidden");
      }
    }
  }

  function openLightbox(src) {
    lightboxImage.src = src;
    lightbox.classList.remove("hidden");
  }

  function closeLightbox() {
    lightbox.classList.add("hidden");
    lightboxImage.src = "";
  }

  lightboxClose.addEventListener("click", closeLightbox);
  lightbox.addEventListener("click", (e) => {
    if (e.target === lightbox) closeLightbox();
  });

  // --- ファイル選択 / D&D ---
  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files.length > 0) {
      handleFileSelect(fileInput.files[0]);
    }
  });

  function handleFileSelect(file) {
    if (!file.type.startsWith("image/")) {
      showError("画像ファイルを選択してください");
      return;
    }
    clearError();
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      previewImage.classList.remove("hidden");
      dropzoneText.classList.add("hidden");
    };
    reader.readAsDataURL(file);

    // シート分解を自動実行
    resetSplitState();
    runSplit(file);
  }

  // --- シート分解(split) ---
  function resetSplitState() {
    inputMode = "file";
    currentSplitId = null;
    selectedFigureIndex = null;
    splitSection.classList.add("hidden");
    splitGrid.innerHTML = "";
    splitMessage.textContent = "";
  }

  async function runSplit(file) {
    generateBtn.disabled = true;
    showStatus("キャラクターを検出中...");
    try {
      const formData = new FormData();
      formData.append("image", file);
      const resp = await fetch("/api/split", { method: "POST", body: formData });
      if (!resp.ok) {
        // 検出失敗時は従来どおり画像全体を入力に
        clearStatus();
        generateBtn.disabled = false;
        return;
      }
      const data = await resp.json();
      clearStatus();
      if (data.count >= 2) {
        currentSplitId = data.split_id;
        showSplitSelection(data);
        // 1体選択するまで生成ボタンは無効
        generateBtn.disabled = true;
      } else {
        // 0〜1体: 従来どおり画像全体を入力に(UI 挙動も従来のまま)
        generateBtn.disabled = false;
      }
    } catch (err) {
      // 通信エラー時も従来どおりのフローにフォールバック
      console.error(err);
      clearStatus();
      generateBtn.disabled = false;
    }
  }

  function showSplitSelection(data) {
    splitMessage.textContent =
      `${data.count}体のキャラクターを検出しました。8方向生成の元にする1体を選んでください(通常は正面向き)。`;
    splitGrid.innerHTML = "";
    for (const fig of data.figures) {
      const cell = document.createElement("div");
      cell.className = "split-cell";
      cell.id = `split-cell-${fig.index}`;

      const img = document.createElement("img");
      img.src = fig.url;
      img.alt = `キャラクター ${fig.index + 1}`;

      const num = document.createElement("div");
      num.className = "split-cell-num";
      num.textContent = String(fig.index + 1);

      cell.appendChild(img);
      cell.appendChild(num);
      cell.addEventListener("click", () => selectFigure(fig.index));
      splitGrid.appendChild(cell);
    }
    splitSection.classList.remove("hidden");
  }

  function selectFigure(index) {
    inputMode = "figure";
    selectedFigureIndex = index;
    for (const cell of splitGrid.children) {
      cell.classList.toggle("selected", cell.id === `split-cell-${index}`);
    }
    generateBtn.disabled = false;
    clearError();
    showStatus(`キャラクター ${index + 1} を選択しました。`);
  }

  useWholeLink.addEventListener("click", (e) => {
    e.preventDefault();
    inputMode = "file";
    selectedFigureIndex = null;
    for (const cell of splitGrid.children) {
      cell.classList.remove("selected");
    }
    generateBtn.disabled = !selectedFile;
    showStatus("シート全体をそのまま入力に使います。");
  });

  // --- 生成開始 ---
  generateBtn.addEventListener("click", async () => {
    if (!selectedFile) {
      showError("画像を選択してください");
      return;
    }
    clearError();
    clearStatus();
    generateBtn.disabled = true;
    busy = true;
    sheetPanel.classList.add("hidden");
    buildViewTiles();

    const formData = new FormData();
    if (inputMode === "figure" && currentSplitId !== null && selectedFigureIndex !== null) {
      formData.append("split_id", currentSplitId);
      formData.append("figure_index", selectedFigureIndex);
    } else {
      formData.append("image", selectedFile);
    }
    const seed = parseInt(seedInput.value, 10) || 0;
    formData.append("seed", seed);

    try {
      const resp = await fetch("/api/generate", {
        method: "POST",
        body: formData,
      });
      if (resp.status === 409) {
        const data = await resp.json().catch(() => ({}));
        showError(data.detail || "別のジョブが実行中です。しばらく待ってから再試行してください。");
        generateBtn.disabled = false;
        busy = false;
        return;
      }
      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        showError(data.detail || `生成開始に失敗しました (status ${resp.status})`);
        generateBtn.disabled = false;
        busy = false;
        return;
      }
      const data = await resp.json();
      currentJobId = data.job_id;
      showStatus("生成を開始しました。モデルの初回ロードには数分かかる場合があります。");
      startPolling(currentJobId);
    } catch (err) {
      showError("通信エラー: " + err.message);
      generateBtn.disabled = false;
      busy = false;
    }
  });

  // --- 修正(refine) ---
  async function doRefine(key) {
    if (!currentJobId || busy) return;
    const input = document.getElementById(`refine-input-${key}`);
    const instruction = (input ? input.value : "").trim();
    if (!instruction) {
      showError("修正指示を入力してください");
      return;
    }
    clearError();
    busy = true;
    setRefineButtonsEnabled(false);
    generateBtn.disabled = true;
    const form = document.getElementById(`refine-form-${key}`);
    if (form) form.classList.add("hidden");

    const seed = parseInt(seedInput.value, 10) || 0;

    try {
      const resp = await fetch(`/api/jobs/${currentJobId}/refine`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key, instruction, seed }),
      });
      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        showError(data.detail || `修正の開始に失敗しました (status ${resp.status})`);
        busy = false;
        setRefineButtonsEnabled(true);
        generateBtn.disabled = !selectedFile;
        return;
      }
      showStatus(`「${VIEW_LABELS[key].ja}」を修正中...`);
      startPolling(currentJobId);
    } catch (err) {
      showError("通信エラー: " + err.message);
      busy = false;
      setRefineButtonsEnabled(true);
      generateBtn.disabled = !selectedFile;
    }
  }

  // --- 背景削除(remove_bg) ---
  async function doRemoveBg(key) {
    if (!currentJobId || busy) return;
    clearError();
    busy = true;
    setRefineButtonsEnabled(false);
    generateBtn.disabled = true;

    try {
      const resp = await fetch(`/api/jobs/${currentJobId}/remove_bg`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key }),
      });
      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        showError(data.detail || `背景削除の開始に失敗しました (status ${resp.status})`);
        busy = false;
        setRefineButtonsEnabled(true);
        generateBtn.disabled = !selectedFile;
        return;
      }
      showStatus(key === "all" ? "全方向の背景を削除中..." : `「${VIEW_LABELS[key].ja}」の背景を削除中...`);
      startPolling(currentJobId);
    } catch (err) {
      showError("通信エラー: " + err.message);
      busy = false;
      setRefineButtonsEnabled(true);
      generateBtn.disabled = !selectedFile;
    }
  }

  // --- 元に戻す(undo) ---
  async function doUndo(key) {
    if (!currentJobId || busy) return;
    clearError();
    busy = true;
    setRefineButtonsEnabled(false);

    try {
      const resp = await fetch(`/api/jobs/${currentJobId}/undo`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key }),
      });
      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        showError(data.detail || `復元に失敗しました (status ${resp.status})`);
        return;
      }
      const job = await resp.json();
      // 対象画像とシートを強制再読み込み
      const img = document.getElementById(`img-${key}`);
      if (img) img.dataset.reload = "1";
      for (const view of job.views) {
        updateViewTile(view);
      }
      refreshSheet(job);
      showStatus(`「${VIEW_LABELS[key].ja}」を入れ替えました(もう一度押すと戻ります)。`);
    } catch (err) {
      showError("通信エラー: " + err.message);
    } finally {
      busy = false;
      setRefineButtonsEnabled(true);
    }
  }

  function refreshSheet(job) {
    if (job.sheet_url) {
      sheetImage.src = job.sheet_url + "?t=" + Date.now();
      sheetPanel.classList.remove("hidden");
      downloadZipBtn.dataset.url = job.zip_url;
      downloadSheetBtn.dataset.url = job.sheet_url;
    }
  }

  function startPolling(jobId) {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(() => pollJob(jobId), 1500);
    pollJob(jobId);
  }

  async function pollJob(jobId) {
    try {
      const resp = await fetch(`/api/jobs/${jobId}`);
      if (!resp.ok) {
        return;
      }
      const job = await resp.json();

      for (const view of job.views) {
        updateViewTile(view);
      }

      if (job.status === "running" || job.status === "queued") {
        showStatus(`生成中... (${job.progress} / ${job.total} 方向完了)`);
      } else if (job.status === "refining") {
        showStatus("修正を適用中...");
      } else if (job.status === "removing_bg") {
        showStatus("背景を削除中...");
      }

      if (job.status === "done") {
        clearInterval(pollTimer);
        pollTimer = null;
        busy = false;
        generateBtn.disabled = !selectedFile;
        clearStatus();
        if (job.refine_error) {
          showError("修正エラー: " + job.refine_error);
        } else {
          showStatus("完了しました。各タイルの「修正」ボタンで個別に修正できます。");
        }
        refreshSheet(job);
        setRefineButtonsEnabled(true);
      } else if (job.status === "error") {
        clearInterval(pollTimer);
        pollTimer = null;
        busy = false;
        generateBtn.disabled = !selectedFile;
        clearStatus();
        showError("生成エラー: " + (job.error || "不明なエラー"));
        setRefineButtonsEnabled(true);
      }
    } catch (err) {
      // ネットワーク瞬断は無視して次のポーリングを待つ
      console.error(err);
    }
  }

  downloadZipBtn.addEventListener("click", () => {
    const url = downloadZipBtn.dataset.url;
    if (url) window.location.href = url;
  });

  removeBgAllBtn.addEventListener("click", () => doRemoveBg("all"));

  downloadSheetBtn.addEventListener("click", () => {
    const url = downloadSheetBtn.dataset.url;
    if (!url) return;
    const a = document.createElement("a");
    a.href = url;
    a.download = "character_sheet.png";
    document.body.appendChild(a);
    a.click();
    a.remove();
  });

  buildViewTiles();
})();
