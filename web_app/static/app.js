// ============================================================================
// HỆ THỐNG GIÁM SÁT & PHÁT HIỆN NGƯỜI ĐUỐI NƯỚC - FRONTEND CONTROLLER
// ============================================================================

let evalDataCache = null;
let audioCtx = null;
let lastSirenTime = 0;
let mutedTemporarily = false;
let lastAlertCountKnown = 0;

// ----------------------------------------------------------------------------
// 1. ÂM THANH CẢNH BÁO KHẨN CẤP (WEB AUDIO API SIREN + VOICE)
// ----------------------------------------------------------------------------
function playEmergencySiren(force = false) {
  const chkAudio = document.getElementById("chkAudioAlert");
  if (!force && (!chkAudio || !chkAudio.checked || mutedTemporarily)) return;

  const now = Date.now();
  if (!force && now - lastSirenTime < 3200) return;
  lastSirenTime = now;

  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === "suspended") {
      audioCtx.resume();
    }

    // Còi báo động 2 âm tần (880Hz - 660Hz)
    const freqs = [880, 660, 880, 660];
    freqs.forEach((freq, idx) => {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(freq, audioCtx.currentTime + idx * 0.25);

      gain.gain.setValueAtTime(0.18, audioCtx.currentTime + idx * 0.25);
      gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + (idx + 1) * 0.25);

      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start(audioCtx.currentTime + idx * 0.25);
      osc.stop(audioCtx.currentTime + (idx + 1) * 0.25);
    });

    // Phát giọng đọc cảnh báo tiếng Việt
    if ("speechSynthesis" in window && !window.speechSynthesis.speaking) {
      const utter = new SpeechSynthesisUtterance("Cảnh báo khẩn cấp! Phát hiện người đuối nước!");
      utter.lang = "vi-VN";
      utter.rate = 1.1;
      window.speechSynthesis.speak(utter);
    }
  } catch (err) {
    console.warn("Không thể phát âm thanh:", err);
  }
}

// ----------------------------------------------------------------------------
// 2. CHUYỂN TAB GIAO DIỆN
// ----------------------------------------------------------------------------
document.querySelectorAll(".nav-tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    const target = document.getElementById(btn.dataset.tab);
    if (target) target.classList.add("active");

    if (btn.dataset.tab === "tab-alerts") {
      loadAlertsAndStats();
    } else if (btn.dataset.tab === "tab-eval") {
      loadModelEvaluation();
    }
  });
});

// ----------------------------------------------------------------------------
// 3. TẢI DANH SÁCH VIDEO / ẢNH & UPLOAD FILE
// ----------------------------------------------------------------------------
async function loadSamples(selectPath = null) {
  const res = await fetch("/api/samples");
  const data = await res.json();
  const sel = document.getElementById("sampleSelect");
  sel.innerHTML = "";

  (data.samples || []).forEach((item) => {
    const opt = document.createElement("option");
    opt.value = item.path;
    const icon = item.type === "video" ? "🎬" : "🖼️";
    opt.textContent = `${icon} ${item.name} (${item.size_kb} KB)`;
    sel.appendChild(opt);
  });

  if (selectPath) {
    sel.value = selectPath;
  } else if (sel.querySelector('option[value="tests/video3.mp4"]')) {
    sel.value = "tests/video3.mp4";
  }
}

const fileInput = document.getElementById("fileUploadInput");
const dropzone = document.getElementById("uploadDropzone");
const uploadStatus = document.getElementById("uploadStatus");

async function handleFileUpload(file) {
  if (!file) return;
  uploadStatus.textContent = `⏳ Đang tải lên "${file.name}"...`;

  const reader = new FileReader();
  reader.onload = async () => {
    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename: file.name,
          data: reader.result,
        }),
      });
      const out = await res.json();
      if (out.ok) {
        uploadStatus.textContent = `✅ Đã tải lên: ${file.name}`;
        await loadSamples(out.path);
      } else {
        uploadStatus.textContent = `❌ Lỗi tải file`;
      }
    } catch (e) {
      uploadStatus.textContent = `❌ Lỗi kết nối khi tải file`;
    }
  };
  reader.readAsDataURL(file);
}

fileInput.addEventListener("change", (e) => handleFileUpload(e.target.files[0]));
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.style.borderColor = "#22c55e";
});
dropzone.addEventListener("dragleave", () => {
  dropzone.style.borderColor = "#38bdf8";
});
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.style.borderColor = "#38bdf8";
  if (e.dataTransfer.files.length > 0) {
    handleFileUpload(e.dataTransfer.files[0]);
  }
});

// ----------------------------------------------------------------------------
// 4. ĐIỀU KHIỂN BẮT ĐẦU / DỪNG GIÁM SÁT & CẬP NHẬT TRẠNG THÁI TRỰC TIẾP
// ----------------------------------------------------------------------------
const confSlider = document.getElementById("confSlider");
const confValueLabel = document.getElementById("confValueLabel");
confSlider.addEventListener("input", () => {
  confValueLabel.textContent = Number(confSlider.value).toFixed(2);
});

const btnStart = document.getElementById("btnStart");
const btnWebcam = document.getElementById("btnWebcam");
const btnStop = document.getElementById("btnStop");
const streamView = document.getElementById("streamView");
const webcamVideo = document.getElementById("webcamVideo");
const webcamCanvas = document.getElementById("webcamCanvas");

let webcamStream = null;
let webcamIntervalId = null;
let webcamFrameCount = 0;
let isWebcamMode = false;
let isProcessingWebcam = false;

async function startWebcamMode() {
  if (isWebcamMode) return;
  mutedTemporarily = false;
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }

  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false
    });
    webcamVideo.srcObject = webcamStream;
    await webcamVideo.play();

    isWebcamMode = true;
    btnStart.disabled = true;
    if (btnWebcam) btnWebcam.disabled = true;
    btnStop.disabled = false;

    document.getElementById("currentSourceTitle").textContent = "📷 Đang truyền trực tiếp từ Webcam máy tính";
    const dot = document.getElementById("liveDot");
    dot.className = "status-dot live";

    webcamFrameCount = 0;
    const vW = webcamVideo.videoWidth || 640;
    const vH = webcamVideo.videoHeight || 480;
    webcamCanvas.width = vW;
    webcamCanvas.height = vH;
    const ctx = webcamCanvas.getContext("2d");

    // Loop gui frame len server de YOLO v3 xu ly
    webcamIntervalId = setInterval(async () => {
      if (!isWebcamMode || isProcessingWebcam || !webcamVideo.videoWidth) return;
      isProcessingWebcam = true;
      try {
        if (webcamCanvas.width !== webcamVideo.videoWidth) {
          webcamCanvas.width = webcamVideo.videoWidth;
          webcamCanvas.height = webcamVideo.videoHeight;
        }
        ctx.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);
        const b64 = webcamCanvas.toDataURL("image/jpeg", 0.82);
        const conf = parseFloat(confSlider.value);
        const selectedModels = getSelectedModels();
        const model_key = selectedModels.join(",");
        webcamFrameCount++;

        const res = await fetch("/api/process_frame", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image: b64,
            conf,
            models: selectedModels,
            model_key,
            frame_idx: webcamFrameCount,
            fps: 16.0
          })
        });
        const data = await res.json();
        if (data.ok) {
          if (data.image) {
            streamView.src = data.image;
          }
          // Cap nhat 4 the dem
          document.getElementById("countTotal").textContent = data.counts.total;
          document.getElementById("countDrowning").textContent = data.counts.drowning;
          document.getElementById("countSwimming").textContent = data.counts.swimming;
          document.getElementById("countOutOfWater").textContent = data.counts.out_of_water;
          document.getElementById("fpsInfo").textContent = `Webcam Live | Đang quét AI`;
          document.getElementById("maxDrownConf").textContent = `Độ tin cậy cao nhất: ${Math.round((data.max_drown_conf || 0) * 100)}%`;
          document.getElementById("frameProgressBadge").textContent = `Live Frame: ${webcamFrameCount}`;

          // Canh bao am thanh va hinh anh
          const drownCard = document.getElementById("drowningCard");
          const videoContainer = document.getElementById("videoContainer");
          const banner = document.getElementById("emergencyBanner");

          if (data.is_drowning) {
            drownCard.classList.add("alert-active");
            videoContainer.classList.add("alert-border");
            if (!mutedTemporarily) {
              banner.classList.remove("hidden");
              document.getElementById("emergencyDetail").textContent =
                `Phát hiện ${data.counts.drowning} người đuối nước trước Camera | Độ tin cậy: ${(data.max_drown_conf * 100).toFixed(1)}%`;
            }
            playEmergencySiren(false);
          } else {
            drownCard.classList.remove("alert-active");
            videoContainer.classList.remove("alert-border");
            banner.classList.add("hidden");
          }

          // Cập nhật bảng danh sách đối tượng trực tiếp từ Webcam
          const tbody = document.getElementById("liveDetectionsBody");
          const dets = data.detections || [];
          if (dets.length === 0) {
            tbody.innerHTML = `<tr><td colspan="4" class="empty-cell">Đang quan sát, chưa phát hiện người nào...</td></tr>`;
          } else {
            tbody.innerHTML = dets
              .map((d) => {
                const badgeClass =
                  d.class_id === 0 ? "state-Drowning" : d.class_id === 1 ? "state-Out" : "state-Swimming";
                return `
                  <tr>
                    <td><strong>#${d.id}</strong></td>
                    <td><span class="state-badge ${badgeClass}">${d.class_vi}</span></td>
                    <td><strong>${(d.confidence * 100).toFixed(1)}%</strong> (${d.confidence.toFixed(2)})</td>
                    <td><code>[${d.bbox.join(", ")}]</code></td>
                  </tr>
                `;
              })
              .join("");
          }
        }
      } catch (err) {
        console.error("Loi xu ly frame webcam:", err);
      } finally {
        isProcessingWebcam = false;
      }
    }, 70); // ~14 FPS cuc ky muot ma

  } catch (err) {
    alert("Không thể mở Webcam: " + err.message + "\nHãy đảm bảo bạn đã cấp quyền truy cập Camera cho trình duyệt!");
    console.error(err);
    stopWebcamMode();
  }
}

function stopWebcamMode() {
  if (webcamIntervalId) {
    clearInterval(webcamIntervalId);
    webcamIntervalId = null;
  }
  if (webcamStream) {
    webcamStream.getTracks().forEach(track => track.stop());
    webcamStream = null;
  }
  isWebcamMode = false;
  btnStart.disabled = false;
  if (btnWebcam) btnWebcam.disabled = false;
  btnStop.disabled = true;
  document.getElementById("currentSourceTitle").textContent = "Chưa chạy giám sát";
  const dot = document.getElementById("liveDot");
  dot.className = "status-dot idle";
  document.getElementById("emergencyBanner").classList.add("hidden");
}

// Slider cấu hình thời gian chìm liên tiếp
const drownSecSlider = document.getElementById("drownSecSlider");
const drownSecLabel = document.getElementById("drownSecLabel");
if (drownSecSlider && drownSecLabel) {
  drownSecSlider.addEventListener("input", (e) => {
    drownSecLabel.textContent = `${e.target.value} giây`;
  });
}

// --- ĐIỀU KHIỂN ĐA MÔ HÌNH (MULTI-MODEL ENSEMBLE) ---
const monitorModelSelect = document.getElementById("monitorModelSelect");
const modelCheckboxes = document.querySelectorAll(".chk-model");

const ENSEMBLE_PRESETS = {
  ensemble_committee: ["yolov26_v3_best", "yolov26_v3_recall", "yolov26_v1"],
  ensemble_v3_rec_v1: ["yolov26_v3_recall", "yolov26_v1"],
  ensemble_v3_v1: ["yolov26_v3_best", "yolov26_v1"],
  yolov26_v3_best: ["yolov26_v3_best"],
  yolov26_v3_recall: ["yolov26_v3_recall"],
  yolov26_v1: ["yolov26_v1"],
  yolov26_v2: ["yolov26_v2"],
  yolov8n: ["yolov8n"],
};

function getSelectedModels() {
  const checked = Array.from(document.querySelectorAll(".chk-model:checked")).map((cb) => cb.value);
  return checked.length > 0 ? checked : ["yolov26_v3_best", "yolov26_v3_recall", "yolov26_v1"];
}

if (monitorModelSelect) {
  monitorModelSelect.addEventListener("change", (e) => {
    const val = e.target.value;
    if (val === "custom") return;
    const targetKeys = ENSEMBLE_PRESETS[val] || [val];
    modelCheckboxes.forEach((cb) => {
      cb.checked = targetKeys.includes(cb.value);
    });
  });
}

modelCheckboxes.forEach((cb) => {
  cb.addEventListener("change", () => {
    const current = getSelectedModels();
    let matchedPreset = "custom";
    for (const [presetKey, keys] of Object.entries(ENSEMBLE_PRESETS)) {
      if (keys.length === current.length && keys.every((k) => current.includes(k))) {
        matchedPreset = presetKey;
        break;
      }
    }
    if (monitorModelSelect) {
      monitorModelSelect.value = matchedPreset;
    }
  });
});

if (btnWebcam) {
  btnWebcam.addEventListener("click", () => {
    startWebcamMode();
  });
}

btnStart.addEventListener("click", async () => {
  mutedTemporarily = false;
  const path = document.getElementById("sampleSelect").value;
  if (path === "webcam") {
    return startWebcamMode();
  }

  const conf = parseFloat(confSlider.value);
  const drown_slider = document.getElementById("drownSecSlider");
  const drown_threshold = drown_slider ? parseFloat(drown_slider.value) : 30.0;
  const smart_filter = document.getElementById("chkSmartFilter").checked;
  const selectedModels = getSelectedModels();
  const model_key = selectedModels.join(",");

  // Khởi tạo AudioContext ngay khi người dùng nhấn nút
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }

  btnStart.disabled = true;
  if (btnWebcam) btnWebcam.disabled = true;
  btnStop.disabled = false;

  const isImage = /\.(jpg|jpeg|png|webp|bmp)$/i.test(path);
  if (!isImage) {
    streamView.src = `/api/stream?t=${Date.now()}`;
  }

  await fetch("/api/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, conf, smart_filter, models: selectedModels, model_key, drown_threshold }),
  });

  if (isImage) {
    setTimeout(() => {
      streamView.src = `/api/frame.jpg?t=${Date.now()}`;
    }, 900);
  }
});

btnStop.addEventListener("click", async () => {
  if (isWebcamMode) {
    stopWebcamMode();
  }
  await fetch("/api/stop", { method: "POST" });
  streamView.src = `/api/frame.jpg?t=${Date.now()}`;
  btnStart.disabled = false;
  if (btnWebcam) btnWebcam.disabled = false;
  btnStop.disabled = true;
  document.getElementById("emergencyBanner").classList.add("hidden");
});

document.getElementById("btnMuteBanner").addEventListener("click", () => {
  mutedTemporarily = true;
  document.getElementById("emergencyBanner").classList.add("hidden");
});

document.getElementById("btnTestAlarm").addEventListener("click", async () => {
  mutedTemporarily = false;
  playEmergencySiren(true);
  const banner = document.getElementById("emergencyBanner");
  banner.classList.remove("hidden");
  document.getElementById("emergencyDetail").textContent =
    "Cảnh báo thử nghiệm (Độ tin cậy: 86.0%) | Đã ghi nhận vào Cơ sở dữ liệu SQLite!";

  await fetch("/api/alerts/simulate", { method: "POST" });
  loadAlertsAndStats();
});

async function pollLiveStatus() {
  if (isWebcamMode) return; // Đang chạy Webcam thì không để polling ghi đè giao diện
  try {
    const res = await fetch("/api/status");
    const st = await res.json();

    const isRunning = Boolean(st.running);
    btnStart.disabled = isRunning;
    btnStop.disabled = !isRunning;

    const dot = document.getElementById("liveDot");
    dot.className = `status-dot ${isRunning ? "live" : "idle"}`;
    document.getElementById("currentSourceTitle").textContent = isRunning
      ? `Đang giám sát: ${st.source_name}`
      : `Nguồn hiện tại: ${st.source_name} (Đã dừng / Hoàn tất)`;

    document.getElementById("fpsInfo").textContent = isRunning
      ? `Tốc độ xử lý: ${st.fps} FPS`
      : `Đã quét xong (${st.fps} FPS)`;

    document.getElementById("frameProgressBadge").textContent =
      `Frame: ${st.frame_idx || 0} / ${st.total_frames || 0}`;

    // Nếu vừa quét xong ảnh tĩnh thì làm mới khung ảnh
    if (st.source_type === "image" && st.frame_idx === 1 && streamView.src.includes("/api/frame.jpg")) {
      const urlObj = new URL(streamView.src, window.location.origin);
      if (!urlObj.searchParams.get("u") || Number(urlObj.searchParams.get("u")) < st.updated_at) {
        streamView.src = `/api/frame.jpg?u=${st.updated_at}`;
      }
    }

    // Cập nhật 4 thẻ đếm
    const counts = st.counts || { total: 0, drowning: 0, swimming: 0, out_of_water: 0 };
    document.getElementById("countTotal").textContent = counts.total;
    document.getElementById("countDrowning").textContent = counts.drowning;
    document.getElementById("countSwimming").textContent = counts.swimming;
    document.getElementById("countOutOfWater").textContent = counts.out_of_water;
    document.getElementById("maxDrownConf").textContent =
      `Độ tin cậy cao nhất: ${(st.max_drowning_conf * 100).toFixed(1)}%`;

    // Kích hoạt cảnh báo hình ảnh & âm thanh theo quy chuẩn 30 giây liên tiếp
    const drownCard = document.getElementById("drowningCard");
    const videoContainer = document.getElementById("videoContainer");
    const banner = document.getElementById("emergencyBanner");
    const maxDrownLabel = document.getElementById("maxDrownConf");

    const duration = st.drowning_duration || 0.0;
    const threshold = st.drowning_threshold || 30.0;

    if (st.is_drowning_alert && counts.drowning > 0) {
      // Đã chìm liên tiếp >= 30 giây -> Kích hoạt còi báo động khẩn cấp & viền Đỏ
      drownCard.classList.remove("alert-warning");
      drownCard.classList.add("alert-active");
      videoContainer.classList.add("alert-border");
      if (maxDrownLabel) {
        maxDrownLabel.textContent = `🚨 ĐÃ CHÌM ${duration.toFixed(1)}s (VƯỢT ${threshold}s)!`;
      }
      if (!mutedTemporarily) {
        banner.classList.remove("hidden");
        document.getElementById("emergencyDetail").textContent =
          `Phát hiện ${counts.drowning} người đuối nước liên tiếp ${duration.toFixed(1)}s tại ${st.source_name} | Độ tin cậy: ${(st.max_drowning_conf * 100).toFixed(1)}% | Đã lưu CSDL`;
      }
      playEmergencySiren(false);
    } else if (st.is_tracking_drowning && counts.drowning > 0) {
      // Đang theo dõi nghi vấn đuối nước (chưa đủ 30 giây) -> Thẻ chuyển màu cam, chưa hú còi
      drownCard.classList.add("alert-warning");
      drownCard.classList.remove("alert-active");
      videoContainer.classList.remove("alert-border");
      banner.classList.add("hidden");
      if (maxDrownLabel) {
        maxDrownLabel.textContent = `⏱️ Đang chìm: ${duration.toFixed(1)}s / ${threshold}s (Đếm liên tiếp)`;
      }
    } else {
      // An toàn bình thường
      drownCard.classList.remove("alert-active");
      drownCard.classList.remove("alert-warning");
      videoContainer.classList.remove("alert-border");
      banner.classList.add("hidden");
      if (maxDrownLabel) {
        maxDrownLabel.textContent = `Thời gian chìm: 0.0s / ${threshold}s`;
      }
    }

    // Cập nhật bảng danh sách đối tượng trong khung hình
    const tbody = document.getElementById("liveDetectionsBody");
    const dets = st.detections || [];
    if (dets.length === 0) {
      tbody.innerHTML = `<tr><td colspan="4" class="empty-cell">Không có đối tượng nào trong khung hình hiện tại.</td></tr>`;
    } else {
      tbody.innerHTML = dets
        .map((d) => {
          const badgeClass =
            d.class_id === 0 ? "state-Drowning" : d.class_id === 1 ? "state-Out" : "state-Swimming";
          return `
            <tr>
              <td><strong>#${d.id}</strong></td>
              <td><span class="state-badge ${badgeClass}">${d.class_vi}</span></td>
              <td><strong>${(d.confidence * 100).toFixed(1)}%</strong> (${d.confidence.toFixed(2)})</td>
              <td><code>[${d.bbox.join(", ")}]</code></td>
            </tr>
          `;
        })
        .join("");
    }
  } catch (e) {
    // Bỏ qua lỗi mạng tạm thời
  }
}

setInterval(pollLiveStatus, 650);

// ----------------------------------------------------------------------------
// 5. LỊCH SỬ & THỐNG KÊ CẢNH BÁO (SQLITE)
// ----------------------------------------------------------------------------
async function loadAlertsAndStats() {
  const res = await fetch("/api/alerts");
  const data = await res.json();
  const sum = data.summary || {};
  const alerts = data.alerts || [];

  document.getElementById("navAlertBadge").textContent = sum.total_alerts || 0;
  document.getElementById("statTotalAlerts").textContent = sum.total_alerts || 0;
  document.getElementById("statPendingAlerts").textContent = sum.pending_alerts || 0;
  document.getElementById("statAvgConf").textContent = `${((sum.avg_confidence || 0) * 100).toFixed(1)}%`;
  document.getElementById("statPeakConf").textContent = `${((sum.peak_confidence || 0) * 100).toFixed(1)}%`;

  const tbody = document.getElementById("alertsHistoryBody");
  if (alerts.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10" class="empty-cell">Chưa có lịch sử cảnh báo nào trong cơ sở dữ liệu.</td></tr>`;
  } else {
    tbody.innerHTML = alerts
      .map((a) => {
        const isPending = a.status === "Chưa xử lý";
        const thumbHtml = a.snapshot_url
          ? `<img src="${a.snapshot_url}" class="thumb-img" onclick="openSnapshotModal('${a.snapshot_url}', 'Cảnh báo #${a.id} - ${a.source_name}')" alt="Ảnh" />`
          : `<span style="color:#64748b">Không có</span>`;
        return `
          <tr>
            <td><strong>#${a.id}</strong></td>
            <td>${a.created_at}</td>
            <td><strong>${a.source_name}</strong></td>
            <td>${a.frame_idx}</td>
            <td><span class="state-badge state-Drowning">🚨 ${a.drowning_count} người</span></td>
            <td>${a.total_people} người</td>
            <td><strong>${(a.max_confidence * 100).toFixed(1)}%</strong></td>
            <td>${thumbHtml}</td>
            <td>
              <span class="state-badge ${isPending ? "state-Out" : "state-Swimming"}">
                ${a.status}
              </span>
            </td>
            <td>
              ${
                isPending
                  ? `<button class="btn btn-sm btn-primary" onclick="resolveAlert(${a.id})">✔ Xác nhận</button>`
                  : ""
              }
              <button class="btn btn-sm btn-danger" onclick="deleteAlert(${a.id})">✖</button>
            </td>
          </tr>
        `;
      })
      .join("");
  }

  drawAlertSourceChart(data.by_source || []);
  drawAlertTimelineChart(alerts.slice(0, 15).reverse());
}

window.openSnapshotModal = function (url, title) {
  document.getElementById("modalTitle").textContent = title;
  document.getElementById("modalImage").src = url;
  document.getElementById("snapshotModal").classList.remove("hidden");
};

document.getElementById("btnCloseModal").addEventListener("click", () => {
  document.getElementById("snapshotModal").classList.add("hidden");
});
document.getElementById("modalBackdrop").addEventListener("click", () => {
  document.getElementById("snapshotModal").classList.add("hidden");
});

window.resolveAlert = async function (id) {
  await fetch("/api/alerts/resolve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id }),
  });
  loadAlertsAndStats();
};

window.deleteAlert = async function (id) {
  await fetch("/api/alerts/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id }),
  });
  loadAlertsAndStats();
};

document.getElementById("btnRefreshAlerts").addEventListener("click", loadAlertsAndStats);
document.getElementById("btnClearAlerts").addEventListener("click", async () => {
  await fetch("/api/alerts/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: "all" }),
  });
  loadAlertsAndStats();
});

// ----------------------------------------------------------------------------
// 6. ĐÁNH GIÁ MÔ HÌNH (PRECISION, RECALL, mAP50, mAP50-95)
// ----------------------------------------------------------------------------
async function loadModelEvaluation() {
  if (!evalDataCache) {
    const res = await fetch("/api/evaluation");
    const data = await res.json();
    evalDataCache = data.models || {};
  }
  renderSelectedModelEval();
  renderModelComparisonTable();
}

document.getElementById("evalModelSelect").addEventListener("change", renderSelectedModelEval);

function renderSelectedModelEval() {
  if (!evalDataCache) return;
  const key = document.getElementById("evalModelSelect").value;
  const info = evalDataCache[key];
  if (!info) return;

  document.getElementById("evalModelMeta").textContent =
    `File: ${info.name} (${info.size_mb} MB) | Kích thước ảnh train: imgsz=${info.imgsz} | Số Epochs đã train: ${info.epochs_trained} | Optimizer: ${info.optimizer} (lr0=${info.lr0})`;

  const m = info.metrics;
  const pPct = (m.precision * 100).toFixed(2);
  const rPct = (m.recall * 100).toFixed(2);
  const m50Pct = (m.mAP50 * 100).toFixed(2);
  const m95Pct = (m.mAP50_95 * 100).toFixed(2);

  document.getElementById("metricPrecision").textContent = `${pPct}%`;
  document.getElementById("metricRecall").textContent = `${rPct}%`;
  document.getElementById("metricMap50").textContent = `${m50Pct}%`;
  document.getElementById("metricMap5095").textContent = `${m95Pct}%`;

  document.getElementById("barPrecision").style.width = `${pPct}%`;
  document.getElementById("barRecall").style.width = `${rPct}%`;
  document.getElementById("barMap50").style.width = `${m50Pct}%`;
  document.getElementById("barMap5095").style.width = `${m95Pct}%`;

  // Vẽ 2 biểu đồ huấn luyện
  drawMultiLineChart(
    "evalMetricsChart",
    info.history.map((h) => h.epoch),
    [
      { label: "mAP@50", color: "#f59e0b", values: info.history.map((h) => h.mAP50 * 100) },
      { label: "Precision", color: "#0ea5e9", values: info.history.map((h) => h.precision * 100) },
      { label: "Recall", color: "#22c55e", values: info.history.map((h) => h.recall * 100) },
      { label: "mAP@50-95", color: "#a855f7", values: info.history.map((h) => h.mAP50_95 * 100) },
    ],
    "%"
  );

  drawMultiLineChart(
    "evalLossChart",
    info.history.map((h) => h.epoch),
    [
      { label: "Train Box Loss", color: "#38bdf8", values: info.history.map((h) => h.train_box_loss) },
      { label: "Val Box Loss", color: "#ef4444", values: info.history.map((h) => h.val_box_loss) },
      { label: "Train Cls Loss", color: "#4ade80", values: info.history.map((h) => h.train_cls_loss) },
      { label: "Val Cls Loss", color: "#f59e0b", values: info.history.map((h) => h.val_cls_loss) },
    ],
    ""
  );

  // Bảng lịch sử từng Epoch
  const tbody = document.getElementById("epochHistoryBody");
  tbody.innerHTML = info.history
    .slice()
    .reverse()
    .map(
      (h) => `
      <tr>
        <td><strong>Epoch ${h.epoch}</strong></td>
        <td>${(h.precision * 100).toFixed(2)}%</td>
        <td>${(h.recall * 100).toFixed(2)}%</td>
        <td><strong>${(h.mAP50 * 100).toFixed(2)}%</strong></td>
        <td><strong>${(h.mAP50_95 * 100).toFixed(2)}%</strong></td>
        <td>${h.train_box_loss.toFixed(4)}</td>
        <td>${h.train_cls_loss.toFixed(4)}</td>
        <td>${h.val_box_loss.toFixed(4)}</td>
        <td>${h.val_cls_loss.toFixed(4)}</td>
      </tr>
    `
    )
    .join("");
}

function renderModelComparisonTable() {
  if (!evalDataCache) return;
  const tbody = document.getElementById("modelComparisonBody");
  const rows = Object.values(evalDataCache);
  tbody.innerHTML = rows
    .map((m) => {
      const mt = m.metrics;
      return `
        <tr>
          <td><strong>${m.name}</strong></td>
          <td>${m.epochs_trained} / ${m.epochs_target}</td>
          <td><code>${m.imgsz}x${m.imgsz}</code></td>
          <td><strong>${(mt.precision * 100).toFixed(2)}%</strong></td>
          <td><strong>${(mt.recall * 100).toFixed(2)}%</strong></td>
          <td>${(mt.f1_score * 100).toFixed(2)}%</td>
          <td><span class="state-badge state-Swimming">${(mt.mAP50 * 100).toFixed(2)}%</span></td>
          <td><span class="state-badge state-Out">${(mt.mAP50_95 * 100).toFixed(2)}%</span></td>
          <td>${mt.val_box_loss.toFixed(4)}</td>
          <td>${mt.val_cls_loss.toFixed(4)}</td>
        </tr>
      `;
    })
    .join("");
}

// ----------------------------------------------------------------------------
// 7. HÀM VẼ BIỂU ĐỒ CANVAS (KHÔNG PHỤ THUỘC THƯ VIỆN NGOÀI)
// ----------------------------------------------------------------------------
function drawMultiLineChart(canvasId, xLabels, seriesList, unitSuffix = "") {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const padL = 48, padR = 20, padT = 32, padB = 32;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const allVals = seriesList.flatMap((s) => s.values);
  if (allVals.length === 0) return;
  const minV = Math.min(...allVals) * 0.92;
  const maxV = Math.max(...allVals) * 1.04;

  // Lưới ngang (Light Theme)
  ctx.strokeStyle = "#e2e8f0";
  ctx.fillStyle = "#64748b";
  ctx.font = "11px -apple-system, Segoe UI, sans-serif";
  for (let i = 0; i <= 4; i++) {
    const y = padT + (plotH * i) / 4;
    const val = maxV - ((maxV - minV) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(W - padR, y);
    ctx.stroke();
    ctx.fillText(`${val.toFixed(1)}${unitSuffix}`, 4, y + 4);
  }

  // Vẽ chú thích (Legend)
  let legX = padL;
  seriesList.forEach((s) => {
    ctx.fillStyle = s.color;
    ctx.fillRect(legX, 8, 12, 12);
    ctx.fillStyle = "#1e293b";
    ctx.fillText(s.label, legX + 16, 18);
    legX += ctx.measureText(s.label).width + 34;
  });

  // Vẽ đường
  const n = xLabels.length;
  seriesList.forEach((s) => {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    s.values.forEach((v, idx) => {
      const x = padL + (plotW * idx) / Math.max(n - 1, 1);
      const y = padT + plotH - ((v - minV) / Math.max(maxV - minV, 1e-6)) * plotH;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  // Nhãn trục X
  ctx.fillStyle = "#64748b";
  const step = Math.max(1, Math.floor(n / 6));
  for (let idx = 0; idx < n; idx += step) {
    const x = padL + (plotW * idx) / Math.max(n - 1, 1);
    ctx.fillText(`Ep ${xLabels[idx]}`, x - 12, H - 8);
  }
}

function drawAlertSourceChart(items) {
  const canvas = document.getElementById("alertSourceChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  if (items.length === 0) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "13px -apple-system, Segoe UI, sans-serif";
    ctx.fillText("Chưa có dữ liệu cảnh báo", W / 2 - 75, H / 2);
    return;
  }

  const maxC = Math.max(...items.map((i) => i.count), 1);
  const padL = 40, padB = 34, padT = 24, padR = 20;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const barW = Math.min(68, plotW / items.length - 16);

  items.forEach((it, idx) => {
    const x = padL + idx * (plotW / items.length) + 12;
    const bh = (it.count / maxC) * plotH;
    const y = padT + plotH - bh;

    ctx.fillStyle = "#ef4444";
    ctx.fillRect(x, y, barW, bh);

    ctx.fillStyle = "#0f172a";
    ctx.font = "bold 12px -apple-system, Segoe UI, sans-serif";
    ctx.fillText(String(it.count), x + barW / 2 - 4, y - 6);

    ctx.fillStyle = "#475569";
    ctx.font = "11px -apple-system, Segoe UI, sans-serif";
    ctx.fillText(it.source_name.slice(0, 12), x, H - 10);
  });
}

function drawAlertTimelineChart(alerts) {
  const canvas = document.getElementById("alertTimelineChart");
  if (!canvas) return;
  if (alerts.length === 0) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#64748b";
    ctx.font = "13px Segoe UI";
    ctx.fillText("Chưa có dữ liệu lịch sử", canvas.width / 2 - 70, canvas.height / 2);
    return;
  }
  drawMultiLineChart(
    "alertTimelineChart",
    alerts.map((a) => `#${a.id}`),
    [{ label: "Độ tin cậy Cảnh báo (%)", color: "#ef4444", values: alerts.map((a) => a.max_confidence * 100) }],
    "%"
  );
}

// Khởi chạy ban đầu
window.addEventListener("DOMContentLoaded", () => {
  loadSamples();
  loadAlertsAndStats();
  loadModelEvaluation();
});
