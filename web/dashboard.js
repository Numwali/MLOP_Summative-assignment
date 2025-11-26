// web/dashboard.js
// Frontend logic for the CIFAR-10 dashboard (no external libs)

const API_BASE = ""; // empty -> uses same origin. if you run backend elsewhere, set full URL (e.g. "http://localhost:8000")

// Elements
const predictBtn = document.getElementById("predict-btn");
const singleFile = document.getElementById("single-file");
const predictResult = document.getElementById("predict-result");

const retrainBtn = document.getElementById("retrain-btn");
const retrainFiles = document.getElementById("retrain-files");
const retrainResult = document.getElementById("retrain-result");

const uptimeEl = document.getElementById("uptime");
const metricsText = document.getElementById("metrics-text");
const samplePredictions = document.getElementById("sample-predictions");

// --- Prediction ---
predictBtn.addEventListener("click", async () => {
  predictResult.innerText = "";
  if (!singleFile.files || singleFile.files.length === 0) {
    alert("Please choose an image to predict.");
    return;
  }
  const fd = new FormData();
  fd.append("file", singleFile.files[0]);

  predictResult.innerText = "Predicting...";

  try {
    const res = await fetch(API_BASE + "/predict", { method: "POST", body: fd });
    const data = await res.json();
    if (data.error) {
      predictResult.innerText = "Error: " + data.error;
      return;
    }
    const { predicted_class, confidence, top_3 } = data;
    predictResult.innerHTML = `<strong>Prediction:</strong> ${predicted_class} <span class="muted">(${(confidence*100).toFixed(2)}% confidence)</span>`;
    // show top 3
    predictResult.innerHTML += "<div class='top3'><strong>Top 3:</strong><ul>" + top_3.map(t => `<li>${t.class} — ${(t.confidence*100).toFixed(2)}%</li>`).join("") + "</ul></div>";
  } catch (err) {
    predictResult.innerText = "Prediction failed: " + err;
  }
});

// --- Retraining ---
retrainBtn.addEventListener("click", async () => {
  retrainResult.innerText = "";
  if (!retrainFiles.files || retrainFiles.files.length === 0) {
    alert("Please choose files for retraining.");
    return;
  }
  const fd = new FormData();
  // Append multiple files under single 'files' key (FastAPI expects files: List[UploadFile])
  for (let i = 0; i < retrainFiles.files.length; i++) {
    fd.append("files", retrainFiles.files[i], retrainFiles.files[i].name);
  }

  retrainResult.innerText = "Uploading and retraining... (this may take a while)";

  try {
    const res = await fetch(API_BASE + "/retrain", { method: "POST", body: fd });
    const data = await res.json();
    if (data.error) {
      retrainResult.innerText = "Error: " + data.error;
      return;
    }
    retrainResult.innerHTML = `<strong>Retrain complete</strong><br/>Test accuracy: ${(data.test_accuracy*100).toFixed(2)}% — Test loss: ${data.test_loss.toFixed(4)}`;
    // refresh metrics and uptime
    fetchMetrics();
    fetchUptime();
  } catch (err) {
    retrainResult.innerText = "Retrain failed: " + err;
  }
});

// --- Metrics & Uptime ---
async function fetchMetrics() {
  metricsText.innerText = "Loading metrics...";
  try {
    const res = await fetch(API_BASE + "/metrics");
    const data = await res.json();
    if (data && data.test_accuracy !== undefined) {
      metricsText.innerHTML = `
        <strong>Accuracy:</strong> ${(data.test_accuracy*100).toFixed(2)}% <br/>
        <strong>Precision:</strong> ${data.precision ?? "N/A"} <br/>
        <strong>Recall:</strong> ${data.recall ?? "N/A"} <br/>
        <strong>F1:</strong> ${data.f1_score ?? "N/A"}
      `;
    } else {
      metricsText.innerText = "No metrics available yet. Train/run notebook to produce training logs.";
    }
  } catch (err) {
    metricsText.innerText = "Failed to load metrics.";
  }
}

async function fetchUptime() {
  try {
    const res = await fetch(API_BASE + "/uptime");
    const data = await res.json();
    uptimeEl.innerText = `Server Uptime: ${data.uptime}`;
  } catch (err) {
    uptimeEl.innerText = "Failed to get uptime";
  }
}

// initial load
fetchMetrics();
fetchUptime();
setInterval(fetchUptime, 60 * 1000); // refresh every minute
