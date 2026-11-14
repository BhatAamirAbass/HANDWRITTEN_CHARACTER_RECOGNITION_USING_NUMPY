const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearButton = document.getElementById("clear");
const recognizeButton = document.getElementById("recognize");
const statusElement = document.getElementById("status");
let drawing = false;
let hasDrawing = false;

function clearCanvas() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  hasDrawing = false;
}
clearCanvas();

ctx.strokeStyle = "#fff";
ctx.lineWidth = 19;
ctx.lineCap = "round";
ctx.lineJoin = "round";

function point(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * canvas.width / rect.width,
    y: (event.clientY - rect.top) * canvas.height / rect.height
  };
}

canvas.addEventListener("pointerdown", event => {
  event.preventDefault();
  drawing = true;
  hasDrawing = true;
  canvas.setPointerCapture(event.pointerId);
  const p = point(event);
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
  ctx.lineTo(p.x + 0.1, p.y + 0.1);
  ctx.stroke();
});
canvas.addEventListener("pointermove", event => {
  if (!drawing) return;
  const p = point(event);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
});
function stopDrawing() { drawing = false; ctx.closePath(); }
canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);

clearButton.addEventListener("click", () => {
  clearCanvas();
  document.getElementById("character").textContent = "?";
  document.getElementById("confidence").textContent = "Draw something to begin.";
  document.getElementById("top3").textContent = "Results will appear here.";
  statusElement.textContent = "Canvas cleared.";
});

function showTop3(items) {
  const target = document.getElementById("top3");
  target.replaceChildren();
  items.forEach(item => {
    const row = document.createElement("div");
    row.className = "prediction";
    const char = document.createElement("strong");
    char.textContent = item.character;
    const track = document.createElement("div");
    track.className = "track";
    const fill = document.createElement("div");
    fill.className = "fill";
    fill.style.width = `${Math.max(0, Math.min(100, item.confidence))}%`;
    track.appendChild(fill);
    const percent = document.createElement("span");
    percent.className = "percent";
    percent.textContent = `${item.confidence.toFixed(2)}%`;
    row.append(char, track, percent);
    target.appendChild(row);
  });
}

recognizeButton.addEventListener("click", async () => {
  if (!hasDrawing) {
    statusElement.textContent = "Draw a character first.";
    return;
  }
  recognizeButton.disabled = true;
  statusElement.textContent = "Recognizing...";
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: canvas.toDataURL("image/png") })
    });
    // const result = await response.json();
    const responseText = await response.text();

let result = {};

try {
  result = responseText ? JSON.parse(responseText) : {};
} catch (error) {
  throw new Error(
    `Server returned an invalid response (HTTP ${response.status}).`
  );
}

if (!response.ok) {
  throw new Error(result.error || `HTTP error: ${response.status}`);
}
    if (!response.ok) throw new Error(result.error || "Prediction failed.");
    document.getElementById("character").textContent = result.character;
    document.getElementById("confidence").textContent =
      `Confidence: ${result.confidence.toFixed(2)}%`;
    showTop3(result.top_3);
    statusElement.textContent = "Prediction complete.";
  } catch (error) {
    statusElement.textContent = error.message;
  } finally {
    recognizeButton.disabled = false;
  }
});
