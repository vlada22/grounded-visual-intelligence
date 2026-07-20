const DATA_URL = "./data/evidence.json";
const FRAME_PERIOD = 0.25;
const COLORS = {
  mask: [255, 91, 74],
  box: "#ff6b57",
  zoneA: "#4bd4e5",
  zoneB: "#ff6b57",
};

const video = document.querySelector("#sceneVideo");
const canvas = document.querySelector("#overlay");
const context = canvas.getContext("2d");
const scrubber = document.querySelector("#scrubber");
const playButton = document.querySelector("#playButton");
const frameBadge = document.querySelector("#frameBadge");
const timecode = document.querySelector("#timecode");
const modelBadge = document.querySelector("#modelBadge");
const trackIdentity = document.querySelector("#trackIdentity");
const scoreLabel = document.querySelector("#scoreLabel");

let evidence;
let activeModel = "sam3";
const layers = { mask: true, box: true, zones: true };
const decodedMasks = new Map();

function formatTime(seconds) {
  const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const minutes = Math.floor(safe / 60).toString().padStart(2, "0");
  const remainder = (safe % 60).toFixed(2).padStart(5, "0");
  return `${minutes}:${remainder}`;
}

function frameIndexAt(seconds) {
  if (!evidence) return 0;
  return Math.min(evidence.video.frameCount - 1, Math.max(0, Math.round(seconds * evidence.video.fps)));
}

function decodeRle(modelKey, frame) {
  const key = `${modelKey}:${frame.frameIndex}`;
  if (decodedMasks.has(key)) return decodedMasks.get(key);
  const { counts, height, width } = frame.mask;
  const decoded = new Uint8Array(width * height);
  let position = 0;
  let value = 0;
  for (const count of counts) {
    if (value === 1) decoded.fill(1, position, position + count);
    position += count;
    value = value === 0 ? 1 : 0;
  }
  decodedMasks.set(key, decoded);
  return decoded;
}

function drawZones() {
  if (!layers.zones) return;
  context.save();
  context.lineWidth = 1.5;
  context.setLineDash([6, 5]);
  for (const zone of evidence.zones) {
    const x = zone.xMin * canvas.width;
    const width = (zone.xMax - zone.xMin) * canvas.width;
    context.strokeStyle = zone.id === "A" ? COLORS.zoneA : COLORS.zoneB;
    context.fillStyle = zone.id === "A" ? "rgba(75,212,229,.06)" : "rgba(255,107,87,.06)";
    context.fillRect(x, 0, width, canvas.height);
    context.strokeRect(x + .75, .75, width - 1.5, canvas.height - 1.5);
    context.setLineDash([]);
    context.fillStyle = "rgba(10,14,15,.74)";
    context.fillRect(x + 9, 10, 88, 24);
    context.fillStyle = zone.id === "A" ? COLORS.zoneA : COLORS.zoneB;
    context.font = "700 10px ui-monospace, monospace";
    context.fillText(`ZONE ${zone.id}`, x + 17, 26);
    context.setLineDash([6, 5]);
  }
  context.restore();
}

function drawFrame() {
  if (!evidence) return;
  context.clearRect(0, 0, canvas.width, canvas.height);

  const model = evidence.models[activeModel];
  const index = frameIndexAt(video.currentTime);
  const frame = model.frames[index];
  if (!frame) return;

  if (layers.mask) {
    const mask = decodeRle(activeModel, frame);
    const image = context.createImageData(frame.mask.width, frame.mask.height);
    for (let pixel = 0; pixel < mask.length; pixel += 1) {
      if (mask[pixel] === 0) continue;
      const offset = pixel * 4;
      image.data[offset] = COLORS.mask[0];
      image.data[offset + 1] = COLORS.mask[1];
      image.data[offset + 2] = COLORS.mask[2];
      image.data[offset + 3] = 116;
    }
    context.putImageData(image, 0, 0);
  }

  drawZones();

  if (layers.box) {
    const { xMin, yMin, xMax, yMax } = frame.bbox;
    context.save();
    context.strokeStyle = COLORS.box;
    context.lineWidth = 2;
    context.strokeRect(xMin, yMin, xMax - xMin, yMax - yMin);
    const label = `${model.shortLabel} · white cup`;
    context.font = "700 9px ui-monospace, monospace";
    const labelWidth = context.measureText(label).width + 12;
    const labelY = Math.max(4, yMin - 20);
    context.fillStyle = "rgba(15,19,20,.82)";
    context.fillRect(xMin, labelY, labelWidth, 17);
    context.fillStyle = COLORS.box;
    context.fillText(label, xMin + 6, labelY + 12);
    context.restore();
  }

  frameBadge.textContent = `frame ${String(index).padStart(2, "0")} · ${frame.timestamp.toFixed(2)} s`;
}

function syncControls() {
  const current = Math.min(10, video.currentTime || 0);
  scrubber.value = String(current);
  timecode.value = `${formatTime(current)} / 00:10.00`;
  drawFrame();
}

function seek(seconds) {
  video.currentTime = Math.min(9.99, Math.max(0, seconds));
  syncControls();
}

function selectModel(modelKey) {
  activeModel = modelKey;
  document.querySelectorAll(".model-button").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.model === modelKey);
  });
  const model = evidence.models[modelKey];
  modelBadge.textContent = model.label;
  trackIdentity.textContent = `${model.trackId} · white cup`;
  scoreLabel.textContent = `${model.scoreLabel} ${model.score.toFixed(3)}`;
  drawFrame();
}

document.querySelectorAll(".model-button").forEach((button) => {
  button.addEventListener("click", () => selectModel(button.dataset.model));
});

document.querySelectorAll(".toggle").forEach((button) => {
  button.addEventListener("click", () => {
    const layer = button.dataset.layer;
    layers[layer] = !layers[layer];
    button.classList.toggle("is-active", layers[layer]);
    button.setAttribute("aria-pressed", String(layers[layer]));
    drawFrame();
  });
  button.setAttribute("aria-pressed", "true");
});

document.querySelectorAll("[data-seek]").forEach((button) => {
  button.addEventListener("click", () => seek(Number(button.dataset.seek)));
});

playButton.addEventListener("click", async () => {
  if (video.paused) await video.play();
  else video.pause();
});
video.addEventListener("play", () => { playButton.textContent = "❚❚"; playButton.setAttribute("aria-label", "Pause video"); });
video.addEventListener("pause", () => { playButton.textContent = "▶"; playButton.setAttribute("aria-label", "Play video"); });
video.addEventListener("timeupdate", syncControls);
video.addEventListener("seeked", syncControls);
video.addEventListener("loadedmetadata", () => { scrubber.max = "10"; syncControls(); });
scrubber.addEventListener("input", () => seek(Number(scrubber.value)));
document.querySelector("#previousFrame").addEventListener("click", () => seek(video.currentTime - FRAME_PERIOD));
document.querySelector("#nextFrame").addEventListener("click", () => seek(video.currentTime + FRAME_PERIOD));

document.addEventListener("keydown", (event) => {
  if (["INPUT", "BUTTON", "A"].includes(document.activeElement?.tagName)) return;
  if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
    event.preventDefault();
    seek(video.currentTime + (event.key === "ArrowLeft" ? -FRAME_PERIOD : FRAME_PERIOD));
  }
  if (event.key === " ") {
    event.preventDefault();
    playButton.click();
  }
});

async function start() {
  const response = await fetch(DATA_URL);
  if (!response.ok) throw new Error(`Unable to load evidence data: ${response.status}`);
  evidence = await response.json();
  selectModel(activeModel);
  syncControls();
}

start().catch((error) => {
  console.error(error);
  modelBadge.textContent = "Evidence unavailable";
});
