"use strict";

const form = document.querySelector("#artwork-form");
const sourceInput = document.querySelector("#source-file");
const silhouetteInput = document.querySelector("#silhouette-file");
const toleranceInput = document.querySelector("#neutral-tolerance");
const colorInputs = ["black", "orange", "white"].map((name) => ({
  name,
  input: document.querySelector(`#color-${name}`),
  value: document.querySelector(`#value-${name}`),
}));
const generateButton = document.querySelector("#generate-button");
const generateLabel = document.querySelector("#generate-label");
const downloadButton = document.querySelector("#download-button");
const maskSelect = document.querySelector("#mask-kind");
const status = document.querySelector("#status");
const statusText = document.querySelector("#status-text");
const errorMessage = document.querySelector("#error-message");
const results = document.querySelector(".results");

let revision = 0;
let activeRequest = null;
let currentResult = null;
let sourceObjectURL = null;

function setStatus(message, state = "idle") {
  statusText.textContent = message;
  status.dataset.state = state;
}

function setBusy(busy) {
  results.setAttribute("aria-busy", String(busy));
  generateButton.disabled = busy || !sourceInput.files.length;
  generateLabel.textContent = busy ? "Generating previews…" : "Generate previews";
}

function setPreview(name, data) {
  const image = document.querySelector(`#${name}-preview`);
  const placeholder = document.querySelector(`#${name}-placeholder`);
  if (data) {
    image.src = data;
    image.hidden = false;
    placeholder.hidden = true;
  } else {
    image.removeAttribute("src");
    image.hidden = true;
    placeholder.hidden = false;
  }
}

function invalidate(message) {
  revision += 1;
  if (activeRequest) activeRequest.abort();
  activeRequest = null;
  currentResult = null;
  downloadButton.disabled = true;
  maskSelect.disabled = true;
  errorMessage.hidden = true;
  errorMessage.textContent = "";
  setPreview("normalized", null);
  setPreview("panel", null);
  setPreview("mask", null);
  document.querySelector("#mask-caption").textContent = "White marks the selected region.";
  document.querySelector("#result-badge").textContent = "Not generated";
  document.querySelector("#warning-summary").textContent = "Boundary notes will appear here.";
  document.querySelector("#warning-list").replaceChildren();
  document.querySelector("#warning-list").hidden = true;
  document.querySelector("#warning-area").classList.remove("has-warnings");
  for (const key of ["source", "dimensions", "method", "components", "hash"]) {
    const field = document.querySelector(`#record-${key}`);
    field.textContent = "—";
    field.removeAttribute("title");
  }
  document.querySelector("#manifest-details").hidden = true;
  document.querySelector("#manifest-json").textContent = "";
  setBusy(false);
  setStatus(message);
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

sourceInput.addEventListener("change", () => {
  const file = sourceInput.files[0];
  invalidate(file ? "Artwork selected. Generate a new proof." : "Choose artwork to begin.");
  if (sourceObjectURL) URL.revokeObjectURL(sourceObjectURL);
  sourceObjectURL = file ? URL.createObjectURL(file) : null;
  setPreview("source", sourceObjectURL);
  document.querySelector("#source-note").textContent = file
    ? `${file.name} · ${formatBytes(file.size)}`
    : "Choose an image with a visible enclosing rim.";
  document.querySelector("#source-caption").textContent = file ? file.name : "Your original image";
});

silhouetteInput.addEventListener("change", () => {
  const file = silhouetteInput.files[0];
  invalidate("Boundary selection changed. Generate a new proof.");
  document.querySelector("#silhouette-note").textContent = file
    ? `${file.name} · White inside; black outside.`
    : "Same dimensions as the artwork. White inside; black outside.";
});

for (const color of colorInputs) {
  color.input.addEventListener("input", () => {
    color.value.textContent = color.input.value.toUpperCase();
    invalidate("Palette changed. Generate a new proof.");
  });
}
toleranceInput.addEventListener("input", () => {
  invalidate("Neutral tolerance changed. Generate a new proof.");
});

function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result !== "string" || !reader.result.includes(",")) {
        reject(new Error("The selected file could not be read."));
        return;
      }
      resolve({ name: file.name, data: reader.result.slice(reader.result.indexOf(",") + 1) });
    };
    reader.onerror = () => reject(new Error("The selected file could not be read."));
    reader.onabort = () => reject(new Error("Reading the selected file was cancelled."));
    reader.readAsDataURL(file);
  });
}

function colorChannels(hex) {
  return [1, 3, 5].map((offset) => Number.parseInt(hex.slice(offset, offset + 2), 16));
}

function isPNG(value) {
  return typeof value === "string" && value.startsWith("data:image/png;base64,");
}

function validateResult(value) {
  const maskNames = ["silhouette", "exterior", "black", "orange", "white", "support-black", "support-orange", "support-white"];
  if (!value || typeof value !== "object" || !value.manifest || !value.previews
      || !isPNG(value.source_preview) || !isPNG(value.previews.normalized)
      || !isPNG(value.previews.panel) || !maskNames.every((name) => isPNG(value.previews[name]))
      || typeof value.archive !== "string" || !value.archive.startsWith("data:application/zip;base64,")) {
    throw new Error("The server returned an incomplete preview. Please generate it again.");
  }
  return value;
}

function showMask() {
  if (!currentResult) return;
  const name = maskSelect.value;
  setPreview("mask", currentResult.previews[name]);
  const label = maskSelect.selectedOptions[0].textContent;
  document.querySelector("#mask-preview").alt = `${label} binary mask`;
  const count = currentResult.manifest.pixel_counts?.[name];
  document.querySelector("#mask-caption").textContent = typeof count === "number"
    ? `${label} · ${count.toLocaleString()} selected pixels · White marks the selected region.`
    : `${label} · White marks the selected region.`;
}
maskSelect.addEventListener("change", showMask);

function showResult(value) {
  currentResult = value;
  const manifest = value.manifest;
  setPreview("source", value.source_preview);
  setPreview("normalized", value.previews.normalized);
  setPreview("panel", value.previews.panel);
  maskSelect.disabled = false;
  downloadButton.disabled = false;
  showMask();

  const warnings = Array.isArray(manifest.warnings) ? manifest.warnings : [];
  const warningList = document.querySelector("#warning-list");
  warningList.replaceChildren();
  for (const warning of warnings) {
    const item = document.createElement("li");
    item.textContent = String(warning);
    warningList.append(item);
  }
  warningList.hidden = warnings.length === 0;
  document.querySelector("#warning-area").classList.toggle("has-warnings", warnings.length > 0);
  document.querySelector("#warning-summary").textContent = warnings.length
    ? `${warnings.length} boundary ${warnings.length === 1 ? "note" : "notes"} to review`
    : "No boundary warnings. Inspect the edge and fine detail before moving on.";
  document.querySelector("#result-badge").textContent = "Ready to inspect";
  document.querySelector("#record-source").textContent = manifest.source?.name || sourceInput.files[0]?.name || "Artwork";
  const dimensions = manifest.dimensions_px;
  document.querySelector("#record-dimensions").textContent = dimensions
    ? `${dimensions.width} × ${dimensions.height} px` : "—";
  document.querySelector("#record-method").textContent = manifest.silhouette_method === "explicit-mask"
    ? "Explicit silhouette mask" : "Exterior-connected white";
  document.querySelector("#record-components").textContent = String(manifest.component_count ?? "—");
  const hash = manifest.source?.sha256;
  const hashField = document.querySelector("#record-hash");
  hashField.textContent = hash ? `${hash.slice(0, 16)}…` : "—";
  if (hash) hashField.title = hash;
  document.querySelector("#manifest-json").textContent = JSON.stringify(manifest, null, 2);
  document.querySelector("#manifest-details").hidden = false;
  setStatus(warnings.length ? "Previews generated. Review the boundary notes below." : "Previews generated. Your working proof is ready.", "ready");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;
  const file = sourceInput.files[0];
  if (!file) return;
  invalidate("Generating a new proof.");
  const requestRevision = revision;
  const controller = new AbortController();
  activeRequest = controller;
  errorMessage.hidden = true;
  downloadButton.disabled = true;
  setBusy(true);
  setStatus("Separating the palette and finding the panel edge…", "busy");
  try {
    const [source, silhouette] = await Promise.all([
      readFile(file),
      silhouetteInput.files[0] ? readFile(silhouetteInput.files[0]) : Promise.resolve(null),
    ]);
    if (requestRevision !== revision) return;
    const payload = {
      source,
      config: {
        palette: colorInputs.map((color) => colorChannels(color.input.value)),
        neutral_tolerance: Number(toleranceInput.value),
      },
    };
    if (silhouette) payload.silhouette = silhouette;
    const response = await fetch("/api/separate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    let value;
    try {
      value = await response.json();
    } catch {
      throw new Error("The local server returned an unreadable response. Please try again.");
    }
    if (requestRevision !== revision) return;
    if (!response.ok) {
      throw new Error(typeof value?.error === "string" ? value.error : "The artwork could not be processed.");
    }
    showResult(validateResult(value));
  } catch (error) {
    if (requestRevision !== revision || error.name === "AbortError") return;
    currentResult = null;
    downloadButton.disabled = true;
    errorMessage.textContent = error instanceof Error ? error.message : "The artwork could not be processed.";
    errorMessage.hidden = false;
    setStatus("The preview could not be generated. Check the message below.", "error");
  } finally {
    if (requestRevision === revision) {
      activeRequest = null;
      setBusy(false);
    }
  }
});

downloadButton.addEventListener("click", () => {
  if (!currentResult) return;
  const link = document.createElement("a");
  link.href = currentResult.archive;
  link.download = "panelchemy-separation.zip";
  document.body.append(link);
  link.click();
  link.remove();
});

window.addEventListener("pagehide", () => {
  if (activeRequest) activeRequest.abort();
  if (sourceObjectURL) URL.revokeObjectURL(sourceObjectURL);
});
