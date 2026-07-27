const DEFAULTS = {
  enabled: false,
  strength: 50,
  denoise: true,
  deblock: 40,
  deband: 30,
  contrast: 25,
  grain: 0,
  grainSize: 1,
  upscale: "2k",
  upscaler: "easu",
  compare: false,
  badge: "corner",
};

const SLIDERS = ["strength", "deblock", "deband", "contrast", "grain", "grainSize"];
const CHECKBOXES = ["enabled", "denoise", "compare"];
const SELECTS = ["upscale", "upscaler", "badge"];

const el = (id) => document.getElementById(id);

if (!navigator.gpu) {
  el("unsupported").style.display = "block";
}

function syncEnabledState(enabled) {
  el("controls").toggleAttribute("disabled", !enabled);
}

chrome.storage.sync.get(DEFAULTS, (stored) => {
  for (const id of CHECKBOXES) el(id).checked = stored[id];
  for (const id of SELECTS) el(id).value = stored[id];
  for (const id of SLIDERS) {
    el(id).value = stored[id];
    el(`${id}Value`).textContent = stored[id];
  }
  syncEnabledState(stored.enabled);
});

for (const id of CHECKBOXES) {
  el(id).addEventListener("change", () => {
    const checked = el(id).checked;
    chrome.storage.sync.set({ [id]: checked });
    if (id === "enabled") syncEnabledState(checked);
  });
}

for (const id of SELECTS) {
  el(id).addEventListener("change", () => {
    chrome.storage.sync.set({ [id]: el(id).value });
  });
}

for (const id of SLIDERS) {
  el(id).addEventListener("input", () => {
    const value = Number(el(id).value);
    el(`${id}Value`).textContent = value;
    chrome.storage.sync.set({ [id]: value });
  });
}
