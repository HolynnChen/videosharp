const DEFAULTS = {
  enabled: false,
  strength: 50,
  denoise: true,
  splitPreview: false,
};

const enabledEl = document.getElementById("enabled");
const strengthEl = document.getElementById("strength");
const strengthValueEl = document.getElementById("strengthValue");
const denoiseEl = document.getElementById("denoise");
const splitPreviewEl = document.getElementById("splitPreview");

if (!navigator.gpu) {
  document.getElementById("unsupported").style.display = "block";
}

chrome.storage.sync.get(DEFAULTS, (stored) => {
  enabledEl.checked = stored.enabled;
  strengthEl.value = stored.strength;
  strengthValueEl.textContent = stored.strength;
  denoiseEl.checked = stored.denoise;
  splitPreviewEl.checked = stored.splitPreview;
});

enabledEl.addEventListener("change", () => {
  chrome.storage.sync.set({ enabled: enabledEl.checked });
});

strengthEl.addEventListener("input", () => {
  const value = Number(strengthEl.value);
  strengthValueEl.textContent = value;
  chrome.storage.sync.set({ strength: value });
});

denoiseEl.addEventListener("change", () => {
  chrome.storage.sync.set({ denoise: denoiseEl.checked });
});

splitPreviewEl.addEventListener("change", () => {
  chrome.storage.sync.set({ splitPreview: splitPreviewEl.checked });
});
