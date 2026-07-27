const DEFAULTS = { enabled: false, strength: 50, denoise: true };

const enabledEl = document.getElementById("enabled");
const strengthEl = document.getElementById("strength");
const strengthValueEl = document.getElementById("strengthValue");
const denoiseEl = document.getElementById("denoise");

if (!navigator.gpu) {
  document.getElementById("unsupported").style.display = "block";
}

chrome.storage.sync.get(DEFAULTS, (stored) => {
  enabledEl.checked = stored.enabled;
  strengthEl.value = stored.strength;
  strengthValueEl.textContent = stored.strength;
  denoiseEl.checked = stored.denoise;
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
