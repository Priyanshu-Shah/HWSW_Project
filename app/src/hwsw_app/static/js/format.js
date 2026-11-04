// Human-readable numbers. Pure functions.

export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 100 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

export function formatSeconds(seconds) {
  if (seconds === null || seconds === undefined) return "–";
  if (seconds >= 1) return `${seconds.toFixed(2)} s`;
  if (seconds >= 1e-3) return `${(seconds * 1e3).toPrecision(3)} ms`;
  if (seconds >= 1e-6) return `${(seconds * 1e6).toPrecision(3)} µs`;
  return `${(seconds * 1e9).toPrecision(3)} ns`;
}

export function formatCount(number) {
  return number.toLocaleString("en-US");
}
