const BASE = "/api";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export const api = {
  getStatus: () => request("/status"),
  getPortfolio: () => request("/portfolio"),
  getTrades: () => request("/trades"),
  getMetrics: () => request("/metrics"),
  startModel: () => request("/model/start", { method: "POST" }),
  stopModel: () => request("/model/stop", { method: "POST" }),
  killModel: () =>
    request("/model/kill", {
      method: "POST",
      body: JSON.stringify({ confirm: true }),
    }),
  runNow: () => request("/model/run-now", { method: "POST" }),
};
