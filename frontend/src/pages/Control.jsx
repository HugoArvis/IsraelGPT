import { useEffect, useState } from "react";
import { api } from "../api/client";
import KillSwitch from "../components/KillSwitch";

export default function Control() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState(null);

  const refreshStatus = () =>
    api.getStatus().then(setStatus).catch(() => {});

  useEffect(() => {
    refreshStatus();
    const id = setInterval(refreshStatus, 10_000);
    return () => clearInterval(id);
  }, []);

  async function handle(action) {
    setLoading(true);
    setMsg(null);
    try {
      const res = await action();
      setMsg({ type: "ok", text: JSON.stringify(res) });
      refreshStatus();
    } catch (e) {
      setMsg({ type: "err", text: e.message });
    } finally {
      setLoading(false);
    }
  }

  const isActive = status?.model_status === "active";

  return (
    <div className="space-y-6 max-w-lg">
      <h1 className="text-xl font-semibold">Model Control</h1>

      <div className="bg-brand-card border border-brand-border rounded-lg p-5 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-gray-400 text-sm">Status</span>
          <span
            className={`text-sm font-bold ${
              isActive ? "text-brand-green" : "text-gray-400"
            }`}
          >
            {status?.model_status ?? "—"}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-400 text-sm">Uptime</span>
          <span className="text-sm">{status?.uptime || "—"}</span>
        </div>
      </div>

      <div className="flex gap-3">
        <button
          disabled={loading || isActive}
          onClick={() => handle(api.startModel)}
          className="flex-1 py-2 rounded-lg bg-brand-green text-black font-semibold text-sm disabled:opacity-40 hover:opacity-90 transition-opacity"
        >
          Start
        </button>
        <button
          disabled={loading || !isActive}
          onClick={() => handle(api.stopModel)}
          className="flex-1 py-2 rounded-lg bg-brand-yellow text-black font-semibold text-sm disabled:opacity-40 hover:opacity-90 transition-opacity"
        >
          Stop
        </button>
      </div>

      <div className="bg-brand-card border border-brand-border rounded-lg p-5 space-y-3">
        <p className="text-xs text-gray-400">
          Run inference now — fetches latest prices, runs the model, and updates
          the dashboard immediately (normally runs automatically at 15:30 NY).
        </p>
        <button
          disabled={loading || !isActive}
          onClick={() => handle(api.runNow)}
          className="w-full py-2 rounded-lg bg-brand-accent text-black font-semibold text-sm disabled:opacity-40 hover:opacity-90 transition-opacity"
        >
          Run Inference Now
        </button>
      </div>

      {msg && (
        <p
          className={`text-sm px-3 py-2 rounded border ${
            msg.type === "ok"
              ? "border-brand-green text-brand-green"
              : "border-brand-red text-brand-red"
          }`}
        >
          {msg.text}
        </p>
      )}

      <div className="border-t border-brand-border pt-6">
        <p className="text-xs text-gray-400 mb-4">
          Emergency — liquidates all positions and stops the model immediately.
        </p>
        <KillSwitch onConfirm={() => handle(api.killModel)} disabled={loading} />
      </div>
    </div>
  );
}
