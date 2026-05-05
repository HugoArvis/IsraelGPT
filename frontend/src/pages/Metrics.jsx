import { useEffect, useState } from "react";
import { api } from "../api/client";
import DrawdownChart from "../components/DrawdownChart";
import PortfolioChart from "../components/PortfolioChart";

function MetricCard({ label, value, good, bad }) {
  const color =
    good !== undefined && value >= good
      ? "text-brand-green"
      : bad !== undefined && value <= bad
      ? "text-brand-red"
      : "text-white";
  return (
    <div className="bg-brand-card border border-brand-border rounded-lg p-5">
      <p className="text-xs text-gray-400">{label}</p>
      <p className={`text-3xl font-bold mt-1 ${color}`}>{value}</p>
    </div>
  );
}

export default function Metrics() {
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api
      .getMetrics()
      .then(setMetrics)
      .catch((e) => setError(e.message));
    const id = setInterval(() => api.getMetrics().then(setMetrics).catch(() => {}), 30_000);
    return () => clearInterval(id);
  }, []);

  if (error) return <p className="text-brand-red">{error}</p>;
  if (!metrics) return <p className="text-gray-400">Loading metrics...</p>;

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Performance Metrics</h1>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard label="Sharpe Ratio" value={metrics.sharpe?.toFixed(2) ?? "—"} good={1.0} />
        <MetricCard label="Sortino Ratio" value={metrics.sortino?.toFixed(2) ?? "—"} good={1.0} />
        <MetricCard
          label="Max Drawdown"
          value={`${(metrics.max_drawdown * 100)?.toFixed(1) ?? "—"}%`}
          bad={-20}
        />
        <MetricCard label="Win Rate" value={`${(metrics.win_rate * 100)?.toFixed(1) ?? "—"}%`} good={53} />
      </div>

      <div className="bg-brand-card border border-brand-border rounded-lg p-4">
        <p className="text-sm text-gray-400 mb-3">Portfolio Equity</p>
        <PortfolioChart />
      </div>

      <div className="bg-brand-card border border-brand-border rounded-lg p-4">
        <p className="text-sm text-gray-400 mb-3">Drawdown</p>
        <DrawdownChart />
      </div>
    </div>
  );
}
