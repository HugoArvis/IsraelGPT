import { useWebSocket } from "../hooks/useWebSocket";
import ScoreGauge from "../components/ScoreGauge";
import OHLCChart from "../components/OHLCChart";
import PortfolioChart from "../components/PortfolioChart";

function StatCard({ label, value, sub, color = "text-white" }) {
  return (
    <div className="bg-brand-card border border-brand-border rounded-lg p-4">
      <p className="text-xs text-gray-400 mb-1">{label}</p>
      <p className={`text-2xl font-bold ${color}`}>{value}</p>
      {sub && <p className="text-xs text-gray-500 mt-0.5">{sub}</p>}
    </div>
  );
}

export default function Dashboard() {
  const { lastMessage: d, connected } = useWebSocket();

  const score = d?.score ?? 5.0;
  const pnlColor =
    (d?.pnl_total_pct ?? 0) >= 0 ? "text-brand-green" : "text-brand-red";
  const todayColor =
    (d?.pnl_today_pct ?? 0) >= 0 ? "text-brand-green" : "text-brand-red";

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Live Dashboard</h1>
        <span
          className={`text-xs px-2 py-1 rounded-full border ${
            connected
              ? "border-brand-green text-brand-green"
              : "border-brand-red text-brand-red"
          }`}
        >
          {connected ? "● LIVE" : "○ DISCONNECTED"}
        </span>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Portfolio Value"
          value={`$${(d?.portfolio_value ?? 100000).toLocaleString("en-US", { maximumFractionDigits: 0 })}`}
        />
        <StatCard
          label="P&L Today"
          value={`${(d?.pnl_today_pct ?? 0) >= 0 ? "+" : ""}${(d?.pnl_today_pct ?? 0).toFixed(2)}%`}
          color={todayColor}
        />
        <StatCard
          label="Total P&L"
          value={`${(d?.pnl_total_pct ?? 0) >= 0 ? "+" : ""}${(d?.pnl_total_pct ?? 0).toFixed(2)}%`}
          color={pnlColor}
        />
        <StatCard
          label="Sharpe (rolling)"
          value={(d?.sharpe_rolling ?? 0).toFixed(2)}
          sub={`Drawdown: ${(d?.drawdown_current ?? 0).toFixed(1)}%`}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-brand-card border border-brand-border rounded-lg p-6 flex flex-col items-center">
          <p className="text-sm text-gray-400 mb-4">Conviction Score</p>
          <ScoreGauge score={score} />
          <div className="mt-4 grid grid-cols-3 gap-3 w-full text-center text-xs">
            <div>
              <p className="text-gray-400">5d Alpha</p>
              <p className={`font-bold ${(d?.predicted_return_5d ?? 0) >= 0 ? "text-brand-green" : "text-brand-red"}`}>
                {(d?.predicted_return_5d ?? 0) >= 0 ? "+" : ""}{(d?.predicted_return_5d ?? 0).toFixed(2)}%
              </p>
            </div>
            <div>
              <p className="text-gray-400">21d Alpha</p>
              <p className={`font-bold ${(d?.predicted_return_21d ?? 0) >= 0 ? "text-brand-green" : "text-brand-red"}`}>
                {(d?.predicted_return_21d ?? 0) >= 0 ? "+" : ""}{(d?.predicted_return_21d ?? 0).toFixed(2)}%
              </p>
            </div>
            <div>
              <p className="text-gray-400">Confidence</p>
              <p className="text-brand-yellow font-bold">{((d?.confidence ?? 0) * 100).toFixed(0)}%</p>
            </div>
          </div>
          <p className="text-xs text-gray-400 mt-2">
            Position: <span className="text-white font-semibold">{(d?.position_pct ?? 0).toFixed(0)}%</span>
            {" · "}Stress: <span className="text-white font-semibold">{((d?.stress ?? 0) * 100).toFixed(0)}%</span>
          </p>
          {d?.strategy && (
            <p className="text-xs text-gray-500 mt-2 text-center italic">{d.strategy}</p>
          )}
        </div>

        <div className="lg:col-span-2 bg-brand-card border border-brand-border rounded-lg p-4">
          <p className="text-sm text-gray-400 mb-3">
            {d?.ticker ?? "AAPL"} — OHLC Chart
          </p>
          <OHLCChart ticker={d?.ticker ?? "AAPL"} />
        </div>
      </div>

      <div className="bg-brand-card border border-brand-border rounded-lg p-4">
        <p className="text-sm text-gray-400 mb-3">Portfolio Value</p>
        <PortfolioChart />
      </div>
    </div>
  );
}
