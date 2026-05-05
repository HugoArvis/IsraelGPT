import { useEffect, useState } from "react";
import { api } from "../api/client";

export default function Trades() {
  const [trades, setTrades] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    api
      .getTrades()
      .then(setTrades)
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <p className="text-brand-red">{error}</p>;

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Trade Log</h1>
      <div className="bg-brand-card border border-brand-border rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-brand-border text-gray-400 text-xs uppercase">
              <th className="text-left px-4 py-3">Time</th>
              <th className="text-left px-4 py-3">Ticker</th>
              <th className="text-left px-4 py-3">Side</th>
              <th className="text-right px-4 py-3">Qty</th>
              <th className="text-right px-4 py-3">Avg Price</th>
              <th className="text-right px-4 py-3">Status</th>
            </tr>
          </thead>
          <tbody>
            {trades.length === 0 ? (
              <tr>
                <td colSpan={6} className="text-center py-8 text-gray-500">
                  No trades yet
                </td>
              </tr>
            ) : (
              trades.map((t) => (
                <tr
                  key={t.id}
                  className="border-b border-brand-border hover:bg-white/5 transition-colors"
                >
                  <td className="px-4 py-3 text-gray-400 text-xs">{t.filled_at ?? "—"}</td>
                  <td className="px-4 py-3 font-semibold">{t.ticker}</td>
                  <td
                    className={`px-4 py-3 font-semibold ${
                      t.side === "buy" ? "text-brand-green" : "text-brand-red"
                    }`}
                  >
                    {t.side?.toUpperCase()}
                  </td>
                  <td className="px-4 py-3 text-right">{t.qty}</td>
                  <td className="px-4 py-3 text-right">
                    ${Number(t.filled_avg_price).toFixed(2)}
                  </td>
                  <td className="px-4 py-3 text-right text-xs text-gray-400">
                    {t.status}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
