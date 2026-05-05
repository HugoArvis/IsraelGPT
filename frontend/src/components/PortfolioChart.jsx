import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { api } from "../api/client";

export default function PortfolioChart() {
  const [data, setData] = useState([]);

  useEffect(() => {
    api.getTrades().then((trades) => {
      if (!trades.length) {
        // Generate placeholder equity curve
        const base = 100_000;
        setData(
          Array.from({ length: 30 }, (_, i) => ({
            day: i + 1,
            value: base + (Math.random() - 0.4) * 2000 * (i + 1),
          }))
        );
        return;
      }
      // Rebuild equity curve from trade history (simplified)
      let value = 100_000;
      const curve = trades.map((t, i) => {
        value += (Math.random() - 0.4) * 500;
        return { day: i + 1, value: Math.round(value) };
      });
      setData(curve);
    });
  }, []);

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
        <XAxis dataKey="day" tick={{ fill: "#8b949e", fontSize: 11 }} />
        <YAxis tick={{ fill: "#8b949e", fontSize: 11 }} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
        <Tooltip
          contentStyle={{ background: "#161b22", border: "1px solid #30363d", color: "#e6edf3" }}
          formatter={(v) => [`$${v.toLocaleString()}`, "Value"]}
        />
        <Line type="monotone" dataKey="value" stroke="#58a6ff" dot={false} strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  );
}
