import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

// Accepts optional `data` prop; defaults to placeholder
export default function DrawdownChart({ data }) {
  const chartData =
    data ||
    Array.from({ length: 30 }, (_, i) => ({
      day: i + 1,
      drawdown: -(Math.random() * 8),
    }));

  return (
    <ResponsiveContainer width="100%" height={180}>
      <AreaChart data={chartData}>
        <defs>
          <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#f85149" stopOpacity={0.4} />
            <stop offset="95%" stopColor="#f85149" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
        <XAxis dataKey="day" tick={{ fill: "#8b949e", fontSize: 11 }} />
        <YAxis tick={{ fill: "#8b949e", fontSize: 11 }} tickFormatter={(v) => `${v.toFixed(0)}%`} />
        <Tooltip
          contentStyle={{ background: "#161b22", border: "1px solid #30363d", color: "#e6edf3" }}
          formatter={(v) => [`${v.toFixed(2)}%`, "Drawdown"]}
        />
        <Area
          type="monotone"
          dataKey="drawdown"
          stroke="#f85149"
          fill="url(#ddGrad)"
          strokeWidth={2}
          dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
