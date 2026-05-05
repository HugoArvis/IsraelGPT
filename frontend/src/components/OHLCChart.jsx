import { useEffect, useRef } from "react";
import { createChart } from "lightweight-charts";

export default function OHLCChart({ ticker = "AAPL" }) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      layout: { background: { color: "#161b22" }, textColor: "#8b949e" },
      grid: { vertLines: { color: "#21262d" }, horzLines: { color: "#21262d" } },
      crosshair: { mode: 1 },
      rightPriceScale: { borderColor: "#30363d" },
      timeScale: { borderColor: "#30363d", timeVisible: true },
      width: containerRef.current.clientWidth,
      height: 280,
    });
    chartRef.current = chart;
    const series = chart.addCandlestickSeries({
      upColor: "#3fb950",
      downColor: "#f85149",
      borderUpColor: "#3fb950",
      borderDownColor: "#f85149",
      wickUpColor: "#3fb950",
      wickDownColor: "#f85149",
    });
    seriesRef.current = series;

    // Seed with placeholder data until live data arrives
    const now = Math.floor(Date.now() / 1000);
    const dummy = Array.from({ length: 30 }, (_, i) => {
      const base = 180 + Math.random() * 20;
      return {
        time: now - (30 - i) * 86400,
        open: base,
        high: base + Math.random() * 5,
        low: base - Math.random() * 5,
        close: base + (Math.random() - 0.5) * 4,
      };
    });
    series.setData(dummy);
    chart.timeScale().fitContent();

    const ro = new ResizeObserver(() => {
      chart.applyOptions({ width: containerRef.current.clientWidth });
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
    };
  }, [ticker]);

  return <div ref={containerRef} className="w-full" />;
}
