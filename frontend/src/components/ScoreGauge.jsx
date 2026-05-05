import { useEffect, useRef } from "react";

const SIZE = 180;
const CX = SIZE / 2;
const CY = SIZE / 2;
const R = 72;
const STROKE = 12;
const START_ANGLE = -210;
const END_ANGLE = 30;

function degToRad(deg) {
  return (deg * Math.PI) / 180;
}

function polarToXY(cx, cy, r, angleDeg) {
  const rad = degToRad(angleDeg);
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function describeArc(cx, cy, r, startDeg, endDeg) {
  const s = polarToXY(cx, cy, r, startDeg);
  const e = polarToXY(cx, cy, r, endDeg);
  const large = endDeg - startDeg > 180 ? 1 : 0;
  return `M ${s.x} ${s.y} A ${r} ${r} 0 ${large} 1 ${e.x} ${e.y}`;
}

function scoreToColor(score) {
  if (score <= 2) return "#f85149";
  if (score <= 3.5) return "#ff7f50";
  if (score <= 6.5) return "#d29922";
  if (score <= 8) return "#7ee787";
  return "#3fb950";
}

export default function ScoreGauge({ score = 5.0 }) {
  const needleRef = useRef(null);
  const arcRef = useRef(null);
  const scoreRef = useRef(null);
  const prevScore = useRef(score);

  useEffect(() => {
    const totalDeg = END_ANGLE - START_ANGLE; // 240 deg
    const ratio = Math.max(0, Math.min(10, score)) / 10;
    const fillEnd = START_ANGLE + totalDeg * ratio;
    const color = scoreToColor(score);

    if (arcRef.current) {
      arcRef.current.setAttribute("d", describeArc(CX, CY, R, START_ANGLE, fillEnd));
      arcRef.current.setAttribute("stroke", color);
    }

    const needleDeg = START_ANGLE + totalDeg * ratio;
    const needleTip = polarToXY(CX, CY, R - STROKE / 2 - 4, needleDeg);
    const needleBase = polarToXY(CX, CY, 12, needleDeg + 180);
    if (needleRef.current) {
      needleRef.current.setAttribute(
        "d",
        `M ${needleBase.x} ${needleBase.y} L ${needleTip.x} ${needleTip.y}`
      );
      needleRef.current.setAttribute("stroke", color);
    }

    if (scoreRef.current) {
      scoreRef.current.textContent = score.toFixed(1);
      scoreRef.current.setAttribute("fill", color);
    }
    prevScore.current = score;
  }, [score]);

  return (
    <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`}>
      {/* Background track */}
      <path
        d={describeArc(CX, CY, R, START_ANGLE, END_ANGLE)}
        fill="none"
        stroke="#30363d"
        strokeWidth={STROKE}
        strokeLinecap="round"
      />
      {/* Filled arc */}
      <path
        ref={arcRef}
        fill="none"
        strokeWidth={STROKE}
        strokeLinecap="round"
        d={describeArc(CX, CY, R, START_ANGLE, START_ANGLE)}
      />
      {/* Needle */}
      <path
        ref={needleRef}
        fill="none"
        strokeWidth={2}
        strokeLinecap="round"
        d={`M ${CX} ${CY} L ${CX} ${CY}`}
      />
      {/* Center dot */}
      <circle cx={CX} cy={CY} r={4} fill="#8b949e" />
      {/* Score label */}
      <text
        ref={scoreRef}
        x={CX}
        y={CY + 28}
        textAnchor="middle"
        fontSize="22"
        fontWeight="bold"
        fill="#3fb950"
        fontFamily="monospace"
      >
        {score.toFixed(1)}
      </text>
      {/* Min/Max labels */}
      <text x={14} y={CY + 30} fontSize="9" fill="#8b949e">0</text>
      <text x={SIZE - 20} y={CY + 30} fontSize="9" fill="#8b949e">10</text>
    </svg>
  );
}
