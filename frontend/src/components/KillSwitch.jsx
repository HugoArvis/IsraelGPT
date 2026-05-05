import { useState } from "react";

export default function KillSwitch({ onConfirm, disabled }) {
  const [confirming, setConfirming] = useState(false);

  if (confirming) {
    return (
      <div className="border border-brand-red rounded-lg p-4 space-y-3">
        <p className="text-brand-red font-semibold text-sm">
          This will liquidate ALL positions immediately. Are you sure?
        </p>
        <div className="flex gap-3">
          <button
            onClick={() => {
              setConfirming(false);
              onConfirm();
            }}
            disabled={disabled}
            className="flex-1 py-2 rounded bg-brand-red text-white font-bold text-sm hover:opacity-80 disabled:opacity-40"
          >
            Yes, Kill All
          </button>
          <button
            onClick={() => setConfirming(false)}
            className="flex-1 py-2 rounded bg-brand-card border border-brand-border text-sm hover:bg-white/5"
          >
            Cancel
          </button>
        </div>
      </div>
    );
  }

  return (
    <button
      onClick={() => setConfirming(true)}
      disabled={disabled}
      className="w-full py-3 rounded-lg bg-brand-red text-white font-bold tracking-wide text-sm hover:opacity-80 disabled:opacity-40 transition-opacity"
    >
      KILL SWITCH — LIQUIDATE ALL
    </button>
  );
}
