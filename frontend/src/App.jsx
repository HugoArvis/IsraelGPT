import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Metrics from "./pages/Metrics";
import Trades from "./pages/Trades";
import Control from "./pages/Control";

const navItems = [
  { to: "/", label: "Dashboard" },
  { to: "/metrics", label: "Metrics" },
  { to: "/trades", label: "Trades" },
  { to: "/control", label: "Control" },
];

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col">
        <nav className="bg-brand-card border-b border-brand-border px-6 py-3 flex items-center gap-8">
          <span className="text-brand-accent font-bold text-lg tracking-tight">
            Trading AI
          </span>
          {navItems.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `text-sm transition-colors ${
                  isActive
                    ? "text-brand-accent border-b-2 border-brand-accent pb-0.5"
                    : "text-gray-400 hover:text-white"
                }`
              }
            >
              {label}
            </NavLink>
          ))}
        </nav>
        <main className="flex-1 p-6">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/metrics" element={<Metrics />} />
            <Route path="/trades" element={<Trades />} />
            <Route path="/control" element={<Control />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
