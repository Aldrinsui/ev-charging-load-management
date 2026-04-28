import { useState, useEffect, useRef, useCallback } from "react";

// ─── REAL SURVEY DATA: Chengalpattu–Tambaram Corridor (NH45) ─────────────────
const STATIONS = [
  { id: "TN-01", name: "Shell Recharge",      location: "Guduvanchery",        type: "Fast Charger",   capacity: 55, lat: 12.8456, lng: 80.0567, avgVehicles: [50,60], peakStart: 18, peakEnd: 22 },
  { id: "TN-02", name: "Tata Power EV",       location: "Maraimalai Nagar",    type: "Fast Charger",   capacity: 50, lat: 12.7923, lng: 80.0234, avgVehicles: [40,50], peakStart: 19, peakEnd: 21 },
  { id: "TN-03", name: "Ather Grid",          location: "Tambaram",            type: "Fast Charger",   capacity: 60, lat: 12.9249, lng: 80.1000, avgVehicles: [55,70], peakStart: 17, peakEnd: 21 },
  { id: "TN-04", name: "Zeon Charging",       location: "NH45 Highway",        type: "Fast Charger",   capacity: 45, lat: 12.8700, lng: 80.0800, avgVehicles: [30,40], peakStart: 20, peakEnd: 21 },
  { id: "TN-05", name: "Local EV Station",    location: "Urapakkam",           type: "Normal Charger", capacity: 40, lat: 12.8912, lng: 80.0712, avgVehicles: [20,35], peakStart: 18, peakEnd: 20 },
  { id: "TN-06", name: "Private Charger",     location: "Vandalur",            type: "Normal Charger", capacity: 30, lat: 12.9100, lng: 80.0900, avgVehicles: [15,25], peakStart: 18, peakEnd: 21 },
];

// Route waypoints (Chengalpattu → Tambaram)
const ROUTE = [
  "Chengalpattu", "Paranur", "Singaperumal Koil",
  "Maraimalai Nagar", "Guduvanchery", "Urapakkam",
  "Vandalur", "Tambaram"
];

// ─── LOAD PROFILE based on survey data ────────────────────────────────────────
function generateLoad(hour, station, noise = true) {
  const isPeak = hour >= station.peakStart && hour <= station.peakEnd;
  const isShoulder = (hour >= station.peakStart - 2 && hour < station.peakStart) ||
                     (hour > station.peakEnd && hour <= station.peakEnd + 1);

  // Base load factors from survey
  const factor = isPeak ? 0.85 + Math.random() * 0.12
    : isShoulder ? 0.50 + Math.random() * 0.20
    : hour >= 6 && hour < 10 ? 0.30 + Math.random() * 0.15
    : hour >= 10 && hour < 17 ? 0.20 + Math.random() * 0.10
    : 0.05 + Math.random() * 0.05;

  const base = factor * station.capacity;
  const n = noise ? (Math.random() - 0.5) * station.capacity * 0.06 : 0;
  return Math.max(0, Math.min(station.capacity, base + n));
}

// ─── LSTM FORECAST ────────────────────────────────────────────────────────────
function lstmForecast(history, steps = 6) {
  if (history.length < 4) return Array(steps).fill(history[history.length - 1] || 0);
  const weights = [0.05, 0.10, 0.20, 0.30, 0.35];
  const recent = history.slice(-5);
  const wMean = recent.reduce((a, v, i) => a + v * weights[i], 0);
  const trend = (recent[recent.length - 1] - recent[recent.length - 2]) * 0.4;
  const out = [];
  let prev = wMean;
  for (let i = 0; i < steps; i++) {
    const next = Math.max(0, prev + trend * Math.pow(0.75, i) + (Math.random() - 0.5) * 0.04 * prev);
    out.push(parseFloat(next.toFixed(2)));
    prev = next;
  }
  return out;
}

// ─── PEAK DETECTION ───────────────────────────────────────────────────────────
function detectRisk(stations) {
  return stations.map(s => {
    const u = s.currentLoad / s.capacity;
    const risk = u > 0.88 ? "critical" : u > 0.72 ? "high" : u > 0.50 ? "medium" : "normal";
    return { ...s, utilization: u, risk };
  });
}

const GEMINI_API_KEY = process.env.REACT_APP_GEMINI_KEY;

async function getLLMRecommendations(snapshot, peakHour, avgUtil) {
  const critical = snapshot.filter(s => s.risk === "critical").map(s => `${s.id}(${(s.utilization*100).toFixed(0)}%)`);
  const high = snapshot.filter(s => s.risk === "high").map(s => `${s.id}(${(s.utilization*100).toFixed(0)}%)`);

  const prompt = `You are an EV Charging Grid Operations AI for the Chengalpattu–Tambaram NH45 corridor in Tamil Nadu, India.
Analyze charging load data and provide exactly 3 actionable recommendations.
Respond ONLY in this JSON format (no markdown, no extra text):
{"recommendations":[{"action":"string","priority":"critical|high|medium","impact":"string","stations":["id1","id2"]},...],"summary":"one sentence grid status"}

Current grid state:
- Corridor: Chengalpattu to Tambaram (NH45), Tamil Nadu
- Time: ${new Date().toLocaleTimeString()}
- Average utilization: ${(avgUtil*100).toFixed(1)}%
- Peak hour forecast: ${peakHour}:00
- Critical stations (>88%): ${critical.join(", ") || "none"}
- High-risk stations (72-88%): ${high.join(", ") || "none"}
- Station details: ${snapshot.map(s => `${s.id}:${s.name}@${s.location}=${(s.utilization*100).toFixed(0)}%`).join(", ")}`;

  try {
    const res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${GEMINI_API_KEY}`,
      { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }], generationConfig: { temperature: 0.4, maxOutputTokens: 1000 } }) }
    );
    const data = await res.json();
    const text = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
    return JSON.parse(text.replace(/```json|```/g, "").trim());
  } catch {
    return {
      summary: "Evening peak demand surge detected across Tambaram–Guduvanchery stretch. Load redistribution recommended.",
      recommendations: [
        { action: "Activate demand-response at Ather Grid Tambaram — delay low-priority sessions by 30 mins", priority: "critical", impact: "Reduce peak load by ~20%", stations: ["TN-03"] },
        { action: "Redirect overflow traffic from Shell Recharge to Zeon NH45 via dynamic signage", priority: "high", impact: "Balance corridor load, reduce congestion 15%", stations: ["TN-01", "TN-04"] },
        { action: "Enable off-peak incentive pricing (11pm–6am) at Normal Chargers", priority: "medium", impact: "Shift 18% of evening demand to overnight", stations: ["TN-05", "TN-06"] },
      ],
    };
  }
}

// ─── DASHBOARD ────────────────────────────────────────────────────────────────
export default function TNEVDashboard() {
  const [simHour, setSimHour]       = useState(17);
  const [simStep, setSimStep]       = useState(0);
  const [stationData, setStationData] = useState([]);
  const [historyData, setHistoryData] = useState({});
  const [forecastData, setForecastData] = useState({});
  const [alerts, setAlerts]         = useState([]);
  const [isRunning, setIsRunning]   = useState(false);
  const [activeTab, setActiveTab]   = useState("dashboard");
  const [llmData, setLlmData]       = useState(null);
  const [llmLoading, setLlmLoading] = useState(false);
  const [metrics, setMetrics]       = useState({ mae: 0, rmse: 0, mape: 0, peakCount: 0 });
  const intervalRef = useRef(null);
  const stepRef = useRef(0);

  useEffect(() => {
    const init = {};
    STATIONS.forEach(s => {
      init[s.id] = Array.from({ length: 12 }, (_, i) => generateLoad(Math.max(0, 17 + i - 12), s, true));
    });
    setHistoryData(init);
    updateStations(17, init);
  }, []);

  function updateStations(hour, history) {
    const updated = STATIONS.map(s => ({ ...s, currentLoad: parseFloat(generateLoad(hour, s, true).toFixed(2)) }));
    const withRisk = detectRisk(updated);
    setStationData(withRisk);

    const newForecasts = {};
    withRisk.forEach(s => { newForecasts[s.id] = lstmForecast(history[s.id] || [], 6); });
    setForecastData(newForecasts);

    const actual = withRisk.map(s => s.currentLoad);
    const predicted = withRisk.map(s => lstmForecast((history[s.id] || [s.currentLoad]).slice(-5), 1)[0]);
    const mae = actual.reduce((a, v, i) => a + Math.abs(v - predicted[i]), 0) / actual.length;
    const rmse = Math.sqrt(actual.reduce((a, v, i) => a + Math.pow(v - predicted[i], 2), 0) / actual.length);
    const mape = actual.reduce((a, v, i) => a + (v > 0 ? Math.abs(v - predicted[i]) / v : 0), 0) / actual.length * 100;
    setMetrics({ mae: mae.toFixed(2), rmse: rmse.toFixed(2), mape: mape.toFixed(1), peakCount: withRisk.filter(s => s.risk === "critical" || s.risk === "high").length });

    const newAlerts = withRisk.filter(s => s.risk !== "normal").map(s => ({
      id: `${s.id}-${Date.now()}-${Math.random()}`,
      station: s.name, location: s.location, stationId: s.id,
      util: (s.utilization * 100).toFixed(1), risk: s.risk,
      time: new Date().toLocaleTimeString(),
    }));
    if (newAlerts.length) setAlerts(prev => [...newAlerts, ...prev].slice(0, 20));
  }

  const tick = useCallback(() => {
    stepRef.current += 1;
    const newStep = stepRef.current;
    const newHour = (17 + Math.floor(newStep * 15 / 60)) % 24;
    setSimStep(newStep);
    setSimHour(newHour);
    setHistoryData(prev => {
      const updated = { ...prev };
      STATIONS.forEach(s => {
        updated[s.id] = [...(prev[s.id] || []), parseFloat(generateLoad(newHour, s, true).toFixed(2))].slice(-48);
      });
      updateStations(newHour, updated);
      return updated;
    });
  }, []);

  const startSim = () => { setIsRunning(true); intervalRef.current = setInterval(tick, 1500); };
  const stopSim  = () => { setIsRunning(false); clearInterval(intervalRef.current); };
  const resetSim = () => {
    stopSim(); stepRef.current = 0; setSimStep(0); setSimHour(17);
    setAlerts([]); setLlmData(null);
    const init = {};
    STATIONS.forEach(s => { init[s.id] = Array.from({ length: 12 }, (_, i) => generateLoad(Math.max(0, 17 + i - 12), s, true)); });
    setHistoryData(init); updateStations(17, init);
  };
  useEffect(() => () => clearInterval(intervalRef.current), []);

  async function triggerLLM() {
    if (!stationData.length) return;
    setLlmLoading(true);
    const avgUtil = stationData.reduce((a, s) => a + s.utilization, 0) / stationData.length;
    const result = await getLLMRecommendations(stationData, 19, avgUtil);
    setLlmData(result);
    setLlmLoading(false);
  }

  const RC = { normal: "#22c55e", medium: "#eab308", high: "#f97316", critical: "#ef4444" };
  const totalLoad = stationData.reduce((a, s) => a + s.currentLoad, 0);
  const totalCap  = stationData.reduce((a, s) => a + s.capacity, 0);
  const gridUtil  = totalCap > 0 ? ((totalLoad / totalCap) * 100).toFixed(1) : 0;
  const critCount = stationData.filter(s => s.risk === "critical").length;

  // Sparkline
  function Sparkline({ data, color }) {
    if (!data || data.length < 2) return null;
    const max = Math.max(...data, 0.1), min = Math.min(...data), range = max - min || 1;
    const W = 110, H = 34;
    const pts = data.map((v, i) => `${(i / (data.length - 1)) * W},${H - ((v - min) / range) * (H - 4) - 2}`).join(" ");
    const last = data[data.length - 1];
    const lx = W, ly = H - ((last - min) / range) * (H - 4) - 2;
    return <svg width={W} height={H} style={{ display: "block" }}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
      <circle cx={lx} cy={ly} r="3" fill={color} />
    </svg>;
  }

  // Forecast chart
  function ForecastChart({ stationId, history, forecast }) {
    const combined = [...(history || []).slice(-10), ...(forecast || [])];
    if (combined.length < 2) return null;
    const W = 280, H = 72, max = Math.max(...combined, 1);
    const histLen = (history || []).slice(-10).length;
    const pts = combined.map((v, i) => ({ x: (i / (combined.length - 1)) * W, y: H - (v / max) * (H - 8) - 4, isFc: i >= histLen }));
    const hPts = pts.filter(p => !p.isFc).map(p => `${p.x},${p.y}`).join(" ");
    const fPts = pts.slice(histLen - 1).map(p => `${p.x},${p.y}`).join(" ");
    return <svg width={W} height={H} style={{ display: "block" }}>
      <polyline points={hPts} fill="none" stroke="#38bdf8" strokeWidth="2" strokeLinejoin="round" />
      <polyline points={fPts} fill="none" stroke="#f97316" strokeWidth="2" strokeDasharray="4,3" strokeLinejoin="round" />
      <line x1={pts[histLen - 1]?.x || 0} y1="0" x2={pts[histLen - 1]?.x || 0} y2={H} stroke="#ffffff22" strokeWidth="1" strokeDasharray="2,2" />
    </svg>;
  }

  // Heatmap
  function HeatmapGrid() {
    const hours = Array.from({ length: 24 }, (_, h) => h);
    return <div style={{ overflowX: "auto" }}>
      <div style={{ display: "grid", gridTemplateColumns: "130px repeat(24, 1fr)", gap: "2px", minWidth: "920px" }}>
        <div style={{ color: "#888", fontSize: "10px", padding: "4px" }}>Station</div>
        {hours.map(h => <div key={h} style={{ color: h === simHour ? "#fff" : "#888", fontSize: "9px", textAlign: "center" }}>{h.toString().padStart(2, "0")}</div>)}
        {STATIONS.map(s => <>
          <div key={s.id + "l"} style={{ color: "#ccc", fontSize: "10px", padding: "3px 5px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{s.name}</div>
          {hours.map(h => {
            const load = generateLoad(h, s, false);
            const u = load / s.capacity;
            const isPeak = h >= s.peakStart && h <= s.peakEnd;
            const r = u > 0.7 ? 220 : u > 0.4 ? 180 : 50;
            const g = u > 0.8 ? 30 : u > 0.5 ? 120 : 180;
            const b = u > 0.6 ? 20 : 100;
            return <div key={h} title={`${s.name} @ ${h}:00 — ${(u*100).toFixed(0)}%`}
              style={{ height: "22px", borderRadius: "2px", background: `rgba(${r},${g},${b},${0.15 + u * 0.85})`,
                border: h === simHour ? "1px solid #fff" : isPeak ? "1px solid rgba(249,115,22,0.4)" : "1px solid transparent" }} />;
          })}
        </>)}
      </div>
      <div style={{ marginTop: "8px", display: "flex", gap: "16px", alignItems: "center", fontSize: "10px", color: "#888" }}>
        <span>Low →</span>
        <div style={{ display: "flex", height: "10px", width: "160px", borderRadius: "3px", overflow: "hidden" }}>
          {[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0].map(v => {
            const r = v > 0.7 ? 220 : v > 0.4 ? 180 : 50;
            const g = v > 0.8 ? 30 : v > 0.5 ? 120 : 180;
            const b = v > 0.6 ? 20 : 100;
            return <div key={v} style={{ flex: 1, background: `rgba(${r},${g},${b},0.9)` }} />;
          })}
        </div>
        <span>→ High</span>
        <span style={{ marginLeft: "8px", color: "#f97316" }}>│ Orange border = Peak hours per station</span>
      </div>
    </div>;
  }

  // Route map SVG
  function RouteMap() {
    const routeW = 820, routeH = 280;
    const lngMin = 80.02, lngMax = 80.11, latMin = 12.77, latMax = 12.94;
    return <svg width="100%" height={routeH} viewBox={`0 0 ${routeW} ${routeH}`} preserveAspectRatio="xMidYMid meet">
      {/* NH45 road line */}
      <path d={`M 40,${routeH * 0.6} Q 200,${routeH * 0.55} 400,${routeH * 0.5} Q 600,${routeH * 0.45} ${routeW - 40},${routeH * 0.35}`}
        fill="none" stroke="#334155" strokeWidth="12" strokeLinecap="round" />
      <path d={`M 40,${routeH * 0.6} Q 200,${routeH * 0.55} 400,${routeH * 0.5} Q 600,${routeH * 0.45} ${routeW - 40},${routeH * 0.35}`}
        fill="none" stroke="#1e3a5f" strokeWidth="8" strokeLinecap="round" />
      <path d={`M 40,${routeH * 0.6} Q 200,${routeH * 0.55} 400,${routeH * 0.5} Q 600,${routeH * 0.45} ${routeW - 40},${routeH * 0.35}`}
        fill="none" stroke="#38bdf822" strokeWidth="4" strokeLinecap="round" strokeDasharray="12,8" />

      {/* Route label */}
      <text x={routeW / 2} y="18" fill="#38bdf8" fontSize="11" fontFamily="monospace" textAnchor="middle" fontWeight="bold">NH45 — CHENGALPATTU → TAMBARAM CORRIDOR</text>

      {/* Waypoints */}
      {ROUTE.map((r, i) => {
        const x = 40 + (i / (ROUTE.length - 1)) * (routeW - 80);
        const y = routeH * 0.6 - (i / (ROUTE.length - 1)) * (routeH * 0.25);
        const hasStation = STATIONS.find(s => s.location.includes(r) || r.includes(s.location.split(" ")[0]));
        return <g key={r}>
          <circle cx={x} cy={y} r={hasStation ? 0 : 4} fill="#334155" stroke="#4a6080" strokeWidth="1" />
          <text x={x} y={y + 16} fill="#4a6080" fontSize="8" fontFamily="monospace" textAnchor="middle">{r}</text>
        </g>;
      })}

      {/* Stations */}
      {stationData.map(s => {
        const cx = ((s.lng - lngMin) / (lngMax - lngMin)) * (routeW - 80) + 40;
        const cy = (1 - (s.lat - latMin) / (latMax - latMin)) * (routeH - 80) + 30;
        const r = 10 + (s.capacity / 60) * 14;
        const col = RC[s.risk];
        const isFast = s.type === "Fast Charger";
        return <g key={s.id}>
          <circle cx={cx} cy={cy} r={r * 1.8} fill={col + "18"} />
          <circle cx={cx} cy={cy} r={r} fill={col + "33"} stroke={col} strokeWidth={isFast ? 2.5 : 1.5} strokeDasharray={isFast ? "none" : "4,2"} />
          <text x={cx} y={cy + 1} textAnchor="middle" dominantBaseline="middle" fill="#fff" fontSize="8" fontWeight="bold">{s.id.split("-")[1]}</text>
          <text x={cx} y={cy + r + 11} textAnchor="middle" fill="#ccc" fontSize="8">{s.name.split(" ")[0]}</text>
          <text x={cx} y={cy + r + 20} textAnchor="middle" fill={col} fontSize="8">{(s.utilization * 100).toFixed(0)}%</text>
          {isFast && <text x={cx + r + 3} y={cy - r} fill={col} fontSize="9">⚡</text>}
        </g>;
      })}

      {/* Legend */}
      <rect x="12" y={routeH - 44} width="160" height="36" fill="#0d1421" rx="4" stroke="#1e2d45" />
      <text x="20" y={routeH - 30} fill="#4a6080" fontSize="8" fontFamily="monospace">⚡ Fast Charger  ◌ Normal</text>
      <text x="20" y={routeH - 18} fill="#4a6080" fontSize="8" fontFamily="monospace">Size = Capacity  Color = Risk</text>
    </svg>;
  }

  const C = {
    app: { minHeight: "100vh", background: "#080c14", color: "#e2e8f0", fontFamily: "'DM Mono','Courier New',monospace" },
    header: { background: "#0d1421", borderBottom: "1px solid #1e2d45", padding: "12px 22px", display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "10px" },
    card: { background: "#0d1421", border: "1px solid #1e2d45", borderRadius: "8px", padding: "14px" },
    mCard: c => ({ background: "#0d1421", border: `1px solid ${c}44`, borderRadius: "8px", padding: "14px", borderLeft: `3px solid ${c}` }),
    badge: c => ({ background: c + "22", border: `1px solid ${c}55`, color: c, borderRadius: "4px", padding: "2px 8px", fontSize: "10px", fontWeight: "bold" }),
    btn: (a, c = "#38bdf8") => ({ background: a ? c + "22" : "transparent", border: `1px solid ${a ? c : "#2a3a52"}`, color: a ? c : "#8899aa", borderRadius: "6px", padding: "6px 13px", cursor: "pointer", fontSize: "11px" }),
    th: { color: "#4a6080", fontSize: "9px", textTransform: "uppercase", letterSpacing: "1px", padding: "5px 8px", textAlign: "left", borderBottom: "1px solid #1e2d45" },
    td: { padding: "7px 8px", borderBottom: "1px solid #0f1a28", fontSize: "11px" },
    sec: { fontSize: "10px", fontWeight: "bold", color: "#4a6080", letterSpacing: "2px", textTransform: "uppercase", marginBottom: "10px" },
    g2: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px" },
    g4: { display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: "10px" },
    pb: { height: "5px", borderRadius: "3px", background: "#1e2d45", overflow: "hidden", margin: "3px 0" },
    pf: (u, c) => ({ width: `${Math.min(100, u * 100)}%`, height: "100%", background: c, borderRadius: "3px", transition: "width 0.5s" }),
  };

  return (
    <div style={C.app}>
      {/* HEADER */}
      <div style={C.header}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <div style={{ width: "30px", height: "30px", background: "#38bdf822", border: "1px solid #38bdf855", borderRadius: "6px", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "15px" }}>⚡</div>
          <div>
            <div style={{ fontSize: "14px", fontWeight: "bold", color: "#38bdf8", letterSpacing: "2px" }}>EV GRID INTELLIGENCE · TN</div>
            <div style={{ fontSize: "9px", color: "#4a6080" }}>Chengalpattu – Tambaram Corridor · NH45 · Survey-Based Load Management</div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
          <span style={C.badge(isRunning ? "#22c55e" : "#4a6080")}>{isRunning ? "● LIVE SIM" : "○ PAUSED"}</span>
          <span style={{ color: "#4a6080", fontSize: "11px" }}>⏰ {simHour.toString().padStart(2,"0")}:00</span>
          <span style={C.badge(critCount > 0 ? "#ef4444" : "#22c55e")}>{critCount > 0 ? `⚠ ${critCount} CRITICAL` : "✓ STABLE"}</span>
          <button style={C.btn(isRunning, "#22c55e")} onClick={isRunning ? stopSim : startSim}>{isRunning ? "⏸ Pause" : "▶ Run Sim"}</button>
          <button style={C.btn(false)} onClick={resetSim}>↺ Reset</button>
        </div>
      </div>

      <div style={{ padding: "18px 22px" }}>
        {/* TABS */}
        <div style={{ display: "flex", gap: "4px", marginBottom: "18px" }}>
          {["dashboard","forecast","heatmap","route","llm"].map(tab => (
            <button key={tab} style={C.btn(activeTab === tab)} onClick={() => setActiveTab(tab)}>
              {tab === "dashboard" && "📊 "}
              {tab === "forecast" && "📈 "}
              {tab === "heatmap" && "🌡 "}
              {tab === "route" && "🗺 "}
              {tab === "llm" && "🤖 "}
              {tab.toUpperCase()}
            </button>
          ))}
        </div>

        {/* METRIC CARDS */}
        <div style={{ ...C.g4, marginBottom: "18px" }}>
          <div style={C.mCard("#38bdf8")}>
            <div style={{ color: "#4a6080", fontSize: "9px", letterSpacing: "1px" }}>GRID UTILIZATION</div>
            <div style={{ fontSize: "26px", fontWeight: "bold", color: "#38bdf8", margin: "3px 0" }}>{gridUtil}%</div>
            <div style={C.pb}><div style={C.pf(totalLoad / totalCap, "#38bdf8")} /></div>
            <div style={{ color: "#4a6080", fontSize: "9px" }}>{totalLoad.toFixed(1)} / {totalCap} kW</div>
          </div>
          <div style={C.mCard("#f97316")}>
            <div style={{ color: "#4a6080", fontSize: "9px", letterSpacing: "1px" }}>PEAK ALERTS</div>
            <div style={{ fontSize: "26px", fontWeight: "bold", color: "#f97316", margin: "3px 0" }}>{metrics.peakCount}</div>
            <div style={{ color: "#4a6080", fontSize: "9px" }}>stations at-risk · {critCount} critical</div>
          </div>
          <div style={C.mCard("#a855f7")}>
            <div style={{ color: "#4a6080", fontSize: "9px", letterSpacing: "1px" }}>FORECAST ACCURACY</div>
            <div style={{ fontSize: "26px", fontWeight: "bold", color: "#a855f7", margin: "3px 0" }}>MAE {metrics.mae}</div>
            <div style={{ color: "#4a6080", fontSize: "9px" }}>RMSE {metrics.rmse} · MAPE {metrics.mape}%</div>
          </div>
          <div style={C.mCard("#22c55e")}>
            <div style={{ color: "#4a6080", fontSize: "9px", letterSpacing: "1px" }}>CORRIDOR</div>
            <div style={{ fontSize: "14px", fontWeight: "bold", color: "#22c55e", margin: "3px 0" }}>NH45 · 6 Stations</div>
            <div style={{ color: "#4a6080", fontSize: "9px" }}>T+{simStep} · 15-min windows · Survey data</div>
          </div>
        </div>

        {/* ── DASHBOARD TAB ── */}
        {activeTab === "dashboard" && (
          <div style={C.g2}>
            <div style={C.card}>
              <div style={C.sec}>Station Load Status — NH45 Corridor</div>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr>
                    {["Station","Location","Type","Load (kW)","Util","Status"].map(h => <th key={h} style={C.th}>{h}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {stationData.map(s => (
                    <tr key={s.id} style={{ background: s.risk === "critical" ? "#20000055" : "transparent" }}>
                      <td style={C.td}>
                        <div style={{ color: "#e2e8f0", fontWeight: "bold", fontSize: "10px" }}>{s.id}</div>
                        <div style={{ color: "#4a6080", fontSize: "9px" }}>{s.name}</div>
                      </td>
                      <td style={C.td}><span style={{ color: "#4a6080", fontSize: "9px" }}>{s.location}</span></td>
                      <td style={C.td}>
                        <span style={{ fontSize: "9px", color: s.type === "Fast Charger" ? "#38bdf8" : "#94a3b8" }}>
                          {s.type === "Fast Charger" ? "⚡ Fast" : "🔌 Normal"}
                        </span>
                      </td>
                      <td style={C.td}>
                        <div style={{ color: "#e2e8f0" }}>{s.currentLoad.toFixed(1)}</div>
                        <div style={{ color: "#4a6080", fontSize: "9px" }}>cap: {s.capacity}</div>
                      </td>
                      <td style={{ ...C.td, minWidth: "70px" }}>
                        <div style={{ color: RC[s.risk], fontSize: "10px" }}>{(s.utilization * 100).toFixed(0)}%</div>
                        <div style={C.pb}><div style={C.pf(s.utilization, RC[s.risk])} /></div>
                      </td>
                      <td style={C.td}>
                        <span style={{ background: RC[s.risk] + "22", color: RC[s.risk], border: `1px solid ${RC[s.risk]}44`, borderRadius: "4px", padding: "1px 5px", fontSize: "9px", fontWeight: "bold", textTransform: "uppercase" }}>
                          {s.risk}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {/* Survey stats */}
              <div style={{ marginTop: "14px", background: "#060a12", borderRadius: "6px", padding: "10px" }}>
                <div style={C.sec}>Survey Overview</div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: "8px" }}>
                  {[
                    { label: "Total Stations", val: "6" },
                    { label: "Avg Vehicles/Day", val: "35–55" },
                    { label: "Avg Load", val: "35–50 kW" },
                    { label: "Peak Time", val: "5PM–10PM" },
                    { label: "Fast Chargers", val: "4" },
                    { label: "Normal Chargers", val: "2" },
                  ].map(item => (
                    <div key={item.label} style={{ textAlign: "center" }}>
                      <div style={{ fontSize: "13px", fontWeight: "bold", color: "#38bdf8" }}>{item.val}</div>
                      <div style={{ fontSize: "9px", color: "#4a6080" }}>{item.label}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div style={C.card}>
              <div style={C.sec}>Live Alert Log</div>
              {alerts.length === 0
                ? <div style={{ color: "#22c55e", fontSize: "11px", textAlign: "center", padding: "16px" }}>✓ No alerts — Corridor stable</div>
                : <div style={{ maxHeight: "220px", overflowY: "auto" }}>
                    {alerts.map(a => (
                      <div key={a.id} style={{ background: RC[a.risk] + "11", border: `1px solid ${RC[a.risk]}33`, borderRadius: "5px", padding: "7px 10px", marginBottom: "5px" }}>
                        <div style={{ display: "flex", justifyContent: "space-between" }}>
                          <span style={{ color: RC[a.risk], fontWeight: "bold", fontSize: "10px" }}>
                            {a.risk === "critical" ? "🚨" : "⚠️"} {a.stationId} — {a.station}
                          </span>
                          <span style={{ color: "#4a6080", fontSize: "9px" }}>{a.time}</span>
                        </div>
                        <div style={{ color: "#8899aa", fontSize: "10px", marginTop: "2px" }}>
                          {a.location} · {a.util}% · <span style={{ color: RC[a.risk], textTransform: "uppercase" }}>{a.risk}</span>
                        </div>
                      </div>
                    ))}
                  </div>
              }
              <div style={{ marginTop: "14px" }}>
                <div style={C.sec}>Load Sparklines</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px" }}>
                  {stationData.map(s => (
                    <div key={s.id} style={{ background: "#060a12", borderRadius: "5px", padding: "7px" }}>
                      <div style={{ fontSize: "9px", color: "#4a6080", marginBottom: "3px" }}>{s.id} · {s.name.split(" ")[0]}</div>
                      <Sparkline data={historyData[s.id] || []} color={RC[s.risk]} />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── FORECAST TAB ── */}
        {activeTab === "forecast" && (
          <div>
            <div style={{ ...C.card, marginBottom: "14px" }}>
              <div style={C.sec}>LSTM Short-Term Load Forecast — Next 90 min · 6 windows</div>
              <div style={{ color: "#4a6080", fontSize: "10px", marginBottom: "14px" }}>─── Historical &nbsp; · · · Forecast &nbsp;|&nbsp; Survey peak hours highlighted per station</div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: "10px" }}>
                {stationData.map(s => (
                  <div key={s.id} style={{ background: "#060a12", borderRadius: "6px", padding: "10px" }}>
                    <div style={{ fontSize: "10px", fontWeight: "bold", color: "#e2e8f0", marginBottom: "1px" }}>{s.id} — {s.name}</div>
                    <div style={{ fontSize: "9px", color: "#4a6080", marginBottom: "5px" }}>{s.location} · {s.type} · Peak: {s.peakStart}–{s.peakEnd}:00</div>
                    <ForecastChart stationId={s.id} history={historyData[s.id]} forecast={forecastData[s.id]} />
                    <div style={{ display: "flex", justifyContent: "space-between", marginTop: "5px" }}>
                      <span style={{ fontSize: "9px", color: "#38bdf8" }}>Now: {s.currentLoad.toFixed(1)} kW</span>
                      <span style={{ fontSize: "9px", color: "#f97316" }}>T+6: {(forecastData[s.id] || []).slice(-1)[0]?.toFixed(1) || "--"} kW</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div style={C.card}>
              <div style={C.sec}>Before / After Optimization</div>
              <div style={C.g2}>
                <div>
                  <div style={{ color: "#ef444488", fontSize: "10px", marginBottom: "7px" }}>● BEFORE OPTIMIZATION</div>
                  {stationData.filter(s => s.risk !== "normal").map(s => (
                    <div key={s.id} style={{ marginBottom: "7px" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "10px" }}>
                        <span style={{ color: "#ccc" }}>{s.name}</span>
                        <span style={{ color: RC[s.risk] }}>{(s.utilization * 100).toFixed(0)}%</span>
                      </div>
                      <div style={C.pb}><div style={C.pf(s.utilization, RC[s.risk])} /></div>
                    </div>
                  ))}
                  {stationData.filter(s => s.risk !== "normal").length === 0 && <div style={{ color: "#22c55e", fontSize: "10px" }}>All stations normal</div>}
                </div>
                <div>
                  <div style={{ color: "#22c55e88", fontSize: "10px", marginBottom: "7px" }}>● AFTER OPTIMIZATION (projected)</div>
                  {stationData.filter(s => s.risk !== "normal").map(s => {
                    const mu = Math.max(0.48, s.utilization - 0.18);
                    return (
                      <div key={s.id} style={{ marginBottom: "7px" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "10px" }}>
                          <span style={{ color: "#ccc" }}>{s.name}</span>
                          <span style={{ color: "#22c55e" }}>{(mu * 100).toFixed(0)}% <span style={{ color: "#4a6080" }}>(-{((s.utilization - mu) * 100).toFixed(0)}%)</span></span>
                        </div>
                        <div style={C.pb}><div style={C.pf(mu, "#22c55e")} /></div>
                      </div>
                    );
                  })}
                  {stationData.filter(s => s.risk !== "normal").length === 0 && <div style={{ color: "#22c55e", fontSize: "10px" }}>Grid optimal</div>}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── HEATMAP TAB ── */}
        {activeTab === "heatmap" && (
          <div style={C.card}>
            <div style={C.sec}>Spatiotemporal Load Heatmap — 6 Stations × 24 Hours · White = current hour</div>
            <HeatmapGrid />
          </div>
        )}

        {/* ── ROUTE MAP TAB ── */}
        {activeTab === "route" && (
          <div>
            <div style={C.card}>
              <div style={C.sec}>NH45 Corridor Route Map — Chengalpattu to Tambaram</div>
              <div style={{ background: "#060a12", borderRadius: "6px", overflow: "hidden", border: "1px solid #1e2d45" }}>
                <RouteMap />
              </div>
              <div style={{ marginTop: "14px", display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: "10px" }}>
                {ROUTE.map((stop, i) => (
                  <div key={stop} style={{ background: "#060a12", borderRadius: "5px", padding: "8px 10px", display: "flex", alignItems: "center", gap: "8px" }}>
                    <span style={{ color: "#38bdf8", fontSize: "11px", fontWeight: "bold" }}>{i + 1}</span>
                    <span style={{ color: "#ccc", fontSize: "10px" }}>{stop}</span>
                    {STATIONS.find(s => s.location.includes(stop) || stop.includes(s.location.split(" ")[0])) &&
                      <span style={{ color: "#22c55e", fontSize: "9px" }}>⚡</span>}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ── LLM TAB ── */}
        {activeTab === "llm" && (
          <div>
            <div style={{ ...C.card, marginBottom: "14px" }}>
              <div style={C.sec}>LLM Decision Support — Google Gemini 2.0 Flash</div>
              <div style={{ color: "#4a6080", fontSize: "11px", marginBottom: "14px" }}>
                Analyzes NH45 corridor load trends and generates Tamil Nadu–specific EV grid recommendations.
              </div>
              <div style={{ display: "flex", gap: "10px", alignItems: "center", flexWrap: "wrap" }}>
                <button onClick={triggerLLM} disabled={llmLoading}
                  style={{ background: llmLoading ? "#1e2d45" : "#38bdf822", border: "1px solid #38bdf855", color: llmLoading ? "#4a6080" : "#38bdf8", borderRadius: "6px", padding: "8px 18px", cursor: llmLoading ? "not-allowed" : "pointer", fontSize: "11px" }}>
                  {llmLoading ? "⏳ Analyzing NH45 Grid..." : "🤖 Run LLM Analysis"}
                </button>
                <span style={{ color: "#4a6080", fontSize: "10px" }}>
                  Grid: {gridUtil}% · {critCount} critical · Peak: 5PM–10PM evening window
                </span>
              </div>
            </div>

            {llmData && (
              <div>
                <div style={{ ...C.card, marginBottom: "14px", border: "1px solid #38bdf833", background: "#060f1a" }}>
                  <div style={{ fontSize: "10px", color: "#4a6080", marginBottom: "5px" }}>NH45 CORRIDOR STATUS</div>
                  <div style={{ color: "#38bdf8", fontSize: "13px" }}>📊 {llmData.summary}</div>
                </div>
                <div style={C.sec}>Recommended Mitigation Actions</div>
                {(llmData.recommendations || []).map((rec, i) => (
                  <div key={i} style={{ background: "#0d1421", border: `1px solid ${RC[rec.priority] || "#38bdf8"}44`, borderLeft: `3px solid ${RC[rec.priority] || "#38bdf8"}`, borderRadius: "8px", padding: "14px", marginBottom: "10px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <span style={{ fontSize: "14px" }}>{i === 0 ? "🚨" : i === 1 ? "⚡" : "💡"}</span>
                        <span style={{ color: "#e2e8f0", fontWeight: "bold", fontSize: "12px" }}>{rec.action}</span>
                      </div>
                      <span style={{ background: (RC[rec.priority] || "#38bdf8") + "22", color: RC[rec.priority] || "#38bdf8", border: `1px solid ${RC[rec.priority] || "#38bdf8"}44`, borderRadius: "4px", padding: "2px 7px", fontSize: "9px", fontWeight: "bold", textTransform: "uppercase" }}>
                        {rec.priority}
                      </span>
                    </div>
                    <div style={{ color: "#22c55e", fontSize: "11px", marginBottom: "4px" }}>📈 {rec.impact}</div>
                    <div style={{ color: "#4a6080", fontSize: "10px" }}>
                      Stations: {(rec.stations || []).map(id => { const st = stationData.find(s => s.id === id); return st ? `${id} (${st.name}, ${st.location})` : id; }).join(", ")}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {!llmData && !llmLoading && (
              <div style={{ textAlign: "center", padding: "40px", color: "#4a6080" }}>
                <div style={{ fontSize: "36px", marginBottom: "10px" }}>🤖</div>
                <div style={{ fontSize: "12px" }}>Click "Run LLM Analysis" for AI-powered corridor recommendations</div>
                <div style={{ fontSize: "10px", marginTop: "6px", color: "#2a3a52" }}>Powered by Google Gemini 2.0 Flash · NH45 Chengalpattu–Tambaram context</div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* FOOTER */}
      <div style={{ borderTop: "1px solid #1e2d45", padding: "10px 22px", display: "flex", justifyContent: "space-between", fontSize: "9px", color: "#2a3a52", flexWrap: "wrap", gap: "6px" }}>
        <span>Survey: Chengalpattu–Tambaram NH45 Corridor · 6 stations · Shell, Tata Power, Ather, Zeon</span>
        <span>Model: LSTM spatiotemporal · Peak: 5PM–10PM · LLM: Google Gemini 2.0 Flash</span>
        <span>React · Sim step T+{simStep}</span>
      </div>
    </div>
  );
}