"""Prompt construction for the LLM decision layer."""


def build_recommendation_prompt(stations, peak_hour, avg_util, timestamp):
    """Build the Gemini prompt from a station snapshot.

    `stations` is a list of dicts with keys: id, name, location, utilization (0-1), risk.
    """
    def fmt_pct(u):
        return f"{u * 100:.0f}%"

    critical = [f"{s['id']}({fmt_pct(s['utilization'])})" for s in stations if s["risk"] == "critical"]
    high = [f"{s['id']}({fmt_pct(s['utilization'])})" for s in stations if s["risk"] == "high"]
    details = ", ".join(
        f"{s['id']}:{s['name']}@{s['location']}={fmt_pct(s['utilization'])}" for s in stations
    )

    return (
        "You are an EV Charging Grid Operations AI for the Chengalpattu–Tambaram NH45 corridor "
        "in Tamil Nadu, India.\n"
        "Analyze charging load data and provide exactly 3 actionable recommendations.\n"
        "Respond ONLY in this JSON format (no markdown, no extra text):\n"
        '{"recommendations":[{"action":"string","priority":"critical|high|medium",'
        '"impact":"string","stations":["id1","id2"]}],"summary":"one sentence grid status"}\n\n'
        "Current grid state:\n"
        "- Corridor: Chengalpattu to Tambaram (NH45), Tamil Nadu\n"
        f"- Time: {timestamp}\n"
        f"- Average utilization: {avg_util * 100:.1f}%\n"
        f"- Peak hour forecast: {peak_hour}:00\n"
        f"- Critical stations (>88%): {', '.join(critical) or 'none'}\n"
        f"- High-risk stations (72-88%): {', '.join(high) or 'none'}\n"
        f"- Station details: {details}"
    )
