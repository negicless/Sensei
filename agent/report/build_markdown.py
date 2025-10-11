from datetime import date

def render_md(code, funda, news, px, levels, ideas):
    last = px.iloc[-1]
    lines = []
    lines += [f"# {code} | Institutional Snapshot  ({date.today()})",
              "",
              "## Snapshot",
              f"- Close: **{last['c']:.2f}**  | Vol: {int(last['v']):,}",
              f"- PER: {funda.get('per','?')}  PBR: {funda.get('pbr','?')}  Yield: {funda.get('yield','?')}%",
              f"- 信用倍率: {funda.get('credit_ratio','?')}",
              f"- Levels → R: {levels['resistance']:.2f} / S: {levels['support']:.2f}",
              "",
              "## Recent News"]
    for n in news[:5]:
        lines.append(f"- {n.get('date','')} — {n.get('title','')}")

    lines += ["", "## Trade Ideas"]
    if ideas:
        for idea in ideas:
            lines += [
                f"### {idea['name']}",
                f"- Entry: **{idea['entry']:.2f}**  | SL: **{idea['stop']:.2f}**",
                f"- Targets: {', '.join(f'{t:.2f}' for t in idea['targets'])}",
                f"- R-table: {', '.join(f"T{idx+1}:{r['R']}R" for idx,r in enumerate(idea['r_table']))}",
            ]
    else:
        lines.append("_No high-quality setups detected today._")

    lines += ["", "## Notes",
              "- Event risk: check earnings/権利付き最終日 within 2 weeks.",
              "- Liquidity: size down if ADTV < your order size × 20."]
    return "\n".join(lines)
