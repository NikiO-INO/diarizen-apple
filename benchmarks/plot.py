#!/usr/bin/env python3
"""Render the benchmark infographic (light + dark SVG) from the numbers in
RESULTS.md. Charts as code: edit the DATA below and re-run.

    python benchmarks/plot.py

Palette is the validated categorical default (blue = native CoreML, orange =
PyTorch); see the diarizen-apple README for how it is used.
"""
from pathlib import Path

# --- data (from benchmarks/RESULTS.md; M2 Pro, base-s80-md, 30 s clip) ---
# group: "coreml" or "pytorch"; charts read best (lowest) first.
SPEED = {  # real-time factor, lower is better
    "title": "Real-time factor",
    "unit": "",
    "rows": [
        ("CoreML CPU+GPU", 0.037, "coreml"),
        ("PyTorch MPS", 0.048, "pytorch"),
        ("CoreML CPU", 0.057, "coreml"),
        ("PyTorch CPU", 0.602, "pytorch"),
    ],
}
MEMORY = {  # peak RSS in MB, lower is better
    "title": "Peak memory (MB)",
    "unit": "",
    "rows": [
        ("CoreML CPU+GPU", 298, "coreml"),
        ("CoreML CPU", 1050, "coreml"),
        ("PyTorch MPS", 1832, "pytorch"),
        ("PyTorch CPU", 7474, "pytorch"),
    ],
}
ACC = {  # DER % on AMI EN2002a, lower is better
    "title": "Diarization error rate, % (AMI EN2002a)",
    "unit": "%",
    "rows": [
        ("large-s80-md-v2", 17.50, "coreml"),
        ("DiariZen base (PyTorch)", 21.10, "pytorch"),
        ("base-s80-md", 21.16, "coreml"),
    ],
}
PANELS = [SPEED, MEMORY, ACC]

THEME = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "sub": "#52514e",
        "muted": "#898781", "baseline": "#d7d6cf",
        "coreml": "#2a78d6", "pytorch": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "sub": "#c3c2b7",
        "muted": "#898781", "baseline": "#383835",
        "coreml": "#3987e5", "pytorch": "#d95926",
    },
}

W = 920
LM = 224           # left column for category labels
RM = 84            # right reserve for value labels
BAR_AREA = W - LM - RM
ROW_H = 34
BAR_H = 18
PANEL_TITLE_H = 30
PANEL_GAP = 16
TOP = 66           # header (title + legend)
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, 'SF Mono', SFMono-Regular, Menlo, Consolas, monospace"


def fmt(v, unit):
    if unit == "%":
        return f"{v:.2f}{unit}"
    s = f"{v:g}" if v < 100 else f"{int(round(v)):,}"
    return f"{s}{unit}"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render(mode):
    c = THEME[mode]
    # total height
    h = TOP
    for p in PANELS:
        h += PANEL_TITLE_H + len(p["rows"]) * ROW_H + PANEL_GAP
    h += 24  # footer
    out = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" '
        f'height="{h}" role="img" aria-label="diarizen-apple benchmarks">'
    )
    out.append(f'<rect x="0" y="0" width="{W}" height="{h}" fill="{c["surface"]}"/>')
    # header: title + legend
    out.append(
        f'<text x="0" y="30" font-family="{SANS}" font-size="21" font-weight="700" '
        f'fill="{c["ink"]}">Benchmarks</text>'
    )
    out.append(
        f'<text x="0" y="52" font-family="{SANS}" font-size="13" fill="{c["muted"]}">'
        f'Apple M2 Pro · lower is better</text>'
    )
    lx = W - 300
    out.append(f'<rect x="{lx}" y="18" width="13" height="13" rx="3" fill="{c["coreml"]}"/>')
    out.append(
        f'<text x="{lx+20}" y="29" font-family="{SANS}" font-size="13" fill="{c["sub"]}">'
        f'native CoreML</text>'
    )
    lx2 = lx + 150
    out.append(f'<rect x="{lx2}" y="18" width="13" height="13" rx="3" fill="{c["pytorch"]}"/>')
    out.append(
        f'<text x="{lx2+20}" y="29" font-family="{SANS}" font-size="13" fill="{c["sub"]}">'
        f'PyTorch</text>'
    )

    y = TOP
    for p in PANELS:
        out.append(
            f'<text x="0" y="{y+18}" font-family="{SANS}" font-size="15" '
            f'font-weight="600" fill="{c["ink"]}">{esc(p["title"])}</text>'
        )
        y += PANEL_TITLE_H
        vmax = max(v for _, v, _ in p["rows"])
        # recessive baseline
        y0 = y - 4
        y1 = y + len(p["rows"]) * ROW_H - (ROW_H - BAR_H) // 2 - 2
        out.append(
            f'<line x1="{LM}" y1="{y0}" x2="{LM}" y2="{y1}" stroke="{c["baseline"]}" '
            f'stroke-width="1"/>'
        )
        for i, (label, val, grp) in enumerate(p["rows"]):
            ry = y + i * ROW_H
            by = ry + (ROW_H - BAR_H) // 2
            bw = max(6, val / vmax * BAR_AREA)
            fill = c[grp]
            out.append(
                f'<text x="{LM-12}" y="{by+BAR_H-4}" text-anchor="end" font-family="{SANS}" '
                f'font-size="13.5" fill="{c["sub"]}">{esc(label)}</text>'
            )
            out.append(
                f'<rect x="{LM}" y="{by}" width="{bw:.1f}" height="{BAR_H}" rx="4" fill="{fill}"/>'
            )
            weight = "700" if i == 0 else "500"
            out.append(
                f'<text x="{LM+bw+8:.1f}" y="{by+BAR_H-4}" font-family="{MONO}" '
                f'font-size="13.5" font-weight="{weight}" fill="{c["ink"]}">'
                f'{fmt(val, p["unit"])}</text>'
            )
        y += len(p["rows"]) * ROW_H + PANEL_GAP

    out.append(
        f'<text x="0" y="{h-8}" font-family="{SANS}" font-size="11.5" fill="{c["muted"]}">'
        f'Speed and memory on a 30 s clip; DER on the full AMI EN2002a meeting. '
        f'Reproduce: benchmarks/run.sh</text>'
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main():
    d = Path(__file__).resolve().parent.parent / "assets"
    d.mkdir(exist_ok=True)
    for mode in ("light", "dark"):
        (d / f"benchmarks-{mode}.svg").write_text(render(mode))
        print("wrote", d / f"benchmarks-{mode}.svg")


if __name__ == "__main__":
    main()
