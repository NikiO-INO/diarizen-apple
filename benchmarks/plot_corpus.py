#!/usr/bin/env python3
"""Render the AMI-SDM corpus-accuracy dumbbell (light + dark SVG) from the
per-meeting DER numbers produced by validation/eval_corpus.py.

    python benchmarks/plot_corpus.py

Each meeting is one row: a dot for base and a dot for large joined by a line, so
the base -> large improvement reads at a glance. Dashed verticals mark the
duration-weighted corpus averages.
"""
from pathlib import Path

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, 'SF Mono', SFMono-Regular, Menlo, Consolas, monospace"

# (meeting, base DER %, large DER %)
ROWS = [
    ("ES2004a", 20.44, 17.39), ("ES2004b", 13.23, 11.84), ("ES2004c", 11.66, 10.40),
    ("ES2004d", 16.05, 14.59), ("IS1009a", 18.37, 17.94), ("IS1009b", 13.18, 8.00),
    ("IS1009c", 10.55, 8.41), ("IS1009d", 14.92, 13.15), ("TS3003a", 21.17, 18.80),
    ("TS3003b", 9.12, 9.00), ("TS3003c", 13.52, 12.72), ("TS3003d", 19.19, 18.47),
    ("EN2002a", 21.16, 17.50), ("EN2002b", 17.46, 14.30), ("EN2002c", 14.06, 13.12),
    ("EN2002d", 21.88, 18.37),
]
BASE_CORPUS, LARGE_CORPUS = 15.79, 13.76
AXIS_MIN, AXIS_MAX = 0.0, 24.0

THEME = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "sub": "#52514e", "muted": "#898781",
              "grid": "#ecebe4", "link": "#c9c8c0", "base": "#2a78d6", "large": "#1baf7a"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "sub": "#c3c2b7", "muted": "#898781",
             "grid": "#262625", "link": "#3a3a37", "base": "#3987e5", "large": "#28c286"},
}

W = 900
LM, RM = 96, 30
AREA = W - LM - RM
ROW_H = 27
TOP = 82
R = 5.5


def render(mode):
    c = THEME[mode]
    rows = sorted(ROWS, key=lambda r: r[1])
    h = TOP + len(rows) * ROW_H + 28
    px = AREA / (AXIS_MAX - AXIS_MIN)

    def X(v):
        return LM + (v - AXIS_MIN) * px

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" '
         f'height="{h}" role="img" aria-label="AMI-SDM per-meeting DER, base vs large; '
         f'corpus base 15.79%, large 13.76%">']
    o.append(f'<rect x="0" y="0" width="{W}" height="{h}" fill="{c["surface"]}"/>')
    o.append(f'<text x="0" y="26" font-family="{SANS}" font-size="19" font-weight="700" '
             f'fill="{c["ink"]}">AMI-SDM accuracy, per meeting</text>')
    o.append(f'<text x="0" y="48" font-family="{SANS}" font-size="13" fill="{c["sub"]}">'
             f'diarization error rate, lower is better · 16-meeting corpus average: '
             f'base 15.79%, large 13.76% (DiariZen published ~15.8%)</text>')
    # legend
    lx = W - 250
    o.append(f'<circle cx="{lx+6}" cy="20" r="6" fill="{c["base"]}"/>')
    o.append(f'<text x="{lx+18}" y="24" font-family="{SANS}" font-size="13" fill="{c["sub"]}">base</text>')
    lx2 = lx + 78
    o.append(f'<circle cx="{lx2+6}" cy="20" r="6" fill="{c["large"]}"/>')
    o.append(f'<text x="{lx2+18}" y="24" font-family="{SANS}" font-size="13" fill="{c["sub"]}">large-v2</text>')

    plot_bottom = TOP + len(rows) * ROW_H - 8
    # gridlines + axis labels
    t = 0
    while t <= AXIS_MAX:
        gx = X(t)
        o.append(f'<line x1="{gx:.1f}" y1="{TOP-8}" x2="{gx:.1f}" y2="{plot_bottom}" '
                 f'stroke="{c["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{gx:.1f}" y="{plot_bottom+18}" text-anchor="middle" '
                 f'font-family="{SANS}" font-size="10.5" fill="{c["muted"]}">{int(t)}%</text>')
        t += 4
    # corpus average dashed verticals, labeled at the top
    for val, key, lab in ((BASE_CORPUS, "base", "15.79"), (LARGE_CORPUS, "large", "13.76")):
        vx = X(val)
        o.append(f'<line x1="{vx:.1f}" y1="{TOP-8}" x2="{vx:.1f}" y2="{plot_bottom}" '
                 f'stroke="{c[key]}" stroke-width="1.5" stroke-dasharray="4 3" opacity="0.7"/>')
        o.append(f'<text x="{vx:.1f}" y="{TOP-14}" text-anchor="middle" font-family="{MONO}" '
                 f'font-size="10.5" font-weight="700" fill="{c[key]}">{lab}</text>')
    # rows: dumbbell
    for i, (m, b, l) in enumerate(rows):
        cy = TOP + i * ROW_H + ROW_H // 2
        o.append(f'<text x="{LM-12}" y="{cy+4}" text-anchor="end" font-family="{MONO}" '
                 f'font-size="11" fill="{c["sub"]}">{m}</text>')
        xb, xl = X(b), X(l)
        o.append(f'<line x1="{xl:.1f}" y1="{cy}" x2="{xb:.1f}" y2="{cy}" stroke="{c["link"]}" '
                 f'stroke-width="3" stroke-linecap="round"/>')
        o.append(f'<circle cx="{xl:.1f}" cy="{cy}" r="{R}" fill="{c["large"]}"/>')
        o.append(f'<circle cx="{xb:.1f}" cy="{cy}" r="{R}" fill="{c["base"]}"/>')
    return "\n".join(o) + "\n</svg>\n"


def main():
    d = Path(__file__).resolve().parent.parent / "assets"
    for mode in ("light", "dark"):
        (d / f"corpus-{mode}.svg").write_text(render(mode))
        print("wrote", d / f"corpus-{mode}.svg")


if __name__ == "__main__":
    main()
