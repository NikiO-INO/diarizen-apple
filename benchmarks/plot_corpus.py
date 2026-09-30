#!/usr/bin/env python3
"""Render the AMI-SDM corpus-accuracy infographic (light + dark SVG) from the
per-meeting DER numbers produced by validation/eval_corpus.py.

    python benchmarks/plot_corpus.py

base = eval_corpus.py --tag swift ; large = --tag large. Corpus averages are the
duration-weighted numbers over all 16 meetings.
"""
from pathlib import Path

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, 'SF Mono', SFMono-Regular, Menlo, Consolas, monospace"

# (meeting, base DER %, large DER %) in AMI-SDM test-set session order.
ROWS = [
    ("ES2004a", 20.44, 17.39), ("ES2004b", 13.23, 11.84),
    ("ES2004c", 11.66, 10.40), ("ES2004d", 16.05, 14.59),
    ("IS1009a", 18.37, 17.94), ("IS1009b", 13.18, 8.00),
    ("IS1009c", 10.55, 8.41), ("IS1009d", 14.92, 13.15),
    ("TS3003a", 21.17, 18.80), ("TS3003b", 9.12, 9.00),
    ("TS3003c", 13.52, 12.72), ("TS3003d", 19.19, 18.47),
    ("EN2002a", 21.16, 17.50), ("EN2002b", 17.46, 14.30),
    ("EN2002c", 14.06, 13.12), ("EN2002d", 21.88, 18.37),
]
BASE_CORPUS, LARGE_CORPUS = 15.79, 13.76
AXIS_MAX = 24.0

THEME = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "sub": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "base": "#2a78d6", "large": "#1baf7a"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "sub": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a", "base": "#3987e5", "large": "#199e70"},
}

W = 920
LM, RM = 84, 54
BAR_AREA = W - LM - RM
ROW_H = 30
BAR_H = 11
TOP = 78


def render(mode):
    c = THEME[mode]
    h = TOP + len(ROWS) * ROW_H + 26
    px = BAR_AREA / AXIS_MAX
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" '
         f'height="{h}" role="img" aria-label="AMI-SDM per-meeting DER, base vs large; '
         f'corpus base 15.79%, large 13.76%">']
    o.append(f'<rect x="0" y="0" width="{W}" height="{h}" fill="{c["surface"]}"/>')
    # header
    o.append(f'<text x="0" y="24" font-family="{SANS}" font-size="18" font-weight="700" '
             f'fill="{c["ink"]}">AMI-SDM test set: per-meeting DER</text>')
    o.append(f'<text x="0" y="46" font-family="{SANS}" font-size="13" fill="{c["sub"]}">'
             f'lower is better · corpus average base 15.79%, large 13.76% '
             f'(DiariZen published ~15.8%)</text>')
    # legend
    lx = W - 232
    o.append(f'<rect x="{lx}" y="12" width="13" height="13" rx="3" fill="{c["base"]}"/>')
    o.append(f'<text x="{lx+20}" y="23" font-family="{SANS}" font-size="13" '
             f'fill="{c["sub"]}">base</text>')
    lx2 = lx + 78
    o.append(f'<rect x="{lx2}" y="12" width="13" height="13" rx="3" fill="{c["large"]}"/>')
    o.append(f'<text x="{lx2+20}" y="23" font-family="{SANS}" font-size="13" '
             f'fill="{c["sub"]}">large-v2</text>')
    # x gridlines every 5%
    plot_bottom = TOP + len(ROWS) * ROW_H - 6
    t = 0
    while t <= AXIS_MAX:
        gx = LM + t * px
        o.append(f'<line x1="{gx:.1f}" y1="{TOP-6}" x2="{gx:.1f}" y2="{plot_bottom}" '
                 f'stroke="{c["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{gx:.1f}" y="{plot_bottom+16}" text-anchor="middle" '
                 f'font-family="{SANS}" font-size="10.5" fill="{c["muted"]}">{int(t)}%</text>')
        t += 5
    # corpus average reference lines (dashed)
    for val, key in ((BASE_CORPUS, "base"), (LARGE_CORPUS, "large")):
        rx = LM + val * px
        o.append(f'<line x1="{rx:.1f}" y1="{TOP-6}" x2="{rx:.1f}" y2="{plot_bottom}" '
                 f'stroke="{c[key]}" stroke-width="1.5" stroke-dasharray="3 3" opacity="0.8"/>')
    # rows
    for i, (m, b, l) in enumerate(ROWS):
        ry = TOP + i * ROW_H
        o.append(f'<text x="{LM-10}" y="{ry+ROW_H//2+2}" text-anchor="end" '
                 f'font-family="{MONO}" font-size="11.5" fill="{c["sub"]}">{m}</text>')
        for val, key, dy in ((b, "base", 2), (l, "large", 2 + BAR_H + 2)):
            by = ry + dy
            bw = max(2, val * px)
            o.append(f'<rect x="{LM}" y="{by}" width="{bw:.1f}" height="{BAR_H}" rx="3" '
                     f'fill="{c[key]}"/>')
            o.append(f'<text x="{LM+bw+5:.1f}" y="{by+BAR_H-1.5}" font-family="{MONO}" '
                     f'font-size="10.5" fill="{c["sub"]}">{val:.1f}</text>')
    return "\n".join(o) + "\n</svg>\n"


def main():
    d = Path(__file__).resolve().parent.parent / "assets"
    for mode in ("light", "dark"):
        (d / f"corpus-{mode}.svg").write_text(render(mode))
        print("wrote", d / f"corpus-{mode}.svg")


if __name__ == "__main__":
    main()
