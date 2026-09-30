#!/usr/bin/env python3
"""Render the VoxConverse many-speaker infographic (light + dark SVG).

    python benchmarks/plot_vox.py

Per-file DER on the 31 VoxConverse files with >= 12 speakers, scored with the
standard 0.25 s collar (validation/eval_corpus.py --dir build/vox --collar 0.25),
base and large. Shows the port keeps DER low across 12-21 speakers and never caps
the count.
"""
from pathlib import Path

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, 'SF Mono', SFMono-Regular, Menlo, Consolas, monospace"

# (id, ref_spk, base DER %, large DER %)
DATA = [
    ("nitgx", 21, 6.39, 5.99), ("kdfqk", 20, 7.58, 7.87), ("vzuru", 19, 2.58, 3.46),
    ("jeymh", 18, 14.57, 14.56), ("vmaiq", 17, 5.19, 4.59), ("qajyo", 17, 6.23, 5.45),
    ("uqxlg", 15, 3.25, 4.36), ("ldnro", 15, 3.02, 2.02), ("lbfnx", 15, 2.76, 2.64),
    ("kajfh", 15, 11.67, 8.33), ("eqsta", 15, 18.31, 17.11), ("diysk", 15, 19.40, 16.71),
    ("cjfer", 15, 6.56, 7.52), ("byapz", 15, 8.58, 7.49), ("usqam", 14, 5.34, 5.97),
    ("qxana", 14, 7.01, 3.47), ("qeejz", 14, 9.90, 8.67), ("vncid", 13, 4.50, 4.73),
    ("pwnsw", 13, 8.47, 7.69), ("jbowg", 13, 7.99, 7.84), ("gtnjb", 13, 2.97, 2.29),
    ("aggyz", 13, 1.51, 0.44), ("zzyyo", 12, 8.04, 7.49), ("wewoz", 12, 4.62, 4.51),
    ("vtzqw", 12, 10.10, 8.28), ("mqxsf", 12, 6.90, 5.66), ("mkhie", 12, 6.27, 4.72),
    ("ibrnm", 12, 5.45, 3.68), ("heolf", 12, 4.97, 4.19), ("epdpg", 12, 3.71, 2.84),
    ("aorju", 12, 7.27, 4.66),
]
BASE_CORPUS, LARGE_CORPUS = 7.24, 6.35
AXIS_MAX = 20.0

THEME = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "sub": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "base": "#2a78d6", "large": "#1baf7a"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "sub": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a", "base": "#3987e5", "large": "#199e70"},
}

W = 920
LM, RM = 150, 46
BAR_AREA = W - LM - RM
ROW_H = 21
BAR_H = 8
TOP = 80


def render(mode):
    c = THEME[mode]
    rows = sorted(DATA, key=lambda r: (-r[1], r[2]))
    h = TOP + len(rows) * ROW_H + 26
    px = BAR_AREA / AXIS_MAX
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" '
         f'height="{h}" role="img" aria-label="VoxConverse per-file DER, base vs large, '
         f'on 12 to 21 speaker files; corpus base 7.24%, large 6.35%, no cap">']
    o.append(f'<rect x="0" y="0" width="{W}" height="{h}" fill="{c["surface"]}"/>')
    o.append(f'<text x="0" y="24" font-family="{SANS}" font-size="18" font-weight="700" '
             f'fill="{c["ink"]}">VoxConverse: DER on 12-21 speaker files</text>')
    o.append(f'<text x="0" y="45" font-family="{SANS}" font-size="12.5" fill="{c["sub"]}">'
             f'in the wild, collar 0.25s · corpus DER base 7.24%, large 6.35% · finds '
             f'8-23 speakers, never capped at 8</text>')
    # legend
    lx = W - 232
    o.append(f'<rect x="{lx}" y="14" width="12" height="12" rx="3" fill="{c["base"]}"/>')
    o.append(f'<text x="{lx+18}" y="24" font-family="{SANS}" font-size="12.5" fill="{c["sub"]}">base</text>')
    lx2 = lx + 74
    o.append(f'<rect x="{lx2}" y="14" width="12" height="12" rx="3" fill="{c["large"]}"/>')
    o.append(f'<text x="{lx2+18}" y="24" font-family="{SANS}" font-size="12.5" fill="{c["sub"]}">large-v2</text>')
    # column headers
    o.append(f'<text x="8" y="{TOP-10}" font-family="{SANS}" font-size="10.5" fill="{c["muted"]}">file</text>')
    o.append(f'<text x="{LM-10}" y="{TOP-10}" text-anchor="end" font-family="{SANS}" font-size="10.5" fill="{c["muted"]}">spk</text>')
    plot_bottom = TOP + len(rows) * ROW_H - 3
    # gridlines
    t = 0
    while t <= AXIS_MAX:
        gx = LM + t * px
        o.append(f'<line x1="{gx:.1f}" y1="{TOP-4}" x2="{gx:.1f}" y2="{plot_bottom}" '
                 f'stroke="{c["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{gx:.1f}" y="{plot_bottom+16}" text-anchor="middle" '
                 f'font-family="{SANS}" font-size="10" fill="{c["muted"]}">{int(t)}%</text>')
        t += 5
    # corpus average dashed lines, labeled at top
    for val, key, lab in ((BASE_CORPUS, "base", f"base {BASE_CORPUS}"), (LARGE_CORPUS, "large", f"large {LARGE_CORPUS}")):
        rx = LM + val * px
        o.append(f'<line x1="{rx:.1f}" y1="{TOP-4}" x2="{rx:.1f}" y2="{plot_bottom}" '
                 f'stroke="{c[key]}" stroke-width="1.5" stroke-dasharray="3 3" opacity="0.85"/>')
    # rows
    for i, (fid, rspk, b, l) in enumerate(rows):
        ry = TOP + i * ROW_H
        o.append(f'<text x="8" y="{ry+14}" font-family="{MONO}" font-size="10.5" fill="{c["sub"]}">{fid}</text>')
        o.append(f'<text x="{LM-10}" y="{ry+14}" text-anchor="end" font-family="{MONO}" font-size="10.5" fill="{c["muted"]}">{rspk}</text>')
        for val, key, dy in ((b, "base", 2), (l, "large", 2 + BAR_H + 1)):
            bw = max(2, val * px)
            o.append(f'<rect x="{LM}" y="{ry+dy}" width="{bw:.1f}" height="{BAR_H}" rx="2.5" fill="{c[key]}"/>')
    # x-axis title
    o.append(f'<text x="{LM+BAR_AREA/2}" y="{h-8}" text-anchor="middle" font-family="{SANS}" '
             f'font-size="11" fill="{c["muted"]}">diarization error rate (lower is better) · sorted by speaker count</text>')
    return "\n".join(o) + "\n</svg>\n"


def main():
    d = Path(__file__).resolve().parent.parent / "assets"
    for mode in ("light", "dark"):
        (d / f"vox-{mode}.svg").write_text(render(mode))
        print("wrote", d / f"vox-{mode}.svg")


if __name__ == "__main__":
    main()
