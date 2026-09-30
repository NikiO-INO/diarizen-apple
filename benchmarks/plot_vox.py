#!/usr/bin/env python3
"""Render the VoxConverse many-speaker infographic (light + dark SVG).

    python benchmarks/plot_vox.py

Per-file DER on the 31 VoxConverse files with >= 12 speakers, scored with the
standard 0.25 s collar (validation/eval_corpus.py --dir build/vox --collar 0.25).
Shows that the port keeps DER low across 12-21 speakers and never caps the count.
"""
from pathlib import Path

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, 'SF Mono', SFMono-Regular, Menlo, Consolas, monospace"

# (id, ref_spk, hyp_spk, DER %)
DATA = [
    ("nitgx", 21, 23, 6.39), ("kdfqk", 20, 17, 7.58), ("vzuru", 19, 18, 2.58),
    ("jeymh", 18, 17, 14.57), ("vmaiq", 17, 14, 5.19), ("qajyo", 17, 15, 6.23),
    ("uqxlg", 15, 11, 3.25), ("ldnro", 15, 13, 3.02), ("lbfnx", 15, 15, 2.76),
    ("kajfh", 15, 14, 11.67), ("eqsta", 15, 9, 18.31), ("diysk", 15, 11, 19.40),
    ("cjfer", 15, 15, 6.56), ("byapz", 15, 12, 8.58), ("usqam", 14, 13, 5.34),
    ("qxana", 14, 12, 7.01), ("qeejz", 14, 14, 9.90), ("vncid", 13, 12, 4.50),
    ("pwnsw", 13, 8, 8.47), ("jbowg", 13, 9, 7.99), ("gtnjb", 13, 11, 2.97),
    ("aggyz", 13, 13, 1.51), ("zzyyo", 12, 10, 8.04), ("wewoz", 12, 11, 4.62),
    ("vtzqw", 12, 9, 10.10), ("mqxsf", 12, 12, 6.90), ("mkhie", 12, 12, 6.27),
    ("ibrnm", 12, 12, 5.45), ("heolf", 12, 12, 4.97), ("epdpg", 12, 11, 3.71),
    ("aorju", 12, 10, 7.27),
]
CORPUS = 7.24
AXIS_MAX = 20.0

THEME = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "sub": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "bar": "#2a78d6"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "sub": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a", "bar": "#3987e5"},
}

W = 920
LM, RM = 150, 46
BAR_AREA = W - LM - RM
ROW_H = 18
BAR_H = 12
TOP = 76


def render(mode):
    c = THEME[mode]
    rows = sorted(DATA, key=lambda r: (-r[1], r[3]))
    h = TOP + len(rows) * ROW_H + 26
    px = BAR_AREA / AXIS_MAX
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" '
         f'height="{h}" role="img" aria-label="VoxConverse per-file DER on 12 to 21 '
         f'speaker files; corpus 7.24%, no speaker cap">']
    o.append(f'<rect x="0" y="0" width="{W}" height="{h}" fill="{c["surface"]}"/>')
    o.append(f'<text x="0" y="24" font-family="{SANS}" font-size="18" font-weight="700" '
             f'fill="{c["ink"]}">VoxConverse: DER on 12-21 speaker files</text>')
    o.append(f'<text x="0" y="45" font-family="{SANS}" font-size="12.5" fill="{c["sub"]}">'
             f'in the wild, collar 0.25s · corpus DER 7.24% · the base model finds '
             f'8-23 speakers, never capped at 8</text>')
    # corpus reference line label (legend-ish)
    o.append(f'<text x="{W}" y="24" text-anchor="end" font-family="{SANS}" font-size="12.5" '
             f'fill="{c["muted"]}">dashed = corpus avg 7.24%</text>')
    col_hdr_y = TOP - 10
    o.append(f'<text x="8" y="{col_hdr_y}" font-family="{SANS}" font-size="10.5" '
             f'fill="{c["muted"]}">file</text>')
    o.append(f'<text x="{LM-10}" y="{col_hdr_y}" text-anchor="end" font-family="{SANS}" '
             f'font-size="10.5" fill="{c["muted"]}">spk</text>')
    plot_bottom = TOP + len(rows) * ROW_H - 4
    # gridlines
    t = 0
    while t <= AXIS_MAX:
        gx = LM + t * px
        o.append(f'<line x1="{gx:.1f}" y1="{TOP-4}" x2="{gx:.1f}" y2="{plot_bottom}" '
                 f'stroke="{c["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{gx:.1f}" y="{plot_bottom+16}" text-anchor="middle" '
                 f'font-family="{SANS}" font-size="10" fill="{c["muted"]}">{int(t)}%</text>')
        t += 5
    # corpus avg dashed line
    cx = LM + CORPUS * px
    o.append(f'<line x1="{cx:.1f}" y1="{TOP-4}" x2="{cx:.1f}" y2="{plot_bottom}" '
             f'stroke="{c["bar"]}" stroke-width="1.5" stroke-dasharray="3 3" opacity="0.8"/>')
    # rows
    for i, (fid, rspk, hspk, der) in enumerate(rows):
        ry = TOP + i * ROW_H
        o.append(f'<text x="8" y="{ry+ROW_H-5}" font-family="{MONO}" font-size="11" '
                 f'fill="{c["sub"]}">{fid}</text>')
        o.append(f'<text x="{LM-10}" y="{ry+ROW_H-5}" text-anchor="end" font-family="{MONO}" '
                 f'font-size="11" fill="{c["muted"]}">{rspk}</text>')
        by = ry + (ROW_H - BAR_H) // 2
        bw = max(2, der * px)
        o.append(f'<rect x="{LM}" y="{by}" width="{bw:.1f}" height="{BAR_H}" rx="3" '
                 f'fill="{c["bar"]}"/>')
        o.append(f'<text x="{LM+bw+5:.1f}" y="{by+BAR_H-2}" font-family="{MONO}" '
                 f'font-size="10" fill="{c["sub"]}">{der:.1f}</text>')
    return "\n".join(o) + "\n</svg>\n"


def main():
    d = Path(__file__).resolve().parent.parent / "assets"
    for mode in ("light", "dark"):
        (d / f"vox-{mode}.svg").write_text(render(mode))
        print("wrote", d / f"vox-{mode}.svg")


if __name__ == "__main__":
    main()
