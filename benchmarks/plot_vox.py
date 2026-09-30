#!/usr/bin/env python3
"""Render the VoxConverse many-speaker scatter (light + dark SVG).

    python benchmarks/plot_vox.py

DER vs number of speakers for the 31 VoxConverse files with >= 12 speakers,
scored with the standard 0.25 s collar (validation/eval_corpus.py --collar 0.25),
base and large. The point is that DER does not climb as the speaker count grows:
the port handles 12-21 speakers and never caps the count.
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
X_MIN, X_MAX = 11.3, 21.7
Y_MIN, Y_MAX = 0.0, 20.0

THEME = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "sub": "#52514e", "muted": "#898781",
              "grid": "#ecebe4", "axis": "#c3c2b7", "base": "#2a78d6", "large": "#1baf7a"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "sub": "#c3c2b7", "muted": "#898781",
             "grid": "#262625", "axis": "#383835", "base": "#3987e5", "large": "#28c286"},
}

W, H = 900, 470
L, RGT, TOPP, BOT = 52, 150, 84, 52
PW = W - L - RGT
PH = H - TOPP - BOT
R = 5.5


def render(mode):
    c = THEME[mode]

    def X(v):
        return L + (v - X_MIN) / (X_MAX - X_MIN) * PW

    def Y(v):
        return TOPP + (1 - (v - Y_MIN) / (Y_MAX - Y_MIN)) * PH

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
         f'height="{H}" role="img" aria-label="VoxConverse DER vs number of speakers, '
         f'base and large; DER stays low across 12 to 21 speakers, no cap">']
    o.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="{c["surface"]}"/>')
    o.append(f'<text x="0" y="26" font-family="{SANS}" font-size="19" font-weight="700" '
             f'fill="{c["ink"]}">VoxConverse: accuracy vs number of speakers</text>')
    o.append(f'<text x="0" y="48" font-family="{SANS}" font-size="13" fill="{c["sub"]}">'
             f'each dot is one in-the-wild file (collar 0.25s) · DER stays low as the '
             f'speaker count grows, and the count is never capped at 8</text>')
    # legend
    lx = W - 250
    o.append(f'<circle cx="{lx+6}" cy="20" r="6" fill="{c["base"]}"/>')
    o.append(f'<text x="{lx+18}" y="24" font-family="{SANS}" font-size="13" fill="{c["sub"]}">base</text>')
    lx2 = lx + 78
    o.append(f'<circle cx="{lx2+6}" cy="20" r="6" fill="{c["large"]}"/>')
    o.append(f'<text x="{lx2+18}" y="24" font-family="{SANS}" font-size="13" fill="{c["sub"]}">large-v2</text>')

    # y gridlines + labels
    yv = 0
    while yv <= Y_MAX:
        gy = Y(yv)
        o.append(f'<line x1="{L}" y1="{gy:.1f}" x2="{L+PW}" y2="{gy:.1f}" stroke="{c["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{L-8}" y="{gy+4:.1f}" text-anchor="end" font-family="{SANS}" '
                 f'font-size="10.5" fill="{c["muted"]}">{int(yv)}%</text>')
        yv += 5
    # x ticks (speaker counts)
    for sp in range(12, 22):
        gx = X(sp)
        o.append(f'<text x="{gx:.1f}" y="{TOPP+PH+18}" text-anchor="middle" font-family="{SANS}" '
                 f'font-size="10.5" fill="{c["muted"]}">{sp}</text>')
    o.append(f'<text x="{L+PW/2}" y="{H-8}" text-anchor="middle" font-family="{SANS}" '
             f'font-size="12" fill="{c["sub"]}">number of speakers in the file</text>')
    o.append(f'<text x="{L-38}" y="{TOPP-14}" font-family="{SANS}" font-size="11" '
             f'fill="{c["muted"]}">DER</text>')

    # corpus average horizontal dashed lines
    for val, key, lab in ((BASE_CORPUS, "base", "base avg 7.24%"), (LARGE_CORPUS, "large", "large avg 6.35%")):
        gy = Y(val)
        o.append(f'<line x1="{L}" y1="{gy:.1f}" x2="{L+PW}" y2="{gy:.1f}" stroke="{c[key]}" '
                 f'stroke-width="1.5" stroke-dasharray="4 3" opacity="0.7"/>')
        o.append(f'<text x="{L+PW+8}" y="{gy+4:.1f}" font-family="{MONO}" font-size="10.5" '
                 f'fill="{c[key]}">{lab}</text>')

    # points (deterministic horizontal jitter per file so same-count files separate)
    for i, (fid, spk, b, l) in enumerate(DATA):
        jit = (((i * 37) % 11) / 11.0 - 0.5) * 0.58
        cx = X(spk + jit)
        o.append(f'<circle cx="{cx:.1f}" cy="{Y(b):.1f}" r="{R}" fill="{c["base"]}" fill-opacity="0.72"/>')
        o.append(f'<circle cx="{cx:.1f}" cy="{Y(l):.1f}" r="{R}" fill="{c["large"]}" fill-opacity="0.72"/>')
    return "\n".join(o) + "\n</svg>\n"


def main():
    d = Path(__file__).resolve().parent.parent / "assets"
    for mode in ("light", "dark"):
        (d / f"vox-{mode}.svg").write_text(render(mode))
        print("wrote", d / f"vox-{mode}.svg")


if __name__ == "__main__":
    main()
