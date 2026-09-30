#!/usr/bin/env python3
"""Generate the pipeline diagram (light + dark SVG) for the README."""
from pathlib import Path

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, 'SF Mono', SFMono-Regular, Menlo, Consolas, monospace"

STAGES = [
    ("Segmentation", "WavLM + Conformer, powerset head", [("CoreML · GPU", "cm")], "waveform"),
    ("Aggregation", "slide the window, decode powerset, median filter", [("Swift · CPU", "sw")], "per-frame speaker scores"),
    ("Speaker embeddings", "Kaldi filterbank + WeSpeaker ResNet34", [("Swift fbank", "sw"), ("CoreML ResNet", "cm")], "active speaker regions"),
    ("Clustering", "VBx: PLDA + Bayesian HMM · no speaker cap", [("Swift · CPU", "sw")], "one vector / speaker / window"),
    ("Reconstruction", "stitch window labels + clusters into turns", [("Swift · CPU", "sw")], "speaker labels"),
]

THEME = {
    "light": {
        "card": "#f6f8fa", "border": "#d0d7de", "title": "#1f2328", "sub": "#57606a",
        "flow": "#8c959f", "label": "#6e7781", "cm": "#2a78d6", "sw": "#6b7280",
        "pill": "#ffffff", "pill_border": "#afb8c1", "pill_txt": "#1f2328",
    },
    "dark": {
        "card": "#161b22", "border": "#30363d", "title": "#e6edf3", "sub": "#8b949e",
        "flow": "#6e7681", "label": "#8b949e", "cm": "#3987e5", "sw": "#9aa4b2",
        "pill": "#0d1117", "pill_border": "#3d444d", "pill_txt": "#e6edf3",
    },
}

W = 780
CX = 390          # center (arrows, pills)
CARD_X, CARD_W = 44, 692
CARD_H = 60
CONN = 42         # connector height between cards
PILL_W, PILL_H = 236, 38


def badge_w(text):
    return int(len(text) * 7.4) + 28


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def arrow(x, y0, y1, c, label):
    """Vertical arrow from y0 to y1 with an arrowhead, plus a flow label."""
    out = [
        f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y1-7}" stroke="{c["flow"]}" stroke-width="2"/>',
        f'<polygon points="{x-5},{y1-8} {x+5},{y1-8} {x},{y1}" fill="{c["flow"]}"/>',
    ]
    if label:
        my = (y0 + y1) / 2 + 4
        out.append(
            f'<text x="{x+14}" y="{my}" font-family="{SANS}" font-size="12" '
            f'font-style="italic" fill="{c["label"]}">{esc(label)}</text>'
        )
    return out


def pill(cx, y, text, c):
    x = cx - PILL_W // 2
    return [
        f'<rect x="{x}" y="{y}" width="{PILL_W}" height="{PILL_H}" rx="{PILL_H//2}" '
        f'fill="{c["pill"]}" stroke="{c["pill_border"]}" stroke-width="1.5"/>',
        f'<text x="{cx}" y="{y+PILL_H//2+5}" text-anchor="middle" font-family="{MONO}" '
        f'font-size="14" font-weight="600" fill="{c["pill_txt"]}">{esc(text)}</text>',
    ]


def card(y, title, sub, badges, c):
    out = [
        f'<rect x="{CARD_X}" y="{y}" width="{CARD_W}" height="{CARD_H}" rx="12" '
        f'fill="{c["card"]}" stroke="{c["border"]}" stroke-width="1"/>',
        f'<text x="{CARD_X+22}" y="{y+27}" font-family="{SANS}" font-size="16.5" '
        f'font-weight="700" fill="{c["title"]}">{esc(title)}</text>',
        f'<text x="{CARD_X+22}" y="{y+47}" font-family="{SANS}" font-size="13" '
        f'fill="{c["sub"]}">{esc(sub)}</text>',
    ]
    # badges, right-aligned in the card's top band
    bx = CARD_X + CARD_W - 18
    for text, kind in reversed(badges):
        bw = badge_w(text)
        bx -= bw
        col = c[kind]
        out.append(
            f'<rect x="{bx}" y="{y+13}" width="{bw}" height="22" rx="11" fill="none" '
            f'stroke="{col}" stroke-width="1.5"/>'
        )
        out.append(
            f'<circle cx="{bx+13}" cy="{y+24}" r="3.5" fill="{col}"/>'
        )
        out.append(
            f'<text x="{bx+22}" y="{y+28}" font-family="{SANS}" font-size="12" '
            f'font-weight="600" fill="{col}">{esc(text)}</text>'
        )
        bx -= 8
    return out


def render(mode):
    c = THEME[mode]
    parts = []
    y = 0
    parts += pill(CX, y, "audio · 16 kHz mono", c)
    y += PILL_H
    for title, sub, badges, flow in STAGES:
        parts += arrow(CX, y, y + CONN, c, flow)
        y += CONN
        parts += card(y, title, sub, badges, c)
        y += CARD_H
    parts += arrow(CX, y, y + CONN, c, "diarized turns")
    y += CONN
    parts += pill(CX, y, "RTTM", c)
    y += PILL_H + 22
    # legend
    parts.append(f'<circle cx="{CARD_X+6}" cy="{y-4}" r="5" fill="{c["cm"]}"/>')
    parts.append(
        f'<text x="{CARD_X+18}" y="{y}" font-family="{SANS}" font-size="12.5" '
        f'fill="{c["sub"]}">CoreML (neural, GPU)</text>'
    )
    parts.append(f'<circle cx="{CARD_X+186}" cy="{y-4}" r="5" fill="{c["sw"]}"/>')
    parts.append(
        f'<text x="{CARD_X+198}" y="{y}" font-family="{SANS}" font-size="12.5" '
        f'fill="{c["sub"]}">native Swift (CPU)</text>'
    )
    h = y + 12
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" '
        f'height="{h}" role="img" aria-label="diarizen-apple pipeline: audio to RTTM, '
        f'neural stages in CoreML on the GPU, the rest in native Swift on the CPU">'
    )
    return head + "\n" + "\n".join(parts) + "\n</svg>\n"


def main():
    d = Path(__file__).resolve().parent
    for mode in ("light", "dark"):
        (d / f"pipeline-{mode}.svg").write_text(render(mode))
        print("wrote", d / f"pipeline-{mode}.svg")


if __name__ == "__main__":
    main()
