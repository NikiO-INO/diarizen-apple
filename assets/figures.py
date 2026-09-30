#!/usr/bin/env python3
"""Generate the README stat tiles and the example-output timeline (light+dark)."""
from pathlib import Path

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, 'SF Mono', SFMono-Regular, Menlo, Consolas, monospace"
ASSETS = Path(__file__).resolve().parent


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------- stat tiles ----------------
TILES = [
    ("0.037", "real-time factor", "~16x vs PyTorch CPU"),
    ("~300 MB", "peak memory", "no Python at inference"),
    ("17.50%", "DER on AMI EN2002a", "large model; 21.16% base"),
    ("∞", "speakers", "no cap"),
]
TILE_THEME = {
    "light": {"card": "#f6f8fa", "border": "#d0d7de", "num": "#2a78d6",
              "l1": "#1f2328", "l2": "#6e7781"},
    "dark": {"card": "#161b22", "border": "#30363d", "num": "#3987e5",
             "l1": "#e6edf3", "l2": "#8b949e"},
}


def tiles(mode):
    c = TILE_THEME[mode]
    W, H, gap = 920, 100, 15
    tw = (W - gap * (len(TILES) - 1)) // len(TILES)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
           f'height="{H}" role="img" aria-label="Highlights: real-time factor 0.037, '
           f'about 300 MB peak memory, 17.50% DER, no speaker cap">']
    for i, (num, l1, l2) in enumerate(TILES):
        x = i * (tw + gap)
        cx = x + tw // 2
        out.append(f'<rect x="{x}" y="0" width="{tw}" height="{H}" rx="14" '
                   f'fill="{c["card"]}" stroke="{c["border"]}" stroke-width="1"/>')
        out.append(f'<text x="{cx}" y="48" text-anchor="middle" font-family="{MONO}" '
                   f'font-size="38" font-weight="700" fill="{c["num"]}">{esc(num)}</text>')
        out.append(f'<text x="{cx}" y="74" text-anchor="middle" font-family="{SANS}" '
                   f'font-size="13.5" font-weight="600" fill="{c["l1"]}">{esc(l1)}</text>')
        out.append(f'<text x="{cx}" y="90" text-anchor="middle" font-family="{SANS}" '
                   f'font-size="12" fill="{c["l2"]}">{esc(l2)}</text>')
    out.append("</svg>\n")
    return "\n".join(out)


# ---------------- example timeline ----------------
SPEAKERS = [
    ("speaker_0", "s0", [(0.005, 2.625), (5.745, 6.405), (7.985, 13.585),
                         (18.905, 19.485), (23.425, 30.345)]),
    ("speaker_1", "s1", [(0.745, 13.565), (17.785, 18.225), (20.305, 22.245)]),
    ("speaker_2", "s2", [(0.005, 0.785), (13.665, 18.365), (19.585, 19.965),
                         (23.265, 23.465)]),
]
TMAX = 30.4
TL_THEME = {
    "light": {"title": "#1f2328", "sub": "#57606a", "lane": "#eef1f4",
              "axis": "#6e7781", "tick": "#d0d7de",
              "s0": "#2a78d6", "s1": "#eb6834", "s2": "#1baf7a"},
    "dark": {"title": "#e6edf3", "sub": "#8b949e", "lane": "#161b22",
             "axis": "#8b949e", "tick": "#30363d",
             "s0": "#3987e5", "s1": "#d95926", "s2": "#199e70"},
}


def timeline(mode):
    c = TL_THEME[mode]
    W = 920
    X0, RPAD = 108, 24
    TW = W - X0 - RPAD
    lane_h, lane_gap = 24, 12
    y_top = 44
    px = TW / TMAX
    n = len(SPEAKERS)
    axis_y = y_top + n * (lane_h + lane_gap) + 4
    H = axis_y + 30
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
           f'height="{H}" role="img" aria-label="Example output as a timeline: three '
           f'speakers over 30 seconds, overlaps aligned vertically">']
    out.append(f'<text x="0" y="20" font-family="{SANS}" font-size="15" font-weight="700" '
               f'fill="{c["title"]}">Example output as a timeline</text>')
    out.append(f'<text x="0" y="38" font-family="{SANS}" font-size="12.5" fill="{c["sub"]}">'
               f'30 s, 3 speakers · overlapping speech lines up vertically</text>')
    # lanes
    for i, (label, key, segs) in enumerate(SPEAKERS):
        ly = y_top + i * (lane_h + lane_gap)
        out.append(f'<rect x="{X0}" y="{ly}" width="{TW}" height="{lane_h}" rx="5" '
                   f'fill="{c["lane"]}"/>')
        out.append(f'<text x="{X0-12}" y="{ly+lane_h-7}" text-anchor="end" '
                   f'font-family="{MONO}" font-size="12.5" fill="{c["sub"]}">{label}</text>')
        col = c[key]  # s0 / s1 / s2
        for s, e in segs:
            bx = X0 + s * px
            bw = max(3, (e - s) * px)
            out.append(f'<rect x="{bx:.1f}" y="{ly}" width="{bw:.1f}" height="{lane_h}" '
                       f'rx="4" fill="{col}"/>')
    # axis ticks
    t = 0
    while t <= 30:
        tx = X0 + t * px
        out.append(f'<line x1="{tx:.1f}" y1="{y_top}" x2="{tx:.1f}" y2="{axis_y}" '
                   f'stroke="{c["tick"]}" stroke-width="1"/>')
        out.append(f'<text x="{tx:.1f}" y="{axis_y+16}" text-anchor="middle" '
                   f'font-family="{SANS}" font-size="11.5" fill="{c["axis"]}">{t}s</text>')
        t += 5
    out.append("</svg>\n")
    return "\n".join(out)


def main():
    for mode in ("light", "dark"):
        (ASSETS / f"highlights-{mode}.svg").write_text(tiles(mode))
        (ASSETS / f"timeline-{mode}.svg").write_text(timeline(mode))
        print("wrote", mode)


if __name__ == "__main__":
    main()
