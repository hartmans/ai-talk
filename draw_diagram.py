"""Draw the 'How LLMs Predict the Next Token' diagram for the talk slide.

Regenerates llm-next-token-prediction.png (the figure on the 'How LLMs
Predict the Next Token' slide). White background, two side-by-side panels
showing next-token prediction before and after the '=' token, 16:9.

Usage:
    python3 draw_diagram.py [output.png]

Requires Pillow. Uses the DejaVu Sans fonts when present, otherwise falls
back to Pillow's built-in default font.
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 1600, 900
BLACK = (17, 17, 17)
DIVIDER = (204, 204, 204)
BAR_DARK = (26, 26, 26)
GRAYS = [(77, 77, 77), (122, 122, 122), (163, 163, 163), (196, 196, 196)]
# Shared px-per-unit for both bar charts so the 0.92 bar reads longer than 0.65.
SCALE = 210.0 / 0.92


def load_font(bold, size):
    name = "DejaVuSans-%s.ttf" % ("Bold" if bold else "Regular")
    for base in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu"):
        path = os.path.join(base, name)
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F_SUB = load_font(True, 34)
F_TOKEN = load_font(True, 38)
F_LLM = load_font(True, 42)
F_LEG = load_font(False, 32)
F_BAR = load_font(False, 34)
F_CAP = load_font(False, 38)

img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)


def txy(x, y, s, font, anchor="mm"):
    d.text((x, y), s, font=font, fill=BLACK, anchor=anchor)


def box(x0, y0, x1, y1, r=12, w=3):
    d.rounded_rectangle((x0, y0, x1, y1), radius=r, outline=BLACK, width=w, fill=(255, 255, 255))


def arrow_h(x0, x1, y, head=13):
    d.line((x0, y, x1 - head, y), fill=BLACK, width=3)
    d.polygon([(x1, y), (x1 - head, y - head * 0.55), (x1 - head, y + head * 0.55)], fill=BLACK)


def arrow_v(x, y0, y1, head=13):
    d.line((x, y0, x, y1 - head), fill=BLACK, width=3)
    d.polygon([(x, y1), (x - head * 0.55, y1 - head), (x + head * 0.55, y1 - head)], fill=BLACK)


def bar(x, y, length, color, h=34):
    d.rounded_rectangle((x, y, x + length, y + h), radius=h // 2, fill=color)


def token_row(cx, y0, labels, trailing=False):
    """Row of 74px token boxes with arrows; cx = group center x."""
    bw, gap = 74, 44
    total = 3 * bw + 2 * gap
    x0 = cx - total // 2
    for i, lab in enumerate(labels):
        bx0 = x0 + i * (bw + gap)
        box(bx0, y0, bx0 + bw, y0 + bw)
        txy(bx0 + bw // 2, y0 + bw // 2, lab, F_TOKEN)
        if i < 2:
            arrow_h(bx0 + bw + 6, bx0 + bw + gap - 6, y0 + bw // 2, head=11)
    if trailing:
        last_x1 = x0 + 2 * (bw + gap) + bw
        arrow_h(last_x1 + 8, last_x1 + 92, y0 + bw // 2)


# ---- divider ---------------------------------------------------------------
d.line((800, 110, 800, 790), fill=DIVIDER, width=2)

# ---- left panel: Step 1 ----------------------------------------------------
txy(60, 140, "Step 1: Before \u201c=\u201d", F_SUB, anchor="lm")
token_row(280, 195, ["2", "+", "2"])
arrow_v(280, 273, 325)
box(190, 330, 370, 434, r=14)
txy(280, 382, "LLM", F_LLM)
txy(430, 340, "Next-token", F_LEG, anchor="lm")
txy(430, 376, "probabilities", F_LEG, anchor="lm")
left_bars = [(0.65, "0.65", BAR_DARK), (0.12, "0.12", GRAYS[0]), (0.08, "0.08", GRAYS[1]),
             (0.06, "0.06", GRAYS[2]), (None, "<newline>", GRAYS[3])]
for i, (val, lab, col) in enumerate(left_bars):
    length = 20 if val is None else max(int(round(SCALE * val)), 6)
    y = 390 + i * 56
    bar(430, y, length, col)
    txy(430 + length + 12, y + 17, lab, F_BAR, anchor="lm")
for i, line in enumerate(["Before seeing the \u201c=\u201d sign, the",
                          "LLM predicts what token most",
                          "likely comes next."]):
    txy(60, 672 + i * 38, line, F_CAP, anchor="lm")

# ---- right panel: Step 2 ---------------------------------------------------
txy(860, 140, "Step 2: After \u201c=\u201d", F_SUB, anchor="lm")
token_row(1080, 195, ["2", "+", "="], trailing=True)
arrow_v(1080, 273, 325)
box(990, 330, 1170, 434, r=14)
txy(1080, 382, "LLM", F_LLM)
arrow_h(1178, 1256, 407)
txy(1270, 340, "Next-token", F_LEG, anchor="lm")
txy(1270, 376, "probabilities", F_LEG, anchor="lm")
right_bars = [(0.92, "0.92", BAR_DARK), (0.04, "0.04", GRAYS[0]),
              (0.02, "0.02", GRAYS[1]), (0.01, "0.01", GRAYS[2])]
for i, (val, lab, col) in enumerate(right_bars):
    length = max(int(round(SCALE * val)), 3)
    y = 390 + i * 56
    bar(1270, y, length, col)
    txy(1270 + length + 12, y + 17, lab, F_BAR, anchor="lm")
arrow_v(1080, 438, 542)
token_row(1080, 547, ["2", "+", "4"])
for i, line in enumerate(["Once the \u201c=\u201d token is included,",
                          "LLM predicts the result with",
                          "high probability."]):
    txy(860, 672 + i * 38, line, F_CAP, anchor="lm")

out = sys.argv[1] if len(sys.argv) > 1 else "llm-next-token-prediction.png"
img.save(out)
print("saved", out, img.size)
