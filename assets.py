"""Visuels : banniere poumon (SVG procedural) et radiographies de demonstration."""
import base64, io, math, random
import numpy as np
from PIL import Image, ImageFilter

def hero_svg(w=1600, h=330):
    random.seed(7)
    lines = []
    def branch(x, y, a, l, d):
        if d == 0 or l < 6:
            return
        x2, y2 = x + l * math.sin(a), y + l * math.cos(a)
        lines.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke-width="{max(d*0.9,0.6):.1f}"/>')
        for s in (-1, 1):
            branch(x2, y2, a + s * random.uniform(.35, .7), l * random.uniform(.7, .8), d - 1)
    cx = 1150
    lines.append(f'<line x1="{cx}" y1="10" x2="{cx}" y2="95" stroke-width="7"/>')
    branch(cx, 95, .55, 62, 7); branch(cx, 95, -.55, 62, 7)
    lung_l = f"M{cx-18} 70 C{cx-110} 70 {cx-215} 170 {cx-205} 285 C{cx-200} 330 {cx-120} 335 {cx-70} 312 C{cx-25} 285 {cx-12} 190 {cx-18} 70Z"
    lung_r = f"M{cx+18} 70 C{cx+110} 70 {cx+215} 170 {cx+205} 285 C{cx+200} 330 {cx+120} 335 {cx+70} 312 C{cx+25} 285 {cx+12} 190 {cx+18} 70Z"
    dots = "".join(f'<circle cx="{random.uniform(0,w):.0f}" cy="{random.uniform(0,h):.0f}" '
                   f'r="{random.uniform(.6,2):.1f}" fill="#67e8f9" opacity="{random.uniform(.15,.6):.2f}"/>'
                   for _ in range(110))
    ecg = ("M0 250 L300 250 L330 250 L345 215 L362 290 L380 160 L400 270 L415 250 L560 250 "
           "L700 250 L715 232 L730 268 L745 250 L1000 250")
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" preserveAspectRatio="xMaxYMid slice">
<defs>
<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#050b18"/><stop offset=".55" stop-color="#0a1c33"/><stop offset="1" stop-color="#06303f"/></linearGradient>
<radialGradient id="glow" cx=".5" cy=".5" r=".5"><stop offset="0" stop-color="#22d3ee" stop-opacity=".55"/><stop offset="1" stop-color="#22d3ee" stop-opacity="0"/></radialGradient>
<linearGradient id="lung" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#5eead4" stop-opacity=".55"/><stop offset="1" stop-color="#0ea5e9" stop-opacity=".18"/></linearGradient>
<filter id="blur"><feGaussianBlur stdDeviation="6"/></filter>
</defs>
<rect width="{w}" height="{h}" fill="url(#bg)"/>
<ellipse cx="{cx}" cy="190" rx="430" ry="230" fill="url(#glow)"/>
{dots}
<path d="{ecg}" fill="none" stroke="#22d3ee" stroke-width="2" opacity=".5"/>
<g filter="url(#blur)" opacity=".8"><path d="{lung_l}" fill="#22d3ee"/><path d="{lung_r}" fill="#22d3ee"/></g>
<path d="{lung_l}" fill="url(#lung)" stroke="#a5f3fc" stroke-width="2"/>
<path d="{lung_r}" fill="url(#lung)" stroke="#a5f3fc" stroke-width="2"/>
<g stroke="#ecfeff" stroke-linecap="round" opacity=".9">{"".join(lines)}</g>
</svg>'''

def hero_uri():
    return "data:image/svg+xml;base64," + base64.b64encode(hero_svg().encode()).decode()

def demo_xray(kind="normal", seed=0, size=224):
    """Radiographie synthetique (DEMONSTRATION uniquement, non clinique)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    img = 0.55 - 0.35 * ((xx - .5) ** 2 + (yy - .55) ** 2) * 3
    for cx in (.33, .67):
        img -= 0.32 * np.exp(-(((xx - cx) / .14) ** 2 + ((yy - .5) / .27) ** 2) ** 1.5)
    img += 0.25 * np.exp(-((xx - .5) / .045) ** 2) + 0.10 * (np.sin(yy * 60) > .6) * (np.abs(xx - .5) > .12)
    if kind == "pneumonia":
        for _ in range(rng.integers(2, 4)):
            cx, cy = rng.choice([.3, .7]) + rng.normal(0, .04), rng.uniform(.4, .68)
            img += rng.uniform(.25, .4) * np.exp(-(((xx - cx) / .09) ** 2 + ((yy - cy) / .11) ** 2))
    img = img + rng.normal(0, .02, img.shape)
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(1.1))
    return im.convert("RGB")

def b64(img: Image.Image, fmt="PNG"):
    buf = io.BytesIO(); img.save(buf, fmt)
    return base64.b64encode(buf.getvalue()).decode()
