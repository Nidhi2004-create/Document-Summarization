"""Builds terminal-style 'screenshots' from the real log files + the STRIDE data-flow diagram."""
import re, textwrap
from PIL import Image, ImageDraw, ImageFont
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from common import OUT_DIR

def _load_font(bold=False, size=15):
    """Load a monospace font cross-platform (Linux DejaVu -> Windows Courier New -> PIL default)."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono{}.ttf".format("-Bold" if bold else ""),
        "C:/Windows/Fonts/cour{}.ttf".format("bd" if bold else ""),  # Courier New on Windows
        "C:/Windows/Fonts/consola{}.ttf".format("b" if bold else ""),  # Consolas
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()  # last resort

FONT = _load_font(bold=False)
BOLD = _load_font(bold=True)

def terminal(lines, title, out, width_chars=104):
    wrapped = []
    for l in lines:
        wrapped += textwrap.wrap(l, width_chars, subsequent_indent="    ", replace_whitespace=False) or [""]
    lh = 21; W = int(width_chars * 9.05) + 40; H = len(wrapped) * lh + 70
    img = Image.new("RGB", (W, H), "#1e1f26"); d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 34], fill="#3a3c47")
    for i, c in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]): d.ellipse([14 + i*22, 11, 26 + i*22, 23], fill=c)
    d.text((W//2 - len(title)*4, 8), title, font=FONT, fill="#c8c9d1")
    y = 48
    for l in wrapped:
        col, f = "#d7dae0", FONT
        if l.startswith(("You >", "$ ")): col, f = "#7ee787", BOLD
        elif l.startswith("Bot >"): col = "#79c0ff"
        elif l.startswith("+") or l.startswith("|"): col = "#e3b341"
        elif l.startswith("==="): col, f = "#ffa657", BOLD
        d.text((20, y), l, font=f, fill=col); y += lh
    img.save(out)

def read(p): return [l.rstrip("\n") for l in open(f"{OUT_DIR}/{p}")]

# --- Screenshot 1: PySpark execution
lg = [l for l in read("log_02_preprocessing.txt") if not l.startswith((" |--", "root"))]
terminal(["$ python 02_pyspark_preprocessing.py"] + lg[:22], "PySpark execution - preprocessing", f"{OUT_DIR}/shot1_pyspark.png")

# --- Screenshot 2: summarisation results
lg = read("log_04_summarizer.txt")
terminal(["$ python 04_summarizer.py"] + [l for l in lg if l.strip()], "Summarisation model - training & evaluation", f"{OUT_DIR}/shot2_model.png", 112)

# --- Screenshot 3: chatbot
lg = read("log_05_chatbot_offline.txt")
lg = [l for l in lg if l.strip()]
terminal(["$ python 05_chatbot.py"] + lg[:16] + ["..."] , "DocSum chatbot (offline mode)", f"{OUT_DIR}/shot3_chatbot_a.png")
pi = next(i for i, l in enumerate(lg) if l.startswith("You > /paste"))
terminal(lg[pi:], "DocSum chatbot - PII redaction & injection-safe prompt", f"{OUT_DIR}/shot3_chatbot_b.png")

# --- STRIDE data-flow diagram
fig, ax = plt.subplots(figsize=(7.4, 3.5)); ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")
def zone(x, y, w, h, label, col):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ls="--", ec=col, lw=1.4)); ax.text(x + 1, y + h - 2.8, label, color=col, fontsize=7.5, weight="bold")
def box(x, y, w, h, txt, fc="#EAF1FB", ec="#2E5EAA", shape="r"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2" if shape == "r" else "round,pad=0.3,rounding_size=4",
                       fc=fc, ec=ec, lw=1.3); ax.add_patch(p); ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=7)
def arrow(a, b, txt="", off=(0, 1.6), col="#444", cs="arc3"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=col, lw=1.2, connectionstyle=cs))
    ax.text((a[0]+b[0])/2 + off[0], (a[1]+b[1])/2 + off[1], txt, ha="center", fontsize=6.3, color=col, bbox=dict(fc="white", ec="none", pad=0.6))
zone(1, 3, 17, 42, "User zone", "#B23B3B"); zone(21, 3, 36, 42, "Application / AWS VPC zone", "#2E5EAA"); zone(60, 3, 39, 42, "Data & 3rd-party zone", "#3A9D6E")
box(3, 20, 13, 9, "User /\nAnalyst\n(browser)", "#FBEAEA", "#B23B3B", "o")
box(23.5, 27, 15, 10, "Chatbot API\n(auth, redaction,\nrate limit)")
box(41, 27, 14.5, 10, "Summariser\nservice + prompt\ntemplates")
box(23.5, 8, 15, 10, "Audit log\n(CloudTrail /\nCloudWatch)", "#FFF6E0", "#C48A00")
box(41, 8, 14.5, 10, "PySpark job\n(EMR / local)\nIAM role")
box(63, 27, 15, 10, "LLM API\n(GPT provider)", "#EAF7F0", "#3A9D6E")
box(63, 8, 15, 10, "S3 bucket\nraw documents\nSSE-KMS", "#EAF7F0", "#3A9D6E")
box(82, 8, 15, 10, "S3 / Parquet\nsummaries", "#EAF7F0", "#3A9D6E")
arrow((16, 26), (23.5, 31), "HTTPS / TLS", (-1.5, 2.6)); arrow((38.5, 32), (41, 32))
arrow((55.5, 32), (63, 32), "TLS + API key", (0, 2.4))
arrow((31, 27), (31, 18), "logs", (2.5, 0), "#C48A00")
arrow((63, 13), (55.5, 13), "read (IAM)", (0, 2.2))
arrow((48, 8), (89.5, 8), "write results", (0, -3.6), "#444", "arc3,rad=0.22")
for (x, y, t) in [(4, 16, "S  R  I"), (25, 38.3, "S T D E"), (43, 38.3, "T I E"), (25, 19.3, "R T"), (43, 19.3, "T E"), (65, 38.3, "I  D"), (65, 19.3, "T I D"), (84, 19.3, "T I")]:
    ax.text(x, y, t, fontsize=7, color="#B23B3B", weight="bold")
ax.text(50, -1.2, "Red letters = STRIDE categories that apply to that element (see Table 7)", ha="center", fontsize=7, style="italic")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig9_dfd.png", dpi=170); print("figures done")
