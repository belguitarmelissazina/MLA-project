import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
import pickle
from tqdm import tqdm
import os

# ============================================================
# CONFIG
# ============================================================
TRAIN_SAMPLES = 500_000
TEST_SAMPLES  = 50_000
OUTPUT_DIR = "./datasets/syn_numbers"
SEED = 42

IMG_H = 32
IMG_W = 64   # larger canvas to allow multi-digit
FINAL_SIZE = 32

# ============================================================
# FONTS
# ============================================================
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
]

def load_fonts():
    fonts = []
    for path in FONT_PATHS:
        if Path(path).exists():
            for size in [22, 24, 26]:
                fonts.append(ImageFont.truetype(path, size))
    if not fonts:
        raise RuntimeError("No fonts found")
    return fonts

# ============================================================
# SIMPLE SVHN-LIKE BACKGROUND
# ============================================================
def random_background():
    base = np.random.randint(30, 200, 3)
    bg = np.ones((IMG_H, IMG_W, 3), dtype=np.uint8) * base
    noise = np.random.normal(0, 8, bg.shape)
    return np.clip(bg + noise, 0, 255).astype(np.uint8)

def random_text_color(bg):
    mean = bg.mean()
    if mean > 128:
        return tuple(np.random.randint(0, 80, 3))
    else:
        return tuple(np.random.randint(180, 255, 3))

# ============================================================
# GENERATION
# ============================================================
def generate_dataset(n_samples, fonts, desc="Generating"):
    data, labels = [], []

    for _ in tqdm(range(n_samples), desc=desc):
        main_digit = np.random.randint(0, 10)

        # multi-digit string
        mode = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
        if mode == 1:
            digits = [main_digit]
            center_idx = 0
        elif mode == 2:
            if np.random.rand() > 0.5:
                digits = [np.random.randint(0, 10), main_digit]
                center_idx = 1
            else:
                digits = [main_digit, np.random.randint(0, 10)]
                center_idx = 0
        else:
            digits = [np.random.randint(0, 10), main_digit, np.random.randint(0, 10)]
            center_idx = 1

        text = "".join(map(str, digits))

        bg = random_background()
        img = Image.fromarray(bg)
        draw = ImageDraw.Draw(img)
        font = np.random.choice(fonts)
        color = random_text_color(bg)

        # text size
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        x = (IMG_W - text_w) // 2 + np.random.randint(-2, 3)
        y = (IMG_H - text_h) // 2 + np.random.randint(-2, 3)

        draw.text((x, y), text, fill=color, font=font)

        # mild augmentation
        if np.random.rand() > 0.5:
            img = img.rotate(
                np.random.uniform(-10, 10),
                resample=Image.BILINEAR,
                fillcolor=tuple(bg.mean(axis=(0, 1)).astype(int))
            )

        if np.random.rand() > 0.5:
            img = img.filter(ImageFilter.GaussianBlur(np.random.uniform(0.5, 1.5)))

        # ----------------------------------
        # CROP CENTRAL DIGIT
        # ----------------------------------
        char_widths = []
        offset = x
        for d in digits:
            b = draw.textbbox((0, 0), str(d), font=font)
            w = b[2] - b[0]
            char_widths.append(w)

        cx1 = offset + sum(char_widths[:center_idx])
        cx2 = cx1 + char_widths[center_idx]

        pad = 4
        cx1 = max(0, cx1 - pad)
        cx2 = min(IMG_W, cx2 + pad)

        crop = img.crop((cx1, 0, cx2, IMG_H))
        crop = crop.resize((FINAL_SIZE, FINAL_SIZE), Image.BILINEAR)

        data.append(np.array(crop))
        labels.append(main_digit)

    return np.array(data), labels

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    np.random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fonts = load_fonts()

    print("Generating train set...")
    train_x, train_y = generate_dataset(TRAIN_SAMPLES, fonts, "Train")

    print("Generating test set...")
    test_x, test_y = generate_dataset(TEST_SAMPLES, fonts, "Test")

    with open(f"{OUTPUT_DIR}/train.pkl", "wb") as f:
        pickle.dump({"data": train_x, "targets": train_y}, f)

    with open(f"{OUTPUT_DIR}/test.pkl", "wb") as f:
        pickle.dump({"data": test_x, "targets": test_y}, f)

    print(" DONE")
    print(f"Train: {train_x.shape}, Test: {test_x.shape}")
