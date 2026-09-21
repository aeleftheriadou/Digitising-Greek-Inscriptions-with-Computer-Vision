import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
# ------------------------------------------------------------
# UTILITY
# ------------------------------------------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ------------------------------------------------------------
# RESIZE WITHOUT DISTORTION
# ------------------------------------------------------------
def resize_proportional(img, target_height=800):
    h, w = img.shape[:2]
    if h == target_height:
        return img

    scale = target_height / float(h)
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)

# ------------------------------------------------------------
# ROUGHNESS DETECTION
# ------------------------------------------------------------
def compute_roughness(gray):
    lap = local_binary_pattern(gray, P=32, R=10, method="uniform")
    return lap.var()

# ------------------------------------------------------------
# PREPROCESS (SAVE ONLY STEP 4x)
# ------------------------------------------------------------
def preprocess_for_easyocr(img_color, filename, target_height=800):

    steps = []

    # --- 1. Resize ---
    resized = resize_proportional(img_color, target_height)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    steps.append(("1. Resized / Gray", gray))

    # --- 2. Normalize brightness ---
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    steps.append(("2. Intensity Normalized", norm))

    # --- 3. Compute roughness ---
    roughness = compute_roughness(norm)
    print(f"{filename}: roughness = {roughness:.2f}")

    # ------------------------------------------------------------
    # ADAPTATION BRANCHES (3X + 4X)
    # ------------------------------------------------------------

    if roughness > 75:
        # Very rough surface → strong smoothing, light contrast
        smoothed = cv2.bilateralFilter(norm, d=10, sigmaColor=10, sigmaSpace=10)
        steps.append(("3A. Strong Smoothing (Rough Surface)", smoothed))

        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(smoothed)
        steps.append(("4A. Light Contrast Boost", enhanced))

    else:
        # Low relief / smooth surface → no smoothing, strong contrast
        enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(norm)
        steps.append(("4B. Strong Contrast Boost (Smooth Surface)", enhanced))



    # ------------------------------------------------------------
    # ✔ STOP HERE — DO NOT THRESHOLD
    # RETURN enhanced AS FINAL OUTPUT
    # ------------------------------------------------------------

    # # --- Plot ---
    # plt.figure(figsize=(12, 12))
    # n = len(steps)

    # for i, (title, im) in enumerate(steps):
    #     plt.subplot((n + 1) // 2, 2, i + 1)
    #     plt.imshow(im, cmap="gray")
    #     plt.title(title)
    #     plt.axis("off")

    # plt.suptitle(f"Processing: {filename}", fontsize=14)
    # plt.tight_layout()
    # plt.show()

    return resized, enhanced   # <--- THIS is step 4X


# ------------------------------------------------------------
# PIPELINE FOR FOLDER
# ------------------------------------------------------------
def prepare_images_for_easyocr(input_dir, output_dir, target_height=800):
    ensure_dir(output_dir)

    paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))

    for path in paths:
        filename = os.path.basename(path)
        img = cv2.imread(path)

        if img is None:
            print(f"❌ Could not read {filename}")
            continue

        resized, enhanced = preprocess_for_easyocr(img, filename, target_height)

        # ✔ Save only the enhanced (step 4x) image
        outpath = os.path.join(output_dir, filename)
        cv2.imwrite(outpath, enhanced)

        print(f"✔ Saved enhanced step (4X) for {filename}")

    print("✅ All images processed.")

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    input_dir = "C:\\Users\\anasi\\Downloads\\dataset\\clean\\Mat_4lights" #change this with your own path
    output_dir = "C:\\Users\\anasi\\Downloads\\dataset\\clean\\Mat_4lights_standardised" #change this with your own path

    prepare_images_for_easyocr(input_dir, output_dir)

if __name__ == "__main__":
    main()
