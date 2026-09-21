
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import csv

# -----------------------------
# UTILITY
# -----------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def detect_scale_card(gray, plot):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # emphasize texture (important!)
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    grad = cv2.convertScaleAbs(sobelx) + cv2.convertScaleAbs(sobely)

    _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    best = None
    best_score = -1

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w * h < 600:
            continue

        if w * h > 100000:
            continue

        roi = gray[y:y+h, x:x+w]

        # 🔥 key idea: checker pattern = high local variance
        score = np.var(roi)

        # also prefer compact regions
        aspect = w / (h + 1e-5)
        if aspect < 0.1 or aspect > 5:
            continue

        score = score / (w * h)

        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best is None:
        return None

    x, y, w, h = best

    if plot:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        plt.imshow(vis)
        plt.title("Detected checker-like scale")
        plt.axis("off")
        plt.show()

    return (x, y, w, h)





def find_min_scale(input_dir):
    scales = []

    for path in glob.glob(os.path.join(input_dir, "*.jpg")):
        img = cv2.imread(path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bbox = detect_scale_card(gray, plot=False)

        if bbox is None:
            continue

        _, _, w, h = bbox
        longest = max(w, h)

        scales.append(longest)

    if len(scales) == 0:
        return None

    # ----------------------------
    # CLUSTER INTO BINS (robust mode)
    # ----------------------------
    scales = np.array(scales)

    bin_size = 100  # adjust depending on noise
    binned = (scales // bin_size) * bin_size

    values, counts = np.unique(binned, return_counts=True)

    most_common_bin = values[np.argmax(counts)]
    print ('most common scale size:', most_common_bin + bin_size / 2)

    # return center of bin
    return most_common_bin + bin_size / 2

# -----------------------------
# RESCALE AND MASK IMAGES
# -----------------------------
def rescale_and_mask_images(input_dir, output_dir, error_dir, scale_target_px, plot):
    ensure_dir(output_dir)
    ensure_dir(error_dir)

    tolerance = 50  # ±50 px tolerance

    for path in glob.glob(os.path.join(input_dir, "*.jpg")):
        img = cv2.imread(path)
        if img is None:
            print(f"❌ Could not read: {os.path.basename(path)}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bbox = detect_scale_card(gray, plot)

        if bbox is None:
            cv2.imwrite(os.path.join(error_dir, os.path.basename(path)), img)
            print(f"⚠️ No scale detected, saved to errors: {os.path.basename(path)}")
            continue

        x, y, w, h = bbox
        current_longest = max(w, h)

        # Skip scales outside 200–700 px
        if current_longest < 100 or current_longest > 800:
            cv2.imwrite(os.path.join(error_dir, os.path.basename(path)), img)
            print(f"⚠️ Scale out of bounds ({current_longest}px), skipped: {os.path.basename(path)}")
            continue

        # Rescale
        scale_factor = scale_target_px / current_longest
        resized = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

        # Detect scale after resizing
        gray_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        bbox_resized = detect_scale_card(gray_resized, plot)
        measured_after = max(bbox_resized[2], bbox_resized[3]) if bbox_resized else None

        if measured_after is None or not (scale_target_px - tolerance <= measured_after <= scale_target_px + tolerance):
            cv2.imwrite(os.path.join(error_dir, os.path.basename(path)), resized)
            print(f"⚠️ Scale out of ±{tolerance}px after resize, saved to errors: "
                  f"{os.path.basename(path)} ({measured_after if measured_after else 'N/A'}px)")
            continue

        # Mask scale card
        x_r, y_r, w_r, h_r = bbox_resized
        resized[y_r:y_r+h_r, x_r:x_r+w_r] = 0

        # Save only rescaled + masked
        cv2.imwrite(os.path.join(output_dir, os.path.basename(path)), resized)

        # Print original scale, scale factor, scale after resizing
        print(f"✅ {os.path.basename(path)}: scale before = {current_longest}px, "
              f"scale factor = {scale_factor:.3f}, scale after = {measured_after}px")

# -----------------------------
# MAIN
# -----------------------------
def main():
    input_dir = "C:\\Users\\anasi\\Downloads\\dataset\\clean\\Berlin_1light_standardised" #change this with your own path
    scaled_dir = "C:\\Users\\anasi\\Downloads\\dataset\\clean\\nBerlin_1light_standardised_scaled"#change this with your own path
    masked_dir = "C:\\Users\\anasi\\Downloads\\dataset\\clean\\nBerlin_1light_standardised_scaled_masked"#change this with your own path
    errors_dir = "C:\\Users\\anasi\\Downloads\\dataset\\clean\\nBerlin_1light_standardised_scaled_errors"#change this with your own path

    print("Finding smallest scale in all images...")
    min_scale = find_min_scale(input_dir)
    if min_scale is None:
        print("❌ No scales detected in any images.")
        return
    print(f"Reference scale for resizing: {min_scale}px")

    print("Masking scale cards with black and saving errors...")
    rescale_and_mask_images(input_dir, masked_dir, errors_dir, min_scale, plot=True)

    print("✅ Done.")

if __name__ == "__main__":
    main()