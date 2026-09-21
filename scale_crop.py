import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# -----------------------------
# UTILITY
# -----------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# -----------------------------
# DETECT SCALE CARD (new 5 cm logic)
# -----------------------------
def detect_scale_card(gray, plot):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    _, binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        print("❌ Not enough objects detected")
        return None

    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    card_cnt = contours_sorted[1]  # second largest = scale card
    x, y, w, h = cv2.boundingRect(card_cnt)

    if plot:
        img_orig = gray.copy()
        img_contours = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_contours, contours, -1, (0, 0, 255), 2)
        img_card = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_card, [card_cnt], -1, (0, 0, 255), 2)
        cv2.rectangle(img_card, (x, y), (x + w, y + h), (0, 255, 0), 2)
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].imshow(img_orig, cmap='gray'); axs[0].set_title("Original Grayscale"); axs[0].axis('off')
        axs[1].imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB)); axs[1].set_title("All Contours"); axs[1].axis('off')
        axs[2].imshow(cv2.cvtColor(img_card, cv2.COLOR_BGR2RGB)); axs[2].set_title("Scale Card Contour"); axs[2].axis('off')
        plt.tight_layout(); plt.show()

    return (x, y, w, h)


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
    input_dir = "C:\\Users\\anasi\\Downloads\\dataset\\clean\\Mat_4lights_standardised" #change this with your own path
    output_dir = "C:\\Users\\anasi\\Downloads\\dataset\\clean\\Mat_4lights_standardised_scaled_masked_new" #change this with your own path
    error_dir = "C:\\Users\\anasi\\Downloads\\dataset\\clean\\Mat_4lights_standardised_scaled_errors_new" #change this with your own path

    # Fixed 5 cm target scale for all images
    target_scale_px = 125 #400
    print(f"Using fixed reference scale for resizing: {target_scale_px}px (5 cm scale)")

    print("Rescaling and masking images...")
    rescale_and_mask_images(input_dir, output_dir, error_dir, scale_target_px=target_scale_px, plot=False)

    print("✅ Done.")

if __name__ == "__main__":
    main()