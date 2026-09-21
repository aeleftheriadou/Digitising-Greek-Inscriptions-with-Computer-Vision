import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from skimage.feature import graycomatrix, graycoprops

sns.set(style="whitegrid")


# ------------------ METRICS ------------------

def compute_metrics(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.std()

    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    p = glcm / glcm.sum()
    entropy = -np.sum(p * np.log2(p + 1e-12))

    return sharpness, contrast, entropy


# ------------------ NORMALIZATION ------------------

def normalize_metrics(df):
    """Add normalized (0-1) columns for sharpness, contrast, entropy."""
    metrics = ['sharpness', 'contrast', 'entropy']
    for m in metrics:
        # Combine values from both folders
        all_vals = pd.concat([df[f'{m}_1'], df[f'{m}_2']])
        min_val = all_vals.min()
        max_val = all_vals.max()
        # Avoid division by zero if all values are identical
        if max_val == min_val:
            df[f'{m}_norm_1'] = 0.5
            df[f'{m}_norm_2'] = 0.5
        else:
            df[f'{m}_norm_1'] = (df[f'{m}_1'] - min_val) / (max_val - min_val)
            df[f'{m}_norm_2'] = (df[f'{m}_2'] - min_val) / (max_val - min_val)
    return df


# ------------------ COMPARISON ------------------

def compare_folders(folder1, folder2):
    results = []

    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)

    print("Folder1 files:", len(files1))
    print("Folder2 files:", len(files2))

    files2_map = {f.lower(): f for f in files2}

    matched = 0

    for fname in files1:
        fname_lower = fname.lower()

        if fname_lower not in files2_map:
            continue

        fname2 = files2_map[fname_lower]

        path1 = os.path.join(folder1, fname)
        path2 = os.path.join(folder2, fname2)

        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        if img1 is None or img2 is None:
            continue

        matched += 1

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        m1 = compute_metrics(img1)
        m2 = compute_metrics(img2)

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        ssim_score = ssim(gray1, gray2)

        results.append([fname, *m1, *m2, ssim_score])

    print("Total matched images:", matched)

    cols = [
        "filename",
        "sharpness_1", "contrast_1", "entropy_1",
        "sharpness_2", "contrast_2", "entropy_2",
        "ssim"
    ]

    return pd.DataFrame(results, columns=cols)


# ------------------ WATERSHED SEGMENTATION ------------------

def watershed_segmentation_white(img):
    """Watershed for segmenting white ridges on dark background."""
    
    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step 1: Invert so white ridges become dark valleys
    inverted = 255 - blur
    
    # Step 2: Find local minima of the inverted image (original ridge centers)
    from skimage.feature import peak_local_max
    
    local_min = peak_local_max(
        -inverted,  # negative to find minima
        min_distance=5,
        exclude_border=True,
        num_peaks=50
    )
    
    # Step 3: Create markers
    markers = np.zeros(gray.shape, dtype=np.int32)
    for i, (row, col) in enumerate(local_min, start=1):
        markers[row, col] = i
    
    # Step 4: Apply watershed on ORIGINAL image (not inverted)
    img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)
    
    img_ws = img_color.copy()
    img_ws[markers == -1] = [0, 0, 255]
    
    return img_ws, markers


def watershed_segmentation_black(img):
    """Watershed for segmenting dark letters on bright background."""
    
    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step 1: Find local MAXIMA (bright background centers), not minima
    from skimage.feature import peak_local_max
    
    local_max = peak_local_max(
        blur,  # positive to find maxima (bright spots)
        min_distance=5,
        exclude_border=True,
        num_peaks=50
    )
    
    # Step 2: Create markers at background centers
    markers = np.zeros(gray.shape, dtype=np.int32)
    for i, (row, col) in enumerate(local_max, start=1):
        markers[row, col] = i
    
    # Step 3: Apply watershed on original image
    img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)
    
    img_ws = img_color.copy()
    img_ws[markers == -1] = [0, 0, 255]
    
    return img_ws, markers


# ------------------ PLOTTING (with normalization support) ------------------

def plot_boxplots(df, label1, label2, normalized=False):
    suffix = "_norm" if normalized else ""
    metrics = [
        ("sharpness", "Sharpness"),
        ("contrast", "Contrast"),
        ("entropy", "Entropy"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    box_width = 0.4
    
    for ax, (m, title) in zip(axes, metrics):
        data = [df[f"{m}{suffix}_1"], df[f"{m}{suffix}_2"]]
        sns.boxplot(data=data, width=box_width, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([label1, label2])
        ax.set_title(title)
    
    plt.tight_layout()
    plt.show()


def plot_differences(df, label1, label2, normalized=False):
    suffix = "_norm" if normalized else ""
    metrics = [
        ("sharpness", "Sharpness"),
        ("contrast", "Contrast"),
        ("entropy", "Entropy"),
    ]
    for m, title in metrics:
        diff = df[f"{m}{suffix}_1"] - df[f"{m}{suffix}_2"]
        plt.figure()
        sns.histplot(diff, kde=True)
        plt.axvline(0, linestyle="--")
        title_str = f"{title} Difference ({label1} - {label2})"
        plt.title(title_str)
        plt.show()


def plot_per_image(df, label1, label2, normalized=False):
    suffix = "_norm" if normalized else ""
    metrics = [
        ("sharpness", "Sharpness"),
        ("contrast", "Contrast"),
        ("entropy", "Entropy"),
    ]
    for m, title in metrics:
        plt.figure(figsize=(10, 4))
        plt.plot(df[f"{m}{suffix}_1"].values, label=label1)
        plt.plot(df[f"{m}{suffix}_2"].values, label=label2)
        title_str = f"{title} per Image"
        plt.title(title_str)
        plt.legend()
        plt.show()


def plot_averages(df, label1, label2, normalized=False):
    suffix = "_norm" if normalized else ""
    metrics = [
        ("sharpness", "Sharpness"),
        ("contrast", "Contrast"),
        ("entropy", "Entropy"),
    ]
    avg_1 = [df[f"{m}{suffix}_1"].mean() for m, _ in metrics]
    avg_2 = [df[f"{m}{suffix}_2"].mean() for m, _ in metrics]
    titles = [title for _, title in metrics]

    x = np.arange(len(metrics))

    plt.figure()
    plt.bar(x - 0.2, avg_1, width=0.4, label=label1)
    plt.bar(x + 0.2, avg_2, width=0.4, label=label2)

    plt.xticks(x, titles)
    title_str = "Average Metrics Comparison"
    plt.title(title_str)
    plt.legend()
    plt.show()


def visualize_watershed_pairs(folder1, folder2, max_pairs):
    """Shows original + watershed boundary visualization and saves figures."""
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)

    files2_map = {f.lower(): f for f in files2}
    shown = 0
    
    output_folder = r"C:\Users\anasi\Desktop\inscriptions\watershed_figures"
    os.makedirs(output_folder, exist_ok=True)

    for fname in files1:
        if shown >= max_pairs:
            break

        fname_lower = fname.lower()
        if fname_lower not in files2_map:
            continue

        path1 = os.path.join(folder1, fname)
        path2 = os.path.join(folder2, files2_map[fname_lower])

        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        if img1 is None or img2 is None:
            continue

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        ws1, markers1 = watershed_segmentation_black(img1)
        ws2, markers2 = watershed_segmentation_white(img2)

        def enhance_boundaries(img, markers):
            boundary = (markers == -1).astype(np.uint8) * 255
            kernel = np.ones((3, 3), np.uint8)
            boundary = cv2.dilate(boundary, kernel, iterations=1)
            img[boundary == 255] = [0, 0, 255]
            return img

        ws1 = enhance_boundaries(ws1, markers1)
        ws2 = enhance_boundaries(ws2, markers2)

        def to_rgb(im):
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 6))

        plt.subplot(2, 2, 1)
        plt.imshow(to_rgb(img1))
        plt.title("1light - Original")
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.imshow(to_rgb(ws1))
        plt.title("1light - Watershed")
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.imshow(to_rgb(img2))
        plt.title("4lights - Original")
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.imshow(to_rgb(ws2))
        plt.title("4lights - Watershed")
        plt.axis("off")

        plt.tight_layout()
        
        name_without_ext = os.path.splitext(fname)[0]
        save_path = os.path.join(output_folder, f"{name_without_ext}_watershed_comparison.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        shown += 1


# ------------------ MAIN ------------------

if __name__ == "__main__":

    folder1 = r"C:\Users\anasi\Downloads\dataset\clean\standardised\cropped_clean\Berlin_1light_standardised_scaled_masked_cropped_selected"
    folder2 = r"C:\Users\anasi\Downloads\dataset\clean\standardised\cropped_clean\Mat_4lights_standardised_scaled_masked_cropped_resized"

    label1 = "1-light"
    label2 = "4-lights"

    df = compare_folders(folder1, folder2)

    print("Data shape:", df.shape)

    if df.empty:
        print("Error: no data collected. Check filenames and paths.")
        exit()

    # Add normalized columns
    df = normalize_metrics(df)

    output_csv = r"C:\Users\anasi\Downloads\dataset\clean\standardised\comparison_results.csv"
    df.to_csv(output_csv, index=False)

    visualize_watershed_pairs(folder1, folder2, max_pairs=1)
    print("CSV saved to:", output_csv)

    # Plot normalized values (0-1)
    print("\n--- Normalized metrics (0-1) ---")
    plot_boxplots(df, label1, label2, normalized=True)
    plot_differences(df, label1, label2, normalized=True)
    plot_per_image(df, label1, label2, normalized=True)
    plot_averages(df, label1, label2, normalized=True)