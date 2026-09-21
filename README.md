# Inscription Image Standardisation Project

This project contains a  set of scripts for preprocessing images of squeezes (acquired with different approaches, either using 1 or 4 lights) and performing simple image-processing / computer vision analysis. The scripts are designed to work with these specific image datasets that include a scale card or scale bar so they should be adapted if intended to be used in other case studies.s

The four core files in this project are:

- `open_cv.py`
- `scale_crop.py`
- `scale_crop_berlin.py`
- `standardisation.py`

---

## Project purpose

This project is meant to be used in a clear processing sequence. The scripts are not interchangeable: each one addresses a different step in preparing inscription images for consistent analysis and OCR.

Recommended order of use:

1. `standardisation.py` — first standardise the image appearance for the dataset.
2. `scale_crop.py` or `scale_crop_berlin.py` — then detect and remove the scale card and resize images to a common scale.
3. `open_cv.py` — finally compare image quality, visual metrics, and segmentation results across processed folders.

This workflow is designed for squeeze datasets where images must be normalised before visual comparison, measurement, OCR, or downstream analysis.

---

## 1. `standardisation.py`

### Purpose
This script should be used first. It prepares raw images for more stable downstream processing by resizing them proportionally and applying a roughness-based enhancement step before OCR or comparison.

### Main functions

- `ensure_dir(path)`
  - Creates output directories if missing.
- `resize_proportional(img, target_height=800)`
  - Resizes the image proportionally to a target height.
- `compute_roughness(gray)`
  - Measures image roughness using local binary patterns.
- `preprocess_for_easyocr(img_color, filename, target_height=800)`
  - Applies brightness normalisation and contrast enhancement based on roughness.
- `prepare_images_for_easyocr(input_dir, output_dir, target_height=800)`
  - Processes an entire folder and saves the enhanced images.

### Why it comes first
Before scale correction and metric comparison, the images need a consistent visual baseline. This step improves contrast and brightness, especially for OCR and later quality analysis.

### Typical use
Use this script to create a cleaner, standardised folder of images before running the scale-correction or comparison steps.

---

## 2. `scale_crop.py`

### Purpose
This script is used after initial image standardisation. It detects the scale card in each image, measures it, rescales the whole image to a fixed target, and masks the scale card out.

### Main functions

- `ensure_dir(path)`
  - Creates output directories if missing.
- `detect_scale_card(gray, plot)`
  - Finds the scale card contour from grayscale image edges.
- `rescale_and_mask_images(input_dir, output_dir, error_dir, scale_target_px, plot)`
  - Processes all JPG files in a directory, rescales them to a target pixel height, and removes the scale card region.

### Why it comes second
Once the images have been standardised visually, the scale card can be detected more reliably and the dataset can be brought to a uniform scale for comparison or further analysis.

### Working idea
The script assumes each image contains a visible calibration card. It:

1. Detects the card using edge and contour extraction.
2. Measures its long dimension.
3. Computes the scaling factor needed to reach the target pixel size.
4. Resizes the image.
5. Validates the scale after resizing.
6. Masks the scale region and saves the output.

### Typical use
Use this script when you want all images to be standardised to a fixed physical scale before further analysis.

### Important note
The script uses hard-coded local paths in `main()`, so the user must update these paths before running it on a different machine or dataset.

---

## 3. `scale_crop_berlin.py`

### Purpose
This is the Berlin-specific version of the scale-correction step. A separate script is used as this dataset was acquired using a scale bar (not a scale card) so it should be detected using a more specialised algorithm.

### Main functions

- `ensure_dir(path)`
- `detect_scale_card(gray, plot)`
  - Uses Sobel gradients and local variance to identify the checker-like scale pattern.
- `find_min_scale(input_dir)`
  - Scans all images, measures the scale card size, and returns the most common minimum size.
- `rescale_and_mask_images(input_dir, output_dir, error_dir, scale_target_px, plot)`
  - Resizes and masks based on the chosen target scale.

### Why it comes after the generic scale script
Use this script when the generic scale-card detector is insufficient for a particular dataset. It is more tuned to the Berlin image set and finds a dataset-wide reference scale based on the most common calibration size.

### How it differs from `scale_crop.py`

- It prefers a checker-pattern detection method.
- It calculates a dataset-wide reference scale instead of relying on a fixed manual value.
- It is adapted for Berlin dataset images and their calibration markers.

### Typical use
Use this script for the Berlin image set when the scale card texture or size distribution differs from the generic version.

---

## 4. `open_cv.py`

### Purpose
This script is the final analytical step in the workflow. It is used after scaling and standardisation to compare image quality, visual metrics, and segmentation results between datasets.

### Main functions

- `compute_metrics(img)`
  - Computes image sharpness, contrast, and entropy.
- `normalize_metrics(df)`
  - Normalises metric columns to a 0–1 range for comparison.
- `compare_folders(folder1, folder2)`
  - Matches images between two folders and computes structural similarity (SSIM) plus visual metrics.
- `watershed_segmentation_white(img)`
  - Segments bright ridges or features on dark backgrounds.
- `watershed_segmentation_black(img)`
  - Segments dark structures on bright backgrounds.
- `plot_boxplots(...)`, `plot_differences(...)`, `plot_per_image(...)`, `plot_averages(...)`
  - Generate charts for comparing metric distributions and image differences.

### Why it comes last
This file is best used once the dataset has already been standardised and scaled, because the comparisons are more meaningful when all images are in a common visual format.

### Typical use
Use this script when you want to study whether two image sets differ in texture, contrast, sharpness, or visual structure.

### Notes
This file is more of an analytical and exploratory utility than a batch-processing pipeline step.

---

## Recommended workflow

The intended chronological sequence is:

1. `standardisation.py` — normalise and prepare images for OCR / analysis.
2. `scale_crop.py` — general scale card detection and resize for ordinary datasets.
3. `scale_crop_berlin.py` — Berlin-specific scale card detection when the generic method is not appropriate.
4. `open_cv.py` — compare processed folders and inspect quality metrics.

This order ensures each script is used at the stage where it is the most useful and reliable.

---

## Dependencies

The scripts rely on common Python image-processing libraries, including:

- OpenCV (`cv2`)
- NumPy
- Matplotlib
- Pandas
- Seaborn
- scikit-image

These can be installed from the project requirements file if present in the workspace.

---

## Execution notes

- Most scripts use absolute Windows-style paths inside `main()`.
- These must be changed before running on another machine.
- Input and output folders should be checked carefully before each run.
- Some scripts create error folders for images that fail scale detection or validation.

---

## Summary

This project is a focused preprocessing and comparison pipeline for inscription images. The recommended sequence is to standardise the images first, remove the scale card and normalise the scale second, and only then use the visual comparison tools for analysis.

- `compute_metrics(img)`
  - Computes image sharpness, contrast, and entropy.
- `normalize_metrics(df)`
  - Normalises metric columns to a 0–1 range for comparison.
- `compare_folders(folder1, folder2)`
  - Matches images between two folders and computes structural similarity (SSIM) plus visual metrics.
- `watershed_segmentation_white(img)`
  - Segments bright ridges or features on dark backgrounds.
- `watershed_segmentation_black(img)`
  - Segments dark structures on bright backgrounds.
- `plot_boxplots(...)`, `plot_differences(...)`, `plot_per_image(...)`, `plot_averages(...)`
  - Generate charts for comparing metric distributions and image differences.

### Typical use
Use this script when you want to study whether two sets of images differ in texture, contrast, sharpness, or visual structure.

### Notes
This file is more of an analytical and exploratory utility than a batch-processing final pipeline step.

---

## 2. `scale_crop.py`

### Purpose
This script is designed to detect a scale card in an image, resize the full image to a fixed reference scale, and mask out the scale card so that the inscription area is standardised.

### Main functions

- `ensure_dir(path)`
  - Creates output directories if missing.
- `detect_scale_card(gray, plot)`
  - Finds the scale card contour from grayscale image edges.
- `rescale_and_mask_images(input_dir, output_dir, error_dir, scale_target_px, plot)`
  - Processes all JPG files in a directory, rescales them to a target pixel height, and removes the scale card region.

### Working idea
The script assumes each image contains a visible calibration card. It:

1. Detects the card using edge and contour extraction.
2. Measures its long dimension.
3. Computes the scaling factor needed to reach the target pixel size.
4. Resizes the image.
5. Validates the scale after resizing.
6. Masks the scale region and saves the output.

### Typical use
Use this script when you want all images to be standardised to a fixed physical scale before further analysis.

### Important note
The script uses hard-coded local paths in `main()`, so the user must update these paths before running it on a different machine or dataset.

---

## 3. `scale_crop_berlin.py`

### Purpose
This is a Berlin-specific variant of the scaling pipeline. It targets images whose scale card is a checker-like calibration pattern and finds a common reference scale across the whole dataset.

### Main functions

- `ensure_dir(path)`
- `detect_scale_card(gray, plot)`
  - Uses Sobel gradients and local variance to identify the checker-like scale pattern.
- `find_min_scale(input_dir)`
  - Scans all images, measures the scale card size, and returns the most common minimum size.
- `rescale_and_mask_images(input_dir, output_dir, error_dir, scale_target_px, plot)`
  - Resizes and masks based on the chosen target scale.

### How it differs from `scale_crop.py`

- It prefers a more specific checker-pattern detection method.
- It calculates a dataset-wide reference scale using all images instead of a fixed manual value.
- It is tuned for Berlin dataset images and their calibration markers.

### Typical use
Use this script for the Berlin image set when the scale card texture or size distribution differs from the generic version.

---

## 4. `standardisation.py`

### Purpose
This script prepares images for image analysis by normalising brightness and contrast while preserving the inscription details as much as possible. 

### Main functions

- `ensure_dir(path)`
- `resize_proportional(img, target_height=800)`
  - Resizes the image proportionally to a target height.
- `compute_roughness(gray)`
  - Measures image roughness using local binary patterns.
- `preprocess_for_easyocr(img_color, filename, target_height=800)`
  - Applies image normalisation and enhancement based on roughness.
- `prepare_images_for_easyocr(input_dir, output_dir, target_height=800)`
  - Processes all JPG files in a folder and writes enhanced images to a standardised output directory.

### Processing logic
The script adapts its enhancement method based on image roughness:

- If the image is very rough, it applies stronger smoothing and light contrast enhancement.
- If the image is smoother, it uses stronger contrast enhancement via CLAHE.

This is intended to balance readability and preserve inscription features before OCR.

### Typical use
Use this script as the final preparation step before OCR or text recognition pipelines.

---

## Recommended workflow

A reasonable processing sequence for a dataset is:

1. Run `standardisation.py` for general preparation.
2. Use `scale_crop.py` or `scale_crop_berlin.py` when calibration scale cards must be detected and removed.
3. Use `open_cv.py` for analysis and comparison of image quality across folders.

The scripts are not all meant to be run together automatically in one single command; they are modular utilities for different processing stages.

---

## Dependencies

The scripts rely on common Python image-processing libraries, including:

- OpenCV (`cv2`)
- NumPy
- Matplotlib
- Pandas
- Seaborn
- scikit-image

These can be installed from the project requirements file if present in the workspace.

---

## Execution notes

- Most scripts use absolute Windows-style paths inside `main()`.
- These must be changed before running on another machine.
- Input and output folders should be checked carefully before each run.
- Some scripts create error folders for images that fail scale detection or validation.

---

## Summary

This project is a focused preprocessing pipeline for images of squeezes. It helps standardise physical scale, improve visual quality, and prepare images for image analysis and later OCR / ML analysis.
