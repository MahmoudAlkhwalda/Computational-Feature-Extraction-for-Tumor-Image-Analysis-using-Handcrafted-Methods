import cv2
import numpy as np
import pandas as pd
from skimage import color, feature, measure, filters, morphology, exposure
from skimage.feature.texture import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from skimage.transform import resize
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

LBP_RADIUS = 1
LBP_N_POINTS = 8 * LBP_RADIUS
GLCM_DISTANCES = [1, 2, 4]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]

def read_image(path, as_gray=False):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if as_gray:
        return color.rgb2gray(img_rgb)
    return img_rgb

def preprocess_image(img_rgb):
    if img_rgb.ndim == 2:
        gray = img_rgb
    else:
        gray = color.rgb2gray(img_rgb)
    blurred = filters.gaussian(gray, sigma=1.0)
    p2, p98 = np.percentile(blurred, (2, 98))
    img_rescale = exposure.rescale_intensity(blurred, in_range=(p2, p98))
    return img_rescale

def segment_tumor(gray_img):
    thresh_val = filters.threshold_otsu(gray_img)
    mask = gray_img < thresh_val  
    mask = morphology.remove_small_objects(mask, min_size=500)
    mask = morphology.remove_small_holes(mask, area_threshold=500)
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.binary_closing(mask, morphology.disk(5))
    return mask.astype(np.uint8)

def get_largest_region(mask):
    labels = measure.label(mask)
    if labels.max() == 0:
        return None, None
    props = measure.regionprops(labels)
    largest = max(props, key=lambda p: p.area)
    largest_mask = (labels == largest.label)
    return largest, largest_mask

def extract_shape_features(region):

    feats = {}
    feats['area'] = region.area
    feats['perimeter'] = region.perimeter
    feats['circularity'] = (4 * np.pi * region.area) / (region.perimeter ** 2 + 1e-9)
    minr, minc, maxr, maxc = region.bbox
    feats['bbox_height'] = maxr - minr
    feats['bbox_width'] = maxc - minc
    feats['extent'] = region.extent 
    feats['solidity'] = region.solidity  
    feats['eccentricity'] = region.eccentricity
    feats['orientation'] = region.orientation
    return feats

def extract_color_features(img_rgb, mask):
    feats = {}
    if img_rgb.ndim == 2:
        region_pixels = img_rgb[mask.astype(bool)]
        feats['mean_gray'] = np.mean(region_pixels) if region_pixels.size else 0
        feats['std_gray'] = np.std(region_pixels) if region_pixels.size else 0
    else:
        region_pixels = img_rgb[mask.astype(bool)]
        if region_pixels.size > 0:
            if region_pixels.ndim == 1:
                region_pixels = region_pixels.reshape(-1, 1)
            if region_pixels.shape[1] >= 3:
                for i, ch in enumerate(['R','G','B']):
                    feats[f'mean_{ch}'] = np.mean(region_pixels[:, i])
                    feats[f'std_{ch}'] = np.std(region_pixels[:, i])
            else:
                for i, ch in enumerate(['R','G','B']):
                    if i < region_pixels.shape[1]:
                        feats[f'mean_{ch}'] = np.mean(region_pixels[:, i])
                        feats[f'std_{ch}'] = np.std(region_pixels[:, i])
                    else:
                        feats[f'mean_{ch}'] = 0
                        feats[f'std_{ch}'] = 0
            gray = color.rgb2gray(img_rgb)
            feats['mean_brightness'] = np.mean(gray[mask.astype(bool)])
            feats['std_brightness'] = np.std(gray[mask.astype(bool)])
        else:
            feats['mean_R'] = feats['mean_G'] = feats['mean_B'] = 0
            feats['std_R'] = feats['std_G'] = feats['std_B'] = 0
            feats['mean_brightness'] = feats['std_brightness'] = 0
    return feats

def extract_glcm_features(gray_img, mask, distances=GLCM_DISTANCES, angles=GLCM_ANGLES):
    feats = {}
    imgq = (gray_img * 255).astype(np.uint8)
    coords = np.argwhere(mask)
    if coords.size == 0:
        for name in ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']:
            feats[f'glcm_{name}_mean'] = 0
            feats[f'glcm_{name}_std'] = 0
        return feats
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    sub = imgq[r0:r1, c0:c1]
    sub_mask = mask[r0:r1, c0:c1].astype(bool)

    try:

        glcm = graycomatrix(sub, distances=distances, angles=angles, levels=256, symmetric=True, normalize=True)
    except TypeError:

        glcm = graycomatrix(sub, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for p in props:
        vals = graycoprops(glcm, p)
        feats[f'glcm_{p}_mean'] = vals.mean()
        feats[f'glcm_{p}_std'] = vals.std()
    return feats

def extract_lbp_features(gray_img, mask, P=LBP_N_POINTS, R=LBP_RADIUS):
    feats = {}
    lbp = local_binary_pattern((gray_img*255).astype(np.uint8), P=P, R=R, method='uniform')
    region_vals = lbp[mask.astype(bool)]
    if region_vals.size == 0:
        feats['lbp_mean'] = 0
        feats['lbp_var'] = 0
    else:
        feats['lbp_mean'] = np.mean(region_vals)
        feats['lbp_var'] = np.var(region_vals)
 
        num_bins = min(P + 3, 20)
        hist, _ = np.histogram(region_vals, bins=num_bins, range=(0, P+2), density=True)
 
        for i in range(min(6, len(hist))):
            feats[f'lbp_hist_{i}'] = hist[i]
    return feats

def extract_hog_features(gray_img, mask):
    feats = {}

    coords = np.argwhere(mask)
    if coords.size == 0:
        feats['hog_mean'] = 0
        feats['hog_var'] = 0
        return feats
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    sub = gray_img[r0:r1, c0:c1]

    if sub.size == 0:
        feats['hog_mean'] = 0
        feats['hog_var'] = 0
        return feats
    try:
    
        target_size = (max(128, sub.shape[0]), max(128, sub.shape[1]))
        sub_resized = resize(sub, target_size, anti_aliasing=True)
    except Exception:
        sub_resized = sub

    if sub_resized.max() <= 1.0:
        sub_for_hog = (sub_resized * 255).astype(np.uint8)
    else:
        sub_for_hog = sub_resized.astype(np.uint8)
    try:
        hog_vec, hog_img = feature.hog(sub_for_hog, pixels_per_cell=(16,16),
                                       cells_per_block=(2,2), visualize=True, feature_vector=True)
    except Exception as e:
        feats['hog_mean'] = 0
        feats['hog_var'] = 0
        return feats
    feats['hog_mean'] = np.mean(hog_vec)
    feats['hog_var'] = np.var(hog_vec)
    return feats

def extract_all_features(image_path, visualize=False):
    img_rgb = read_image(image_path, as_gray=False)
    gray = preprocess_image(img_rgb)
    mask = segment_tumor(gray)
    region, region_mask = get_largest_region(mask)
    if region is None:
        raise RuntimeError("No connected region was revealed in the image after segmentation.")
    shape_feats = extract_shape_features(region)
    color_feats = extract_color_features(img_rgb, region_mask)
    glcm_feats = extract_glcm_features(gray, region_mask)
    lbp_feats = extract_lbp_features(gray, region_mask)
    hog_feats = extract_hog_features(gray, region_mask)

    all_feats = {}
    all_feats.update(shape_feats)
    all_feats.update(color_feats)
    all_feats.update(glcm_feats)
    all_feats.update(lbp_feats)
    all_feats.update(hog_feats)

    df = pd.DataFrame([all_feats])

    if visualize:
        fig, axes = plt.subplots(1, 4, figsize=(16,5))
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original RGB")
        axes[1].imshow(gray, cmap='gray')
        axes[1].set_title("Preprocessed Gray")
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title("Segmentation Mask")
        axes[3].imshow(img_rgb)
        try:
            contours = measure.find_contours(region_mask.astype(float), 0.5)
            for cnt in contours:
                if len(cnt) > 0:
                    axes[3].plot(cnt[:, 1], cnt[:, 0], '-r', linewidth=1)
        except Exception:
            pass
        axes[3].set_title("Detected Region Overlay")
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    return df, region, region_mask


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python cancer_features.py <image_path> [--viz]")
        print("\nNo image path provided. Creating a test image for demonstration...")
        test_img = np.ones((400, 400, 3), dtype=np.uint8) * 200
        center = (200, 200)
        radius = 80
        y, x = np.ogrid[:400, :400]
        mask_circle = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        test_img[mask_circle] = [100, 50, 50]  # Darker region
        noise = np.random.randint(-20, 20, (400, 400, 3))
        test_img = np.clip(test_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        test_path = "test_image.png"
        cv2.imwrite(test_path, cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
        print(f"Created test image: {test_path}")
        path = test_path
        viz = True  
    else:
        path = sys.argv[1]
        viz = '--viz' in sys.argv
    
    try:
        df, region, mask = extract_all_features(path, visualize=viz)
        print("=== Extracted Features ===")
        print(df.T) 
        df.to_csv("extracted_features.csv", index=False)
        print("\nSaved features to extracted_features.csv")
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
