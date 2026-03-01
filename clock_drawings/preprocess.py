import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
import argparse

# --- Logic for the New Cropping Algorithm ---

def find_longest_island(pixel_counts, chasm_threshold):
    """
    Finds the longest consecutive section where gaps of zeros 
    do not exceed the chasm_threshold.
    """
    # Create a boolean array where True means there is content
    has_content = pixel_counts > 0
    
    if not np.any(has_content):
        return 0, 0

    # Find indices where content exists
    indices = np.where(has_content)[0]
    
    # Calculate gaps between consecutive content indices
    gaps = np.diff(indices)
    
    # Find positions where the gap exceeds the threshold (the "chasms")
    chasm_breaks = np.where(gaps > chasm_threshold)[0]
    
    # Define the start and end of each "valid island"
    starts = np.insert(indices[chasm_breaks + 1], 0, indices[0])
    ends = np.append(indices[chasm_breaks], indices[-1])
    
    # Find the longest island
    lengths = ends - starts
    longest_idx = np.argmax(lengths)
    
    return starts[longest_idx], ends[longest_idx]

def island_crop(img_array, chasm_threshold, border_size, output_size):
    """
    Crops the image based on the longest consecutive section of content 
    on the X and Y axes.
    """
    # Get black pixel counts for rows (axis 1) and columns (axis 0)
    # Binary: content is 0, background is 255. 
    binary_content = (img_array == 0)
    row_counts = np.sum(binary_content, axis=1)
    col_counts = np.sum(binary_content, axis=0)
    
    # Find bounds using the island logic
    r_min, r_max = find_longest_island(row_counts, chasm_threshold)
    c_min, c_max = find_longest_island(col_counts, chasm_threshold)
    
    if r_min == r_max or c_min == c_max:
        return np.ones((output_size, output_size), dtype=np.uint8) * 255

    # Add border and clamp to image bounds
    r_min, r_max = max(0, r_min - border_size), min(img_array.shape[0], r_max + border_size)
    c_min, c_max = max(0, c_min - border_size), min(img_array.shape[1], c_max + border_size)
    
    # Make it square
    height, width = r_max - r_min, c_max - c_min
    side = max(height, width)
    center_r, center_c = (r_min + r_max) // 2, (c_min + c_max) // 2
    
    sr1, sr2 = center_r - side // 2, center_r + side // 2
    sc1, sc2 = center_c - side // 2, center_c + side // 2
    
    # Create canvas and paste
    canvas = np.ones((side, side), dtype=np.uint8) * 255
    src_r1, src_r2 = max(0, sr1), min(img_array.shape[0], sr2)
    src_c1, src_c2 = max(0, sc1), min(img_array.shape[1], sc2)
    dest_r1, dest_c1 = max(0, -sr1), max(0, -sc1)
    dest_r2, dest_c2 = dest_r1 + (src_r2 - src_r1), dest_c1 + (src_c2 - src_c1)
    
    canvas[dest_r1:dest_r2, dest_c1:dest_c2] = img_array[src_r1:src_r2, src_c1:src_c2]
    
    # Resize and clean up antialiasing
    res_img = Image.fromarray(canvas).resize((output_size, output_size), Image.Resampling.LANCZOS)
    return np.where(np.array(res_img) < 128, 0, 255).astype(np.uint8)

# --- Standard Processing Functions ---

def convert_to_bw(image):
    if image.mode != 'L': image = image.convert('L')
    return np.where(np.array(image) < 128, 0, 255).astype(np.uint8)

def is_rectangle_by_density(slice_obj, region_mask, density_threshold):
    actual_area = np.sum(region_mask)
    h, w = slice_obj[0].stop - slice_obj[0].start, slice_obj[1].stop - slice_obj[1].start
    return (actual_area / (h * w)) >= density_threshold

def is_near_edge(slice_obj, img_shape, edge_threshold):
    h, w = img_shape
    return (slice_obj[0].start < edge_threshold or slice_obj[0].stop > (h - edge_threshold) or
            slice_obj[1].start < edge_threshold or slice_obj[1].stop > (w - edge_threshold))

def process_image(image_path, args, output_dir, cropped_dir, current_idx, total_count):
    comp_path = output_dir / (image_path.stem + "_comparison.tif")
    crop_path = cropped_dir / (image_path.stem + "_clock.tif")

    progress_str = f"[{(current_idx/total_count)*100:6.2f}%] ({current_idx}/{total_count})"

    if comp_path.exists() and crop_path.exists():
        print(f"{progress_str} Skipping: {image_path.name}")
        return

    print(f"{progress_str} Processing: {image_path.name}")
    
    img_array = convert_to_bw(Image.open(image_path))
    original_array = img_array.copy()
    
    # Cleaning Blobs
    labeled_array, num_features = ndimage.label(img_array == 0)
    slices = ndimage.find_objects(labeled_array)
    
    for i, slc in enumerate(slices):
        if slc is None: continue
        mask = (labeled_array[slc] == (i + 1))
        if (np.sum(mask) < args.dirt_threshold or 
            is_rectangle_by_density(slc, mask, args.rectangle_density) or 
            is_near_edge(slc, img_array.shape, args.edge_threshold)):
            img_array[slc][mask] = 255

    # Save Comparison
    sep = np.zeros((img_array.shape[0], 20), dtype=np.uint8)
    Image.fromarray(np.hstack([img_array, sep, original_array])).save(comp_path, compression="tiff_deflate")
    
    # Save Island Crop
    cropped = island_crop(img_array, args.chasm_threshold, args.border_size, args.output_size)
    Image.fromarray(cropped).save(crop_path, compression="tiff_deflate")

def main():
    parser = argparse.ArgumentParser(description="Clock Processor with Island Cropping")
    parser.add_argument("--input-folder", default="drawings")
    parser.add_argument("--chasm-threshold", type=int, default=30, help="Max gap pixels allowed in an island")
    parser.add_argument("--dirt-threshold", type=int, default=50)
    parser.add_argument("--rectangle-density", type=float, default=0.95)
    parser.add_argument("--edge-threshold", type=int, default=50)
    parser.add_argument("--border-size", type=int, default=20)
    parser.add_argument("--output-size", type=int, default=640)
    
    args = parser.parse_args()
    
    input_path = Path(args.input_folder)
    out_path, crop_path = input_path / "processed", input_path / "cropped"
    out_path.mkdir(parents=True, exist_ok=True); crop_path.mkdir(parents=True, exist_ok=True)
    
    files = sorted(list(input_path.glob("*.tif*")))
    for idx, f in enumerate(files, 1):
        try:
            process_image(f, args, out_path, crop_path, idx, len(files))
        except Exception as e:
            print(f"Error on {f.name}: {e}")

if __name__ == "__main__":
    main()