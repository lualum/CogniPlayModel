import os
import cv2
import numpy as np
from pathlib import Path

def clean_drawing(image_path, output_path, params=None):
    """
    Clean a drawing by removing noise, edge artifacts, and small imperfections
    """
    # Default parameters
    if params is None:
        params = {
            'bilateral_d': 9,              # Diameter for bilateral filter
            'bilateral_sigma_color': 75,   # Color sigma for bilateral filter
            'bilateral_sigma_space': 75,   # Space sigma for bilateral filter
            'morph_kernel_size': 3,        # Kernel size for morphological ops
            'min_area': 50,                # Minimum contour area to keep
            'border_crop': 10,             # Pixels to crop from edges
        }
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Get dimensions
    height, width = img.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Crop border edges to remove edge noise
    crop = params['border_crop']
    if crop > 0:
        gray = gray[crop:height-crop, crop:width-crop]
    
    # Step 2: Apply bilateral filter to smooth while preserving edges
    filtered = cv2.bilateralFilter(
        gray, 
        params['bilateral_d'], 
        params['bilateral_sigma_color'], 
        params['bilateral_sigma_space']
    )
    
    # Step 3: Adaptive thresholding for better line detection
    binary = cv2.adaptiveThreshold(
        filtered, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 
        2
    )
    
    # Step 4: Remove small noise using morphological operations
    kernel = np.ones((params['morph_kernel_size'], params['morph_kernel_size']), np.uint8)
    
    # Opening: removes small white noise
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Closing: fills small holes in lines
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Step 5: Remove small isolated components
    # Find all contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask for valid contours
    mask = np.zeros_like(closed)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= params['min_area']:
            cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # Step 6: Apply mask to get clean result
    result = cv2.bitwise_and(closed, mask)
    
    # Invert back to white background
    result = cv2.bitwise_not(result)
    
    # Add back the border if it was cropped
    if crop > 0:
        result = cv2.copyMakeBorder(
            result, 
            crop, crop, crop, crop, 
            cv2.BORDER_CONSTANT, 
            value=255
        )
    
    # Save result
    cv2.imwrite(output_path, result)
    
    return result

def process_all_tiffs(input_folder, output_folder, params=None):
    """
    Process all TIFF files in a folder
    """
    # Create output folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all TIFF files
    input_path = Path(input_folder)
    tiff_files = list(input_path.glob('*.tif')) + list(input_path.glob('*.tiff'))
    
    if not tiff_files:
        print("No TIFF files found in the folder.")
        return
    
    print(f"Found {len(tiff_files)} TIFF file(s). Processing...\n")
    
    # Process each file
    for i, tiff_file in enumerate(tiff_files, 1):
        print(f"[{i}/{len(tiff_files)}] Processing: {tiff_file.name}")
        
        try:
            output_path = Path(output_folder) / f"cleaned_{tiff_file.stem}.tiff"
            clean_drawing(str(tiff_file), str(output_path), params)
            print(f"  ✓ Saved to: {output_path.name}\n")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}\n")
    
    print("Processing complete!")

def preview_with_different_params(image_path):
    """
    Helper function to test different parameters on a single image
    """
    import matplotlib.pyplot as plt
    
    # Test different parameter sets
    param_sets = [
        {'name': 'Light Cleaning', 'min_area': 30, 'morph_kernel_size': 2},
        {'name': 'Medium Cleaning', 'min_area': 50, 'morph_kernel_size': 3},
        {'name': 'Heavy Cleaning', 'min_area': 100, 'morph_kernel_size': 4},
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Different cleaning levels
    for idx, params in enumerate(param_sets):
        result = clean_drawing(image_path, None, params)
        ax = axes[(idx+1)//2, (idx+1)%2]
        ax.imshow(result, cmap='gray')
        ax.set_title(params['name'])
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('preview_comparison.png', dpi=150, bbox_inches='tight')
    print("Preview saved as 'preview_comparison.png'")
    plt.show()

if __name__ == "__main__":
    # Configuration
    INPUT_FOLDER = "./drawings"
    OUTPUT_FOLDER = "./cleaned_drawings"
    
    # Custom parameters (adjust these based on your drawings)
    PARAMS = {
        'bilateral_d': 9,              # Higher = more smoothing
        'bilateral_sigma_color': 75,   
        'bilateral_sigma_space': 75,   
        'morph_kernel_size': 3,        # Larger = removes bigger noise
        'min_area': 50,                # Larger = removes more small specs
        'border_crop': 10,             # Pixels to remove from edges
    }
    
    # Process all files
    process_all_tiffs(INPUT_FOLDER, OUTPUT_FOLDER, PARAMS)
    
    # Uncomment to preview different settings on one image first:
    # preview_with_different_params("./input_drawings/sample.tiff")