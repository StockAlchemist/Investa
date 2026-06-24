from PIL import Image
import glob
import os

def process_file(path, scale_factor=1.35):
    try:
        img = Image.open(path).convert("RGBA")
        orig_w, orig_h = img.size
        
        # Scale the entire image
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        scaled_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Crop the center to the original size
        left = (new_w - orig_w) // 2
        top = (new_h - orig_h) // 2
        right = left + orig_w
        bottom = top + orig_h
        
        cropped_img = scaled_img.crop((left, top, right, bottom))
        
        cropped_img.save(path)
        print(f"Processed {path} ({orig_w}x{orig_h})")
    except Exception as e:
        print(f"Failed to process {path}: {e}")

# Process AppIcon
for p in glob.glob("./Assets.xcassets/AppIcon.appiconset/*.png"):
    process_file(p)

# Process AppLogo
for p in glob.glob("./Assets.xcassets/AppLogo.imageset/*.png"):
    process_file(p)

