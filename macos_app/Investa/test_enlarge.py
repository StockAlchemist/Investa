from PIL import Image

def process_image(path, out_path, scale_factor=1.25):
    img = Image.open(path).convert("RGBA")
    width, height = img.size
    
    bg_color = img.getpixel((0, 0))
    
    # Find bounding box
    min_x, min_y = width, height
    max_x, max_y = 0, 0
    
    for y in range(height):
        for x in range(width):
            r, g, b, a = img.getpixel((x, y))
            br, bg, bb, ba = bg_color
            if abs(r - br) > 10 or abs(g - bg) > 10 or abs(b - bb) > 10:
                if x < min_x: min_x = x
                if y < min_y: min_y = y
                if x > max_x: max_x = x
                if y > max_y: max_y = y
                
    print(f"[{path}] Found symbol bbox: ({min_x}, {min_y}) to ({max_x}, {max_y})")
    
    symbol_w = max_x - min_x + 1
    symbol_h = max_y - min_y + 1
    
    symbol = img.crop((min_x, min_y, max_x + 1, max_y + 1))
    
    new_w = int(symbol_w * scale_factor)
    new_h = int(symbol_h * scale_factor)
    
    symbol_resized = symbol.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create new background
    new_img = Image.new("RGBA", (width, height), bg_color)
    
    # Paste centered
    paste_x = (width - new_w) // 2
    paste_y = (height - new_h) // 2
    new_img.paste(symbol_resized, (paste_x, paste_y), symbol_resized)
    
    new_img.save(out_path)
    print(f"Saved {out_path}")

process_image("./Assets.xcassets/AppIcon.appiconset/icon_1024x1024.png", "/Users/kmatan/.gemini/antigravity-ide/brain/aa83e2ed-05c1-473f-9fa4-1cfd1d822ae6/artifacts/test_appicon.png", 1.25)
process_image("./Assets.xcassets/AppLogo.imageset/logo.png", "/Users/kmatan/.gemini/antigravity-ide/brain/aa83e2ed-05c1-473f-9fa4-1cfd1d822ae6/artifacts/test_applogo.png", 1.25)

