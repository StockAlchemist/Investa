from PIL import Image
import os

def check_bbox(path):
    img = Image.open(path).convert("RGBA")
    width, height = img.size
    bg_color = img.getpixel((0, 0))
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
    print(f"[{path}] Size: {width}x{height}, BG: {bg_color}, BBox: ({min_x}, {min_y}) to ({max_x}, {max_y})")

check_bbox("./Assets.xcassets/AppIcon.appiconset/icon_1024x1024.png")
check_bbox("./Assets.xcassets/AppLogo.imageset/logo.png")
check_bbox("./Assets.xcassets/AppLogo.imageset/logo-dark.png")
