import os
import sys
import csv
from lxml import etree
from PIL import Image, ImageDraw

# --- CONFIGURATION ---
PROCESSED_DATA_DIR = 'data/'
PROCESSED_IMAGES_DIR = os.path.join(PROCESSED_DATA_DIR, 'images')

PADDING = 15
IMG_WIDTH = 224
IMG_HEIGHT = 224

RECOVERING_PARSER = etree.XMLParser(recover=True)

def process_single_inkml(inkml_file_path):
    """
    Processes a single InkML file and prints the result as a CSV row.
    """
    try:
        tree = etree.parse(inkml_file_path, RECOVERING_PARSER)
        ns = {'ns': 'http://www.w3.org/2003/InkML'}
        latex_annotation_element = tree.find(".//ns:annotation[@type='truth']", namespaces=ns)
        if latex_annotation_element is None or not latex_annotation_element.text:
            return

        latex_string = latex_annotation_element.text.strip()
        traces = tree.findall('.//ns:trace', namespaces=ns)
        all_points = []
        for trace in traces:
            points_str = trace.text.strip() if trace.text else ""
            if not points_str: continue
            try:
                points = [(float(p.split()[0]), float(p.split()[1])) for p in points_str.split(',')]
                all_points.extend(points)
                all_points.append(None)
            except (ValueError, IndexError):
                continue
        
        valid_points = [p for p in all_points if p is not None]
        if not valid_points: return

        min_x, max_x = min(p[0] for p in valid_points), max(p[0] for p in valid_points)
        min_y, max_y = min(p[1] for p in valid_points), max(p[1] for p in valid_points)
        if max_x == min_x or max_y == min_y: return

        img_size = (int(max_x - min_x) + 2 * PADDING, int(max_y - min_y) + 2 * PADDING)
        image = Image.new('L', img_size, 255)
        draw = ImageDraw.Draw(image)

        for i in range(len(all_points) - 1):
            if all_points[i] and all_points[i+1]:
                p1 = (all_points[i][0] - min_x + PADDING, all_points[i][1] - min_y + PADDING)
                p2 = (all_points[i+1][0] - min_x + PADDING, all_points[i+1][1] - min_y + PADDING)
                draw.line([p1, p2], fill=0, width=2)
        
        # Determine if it's train, val, or test to save in the right subfolder
        path_parts = inkml_file_path.split(os.sep)
        split_name = 'train' # default
        if 'val' in path_parts:
            split_name = 'val'
        elif 'test' in path_parts:
            split_name = 'test'
            
        image_split_dir = os.path.join(PROCESSED_IMAGES_DIR, split_name)
        if not os.path.exists(image_split_dir):
            os.makedirs(image_split_dir)
        
        unique_filename = os.path.splitext(os.path.basename(inkml_file_path))[0] + '.png'
        image_save_path = os.path.join(image_split_dir, unique_filename)
        
        final_image = Image.new('L', (IMG_WIDTH, IMG_HEIGHT), 255)
        image.thumbnail((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
        paste_x = (IMG_WIDTH - image.width) // 2
        paste_y = (IMG_HEIGHT - image.height) // 2
        final_image.paste(image, (paste_x, paste_y))
        final_image.save(image_save_path)
        
        # Print the output as a CSV row. This will be captured by the shell script.
        # Use a strange delimiter to avoid issues with commas in LaTeX
        print(f'"{image_save_path}"<SEP>"{latex_string}"')

    except Exception:
        # If any error occurs, just exit silently for this file.
        return

if __name__ == '__main__':
    # This script expects exactly one command-line argument: the path to the inkml file.
    if len(sys.argv) != 2:
        sys.exit(1)
    
    inkml_path = sys.argv[1]
    process_single_inkml(inkml_path)
