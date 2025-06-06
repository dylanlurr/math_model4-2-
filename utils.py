import os
import csv
from lxml import etree
from PIL import Image, ImageDraw
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_CROHME_DIR = 'CROHME_dataset/'
PROCESSED_DATA_DIR = 'data/'
PROCESSED_IMAGES_DIR = os.path.join(PROCESSED_DATA_DIR, 'images')

PADDING = 15
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 1000 # Process 1000 files at a time to keep memory usage low

# Create a lenient XML parser that can recover from formatting errors.
RECOVERING_PARSER = etree.XMLParser(recover=True)

def inkml_to_image_and_latex(inkml_file_path):
    """
    Parses a single InkML file. This function remains the same.
    """
    try:
        tree = etree.parse(inkml_file_path, RECOVERING_PARSER)
        ns = {'ns': 'http://www.w3.org/2003/InkML'}
        latex_annotation_element = tree.find(".//ns:annotation[@type='truth']", namespaces=ns)
        if latex_annotation_element is None or not latex_annotation_element.text:
            return None, None
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
        
        if not any(p for p in all_points): return None, None
        valid_points = [p for p in all_points if p is not None]
        if not valid_points: return None, None
        min_x, max_x = min(p[0] for p in valid_points), max(p[0] for p in valid_points)
        min_y, max_y = min(p[1] for p in valid_points), max(p[1] for p in valid_points)
        if max_x == min_x or max_y == min_y: return None, None

        img_size = (int(max_x - min_x) + 2 * PADDING, int(max_y - min_y) + 2 * PADDING)
        image = Image.new('L', img_size, 255)
        draw = ImageDraw.Draw(image)

        for i in range(len(all_points) - 1):
            if all_points[i] and all_points[i+1]:
                p1 = (all_points[i][0] - min_x + PADDING, all_points[i][1] - min_y + PADDING)
                p2 = (all_points[i+1][0] - min_x + PADDING, all_points[i+1][1] - min_y + PADDING)
                draw.line([p1, p2], fill=0, width=2)
        
        return image, latex_string
    except Exception:
        return None, None

def process_data_split(split_name):
    """
    Processes a data split in smaller batches to prevent memory overload.
    """
    inkml_split_dir = os.path.join(BASE_CROHME_DIR, 'INKML', split_name)
    image_split_dir = os.path.join(PROCESSED_IMAGES_DIR, split_name)
    annotations_file = os.path.join(PROCESSED_DATA_DIR, f'{split_name}_annotations.csv')

    if not os.path.exists(image_split_dir):
        os.makedirs(image_split_dir)

    # Step 1: Gather a complete list of all files to process. This uses minimal memory.
    inkml_files_to_process = []
    for root, _, files in os.walk(inkml_split_dir):
        for f in files:
            if f.endswith('.inkml'):
                inkml_files_to_process.append(os.path.join(root, f))
    
    if not inkml_files_to_process:
        print(f"Warning: No .inkml files found in directory: {inkml_split_dir}")
        return

    print(f"Found {len(inkml_files_to_process)} InkML files in '{split_name}'. Starting processing in batches of {BATCH_SIZE}...")
    
    total_success_count = 0
    is_first_batch = True

    # Step 2: Process the list of files in batches.
    for i in tqdm(range(0, len(inkml_files_to_process), BATCH_SIZE), desc=f"Processing {split_name} batches"):
        batch_files = inkml_files_to_process[i:i + BATCH_SIZE]
        records = []

        for inkml_path in batch_files:
            image, latex_string = inkml_to_image_and_latex(inkml_path)

            if image and latex_string:
                final_image = Image.new('L', (IMG_WIDTH, IMG_HEIGHT), 255)
                image.thumbnail((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
                paste_x = (IMG_WIDTH - image.width) // 2
                paste_y = (IMG_HEIGHT - image.height) // 2
                final_image.paste(image, (paste_x, paste_y))
                
                unique_filename = os.path.splitext(os.path.relpath(inkml_path, inkml_split_dir))[0].replace(os.sep, '_') + '.png'
                image_save_path = os.path.join(image_split_dir, unique_filename)
                final_image.save(image_save_path)
                
                records.append({'image_path': image_save_path, 'formula': latex_string})

        # Step 3: Write the results of the current batch to the CSV file.
        if not records:
            continue

        # Open in 'write' mode for the first batch to create the file and header.
        # Open in 'append' mode for all subsequent batches.
        write_mode = 'w' if is_first_batch else 'a'
        with open(annotations_file, write_mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image_path', 'formula'])
            if is_first_batch:
                writer.writeheader()
            writer.writerows(records)

        is_first_batch = False
        total_success_count += len(records)
    
    # Final summary message
    if total_success_count > 0:
        print(f"\nFinished processing '{split_name}' split. Successfully created {total_success_count} pairs.")
        print(f"Annotations saved to '{annotations_file}'")
    else:
        print(f"CRITICAL: No data was successfully processed for the '{split_name}' split.")

if __name__ == '__main__':
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
        
    process_data_split('train')
    process_data_split('val')
    
    print("\nPreprocessing finished for all specified splits.")
