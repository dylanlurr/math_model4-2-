import os

# --- CONFIGURATION ---
BASE_CROHME_DIR = 'CROHME_dataset/'
PROCESSED_DATA_DIR = 'data/'

def generate_list_for_split(split_name):
    """
    Scans a directory and saves all .inkml file paths to a text file.
    """
    inkml_split_dir = os.path.join(BASE_CROHME_DIR, 'INKML', split_name)
    output_txt_file = os.path.join(PROCESSED_DATA_DIR, f'file_list_{split_name}.txt')
    
    file_list = []
    for root, _, files in os.walk(inkml_split_dir):
        for f in files:
            if f.endswith('.inkml'):
                file_list.append(os.path.join(root, f))

    if file_list:
        print(f"Found {len(file_list)} files in '{split_name}'. Saving list to {output_txt_file}")
        with open(output_txt_file, 'w') as f:
            for item in file_list:
                f.write(f"{item}\n")
    else:
        print(f"Warning: No .inkml files found for split '{split_name}' in directory {inkml_split_dir}")

if __name__ == '__main__':
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
        
    generate_list_for_split('train')
    generate_list_for_split('val')
    # generate_list_for_split('test')
    print("\nFile lists generated successfully.")