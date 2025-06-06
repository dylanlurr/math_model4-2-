import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os

# Import from our custom files
from model import MathOCRModel
from dataset import Vocabulary

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_FILE = "best_checkpoint.pth.tar" # Use the best saved model
PROCESSED_DATA_DIR = 'data/'
TRAIN_ANNOTATIONS_FILE = os.path.join(PROCESSED_DATA_DIR, 'train_annotations.csv')

# --- MODEL PARAMETERS (must match the trained model) ---
EMBED_SIZE = 512
NUM_LAYERS = 6
NUM_HEADS = 8
DROPOUT = 0.1

def load_model_and_vocab():
    """Loads the trained model checkpoint and the vocabulary."""
    # 1. Load the vocabulary that was built during training
    df = pd.read_csv(TRAIN_ANNOTATIONS_FILE)
    vocab = Vocabulary.build_vocabulary(df['formula'].tolist(), freq_threshold=5)
    vocab_size = len(vocab)
    pad_idx = vocab.stoi["<PAD>"]
    
    # 2. Initialize the model with the same architecture
    model = MathOCRModel(
        embed_size=EMBED_SIZE,
        vocab_size=vocab_size,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pad_idx=pad_idx
    ).to(DEVICE)
    
    # 3. Load the saved weights
    print(f"=> Loading checkpoint '{CHECKPOINT_FILE}'")
    if not os.path.exists(CHECKPOINT_FILE):
        raise FileNotFoundError(f"Checkpoint file not found! Please train the model first by running train.py")
        
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    
    model.eval() # Set the model to evaluation mode
    return model, vocab

def predict(model, vocab, image_path):
    """
    Takes a single image, preprocesses it, and returns the predicted LaTeX string.
    """
    # Define the same transformations used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Must match training image size
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Load and preprocess the image
    try:
        image = Image.open(image_path).convert("L") # Convert to grayscale
        image = transform(image).unsqueeze(0).to(DEVICE) # Add batch dimension
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return
        
    # Start the prediction loop
    with torch.no_grad():
        features = model.encoder(image)
        # Start the sequence with the <SOS> token
        caps = [vocab.stoi["<SOS>"]]
        
        for i in range(256): # Max length of the formula
            trg_tensor = torch.LongTensor(caps).unsqueeze(0).to(DEVICE)
            
            # The model's forward method is designed for training, 
            # so we call the decoder directly.
            output = model.decoder(features, trg_tensor)
            
            # Get the single last prediction
            predicted_id = output.argmax(dim=2)[:, -1].item()
            caps.append(predicted_id)
            
            if predicted_id == vocab.stoi["<EOS>"]:
                break
                
    # Convert the list of indices back to a human-readable formula
    predicted_formula = "".join([vocab.itos[idx] for idx in caps[1:-1]]) # Exclude SOS/EOS
    return predicted_formula

if __name__ == '__main__':
    # Load the trained model and vocabulary
    model, vocab = load_model_and_vocab()
    
    # --- TEST WITH YOUR OWN IMAGE ---
    # 1. Create a folder named 'my_test_images' in your project directory.
    # 2. Place an image of a math equation inside it.
    # 3. Change the path below to your image file.
    image_to_test = 'my_test_images/my_equation.png'
    
    print(f"\nMaking a prediction for image: {image_to_test}")
    result = predict(model, vocab, image_to_test)
    
    print("\n--- PREDICTION ---")
    print(f"Predicted LaTeX: {result}")
    print("--------------------")

