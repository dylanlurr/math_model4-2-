import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
import os

# Import from our custom files
from model import MathOCRModel
from dataset import get_loader

# --- CONFIGURATION ---
PROCESSED_DATA_DIR = 'data/'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- HYPERPARAMETERS ---
LEARNING_RATE = 3e-4
BATCH_SIZE = 16
NUM_EPOCHS = 50
LOAD_MODEL = False # Set to True to load a saved checkpoint
CHECKPOINT_FILE = "my_checkpoint.pth.tar"

# Model dimensions
EMBED_SIZE = 512
NUM_LAYERS = 6
NUM_HEADS = 8
DROPOUT = 0.1
TRAIN_CNN = False

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def validate_model(model, loader, criterion, vocab_size, pad_idx):
    """Evaluates the model on the validation set."""
    model.eval() # Set model to evaluation mode
    total_loss = 0
    with torch.no_grad():
        loop = tqdm(loader, total=len(loader), leave=False)
        for imgs, formulas in loop:
            imgs = imgs.to(DEVICE)
            formulas = formulas.to(DEVICE)
            
            outputs = model(imgs, formulas)
            targets = formulas[:, 1:].reshape(-1)
            outputs_reshaped = outputs.reshape(-1, vocab_size)
            
            loss = criterion(outputs_reshaped, targets)
            total_loss += loss.item()
            loop.set_description("Validation")
            
    avg_loss = total_loss / len(loader)
    model.train() # Set model back to training mode
    return avg_loss

def main():
    # --- DATA LOADING ---
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    print("Loading training data...")
    train_loader, train_vocab = get_loader(
        annotations_file=os.path.join(PROCESSED_DATA_DIR, 'train_annotations.csv'),
        transform=transform, batch_size=BATCH_SIZE
    )
    print("Loading validation data...")
    val_loader, _ = get_loader(
        annotations_file=os.path.join(PROCESSED_DATA_DIR, 'val_annotations.csv'),
        transform=transform, batch_size=BATCH_SIZE, vocab=train_vocab
    )
    
    # --- MODEL, OPTIMIZER, LOSS ---
    vocab_size = len(train_vocab)
    pad_idx = train_vocab.stoi["<PAD>"]
    
    model = MathOCRModel(
        embed_size=EMBED_SIZE, vocab_size=vocab_size, num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS, dropout=DROPOUT, pad_idx=pad_idx, train_cnn=TRAIN_CNN
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    if LOAD_MODEL and os.path.exists(CHECKPOINT_FILE):
        load_checkpoint(torch.load(CHECKPOINT_FILE), model, optimizer)

    best_val_loss = float('inf')

    # --- TRAINING LOOP ---
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        total_train_loss = 0
        
        for imgs, formulas in loop:
            imgs = imgs.to(DEVICE)
            formulas = formulas.to(DEVICE)
            
            outputs = model(imgs, formulas)
            targets = formulas[:, 1:].reshape(-1)
            outputs_reshaped = outputs.reshape(-1, vocab_size)
            
            optimizer.zero_grad()
            loss = criterion(outputs_reshaped, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            
            total_train_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
            loop.set_postfix(loss=loss.item())
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- VALIDATION ---
        avg_val_loss = validate_model(model, val_loader, criterion, vocab_size, pad_idx)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint, filename="best_checkpoint.pth.tar")

if __name__ == "__main__":
    main()
