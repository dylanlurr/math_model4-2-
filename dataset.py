import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from PIL import Image
from collections import Counter
import torchvision.transforms as transforms
import os

class Vocabulary:
    """Handles the mapping between characters and numerical indices."""
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @classmethod
    def build_vocabulary(cls, sentence_list, freq_threshold):
        """Builds the vocabulary from a list of formulas and returns a new vocab object."""
        vocab = cls(freq_threshold)
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for char in str(sentence): # Ensure sentence is a string
                frequencies[char] += 1
        
        for char, freq in frequencies.items():
            if freq >= freq_threshold:
                vocab.stoi[char] = idx
                vocab.itos[idx] = char
                idx += 1
        return vocab
    
    def numericalize(self, text):
        """Converts a text string into a sequence of numerical indices."""
        text = str(text) # Ensure text is a string
        return [self.stoi.get(char, self.stoi["<UNK>"]) for char in text]

class CROHMEDataset(Dataset):
    """Custom PyTorch Dataset for the CROHME data."""
    def __init__(self, annotations_file, transform=None, vocab=None):
        self.df = pd.read_csv(annotations_file)
        self.transform = transform
        self.vocab = vocab
        
        self.imgs = self.df["image_path"]
        self.formulas = self.df["formula"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        formula = self.formulas[index]
        img_path = self.imgs[index]
        
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)
            
        numericalized_formula = [self.vocab.stoi["<SOS>"]]
        numericalized_formula.extend(self.vocab.numericalize(formula))
        numericalized_formula.append(self.vocab.stoi["<EOS>"])
        
        return image, torch.tensor(numericalized_formula)

class Collate:
    """A custom collate function to handle padding."""
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        targets = [item[1] for item in batch]
        
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        imgs = torch.cat(imgs, dim=0)
        
        return imgs, targets

def get_loader(annotations_file, transform, batch_size=32, num_workers=4, shuffle=True, pin_memory=True, vocab=None, freq_threshold=5):
    """
    Creates and returns a DataLoader for a specific data split.
    If vocab is None, it builds a new vocabulary from the training data.
    """
    is_train = vocab is None
    
    if is_train:
        df = pd.read_csv(annotations_file)
        vocab = Vocabulary.build_vocabulary(df['formula'].tolist(), freq_threshold)

    dataset = CROHMEDataset(annotations_file, transform=transform, vocab=vocab)
    pad_idx = vocab.stoi["<PAD>"]
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=Collate(pad_idx=pad_idx)
    )
    
    return loader, vocab

if __name__ == '__main__':
    # --- Example of how to use the updated DataLoader ---
    PROCESSED_DATA_DIR = 'data/'
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    
    # 1. Get the training loader AND build the vocabulary from it
    train_loader, train_vocab = get_loader(
        annotations_file=os.path.join(PROCESSED_DATA_DIR, 'train_annotations.csv'),
        transform=transform,
        batch_size=2
    )
    
    # 2. Get the validation loader, PASSING IN the vocabulary from the training set
    val_loader, _ = get_loader(
        annotations_file=os.path.join(PROCESSED_DATA_DIR, 'val_annotations.csv'),
        transform=transform,
        batch_size=2,
        vocab=train_vocab # Crucial step!
    )
    
    print(f"Vocabulary size: {len(train_vocab)}")

    print("\n--- Training Batch ---")
    train_imgs, train_formulas = next(iter(train_loader))
    print("Images batch shape:", train_imgs.shape)
    print("Formulas batch shape:", train_formulas.shape)

    print("\n--- Validation Batch ---")
    val_imgs, val_formulas = next(iter(val_loader))
    print("Images batch shape:", val_imgs.shape)
    print("Formulas batch shape:", val_formulas.shape)
