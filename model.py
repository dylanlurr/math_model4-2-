import torch
import torch.nn as nn
import torchvision.models as models
import math

class EncoderCNN(nn.Module):
    """
    CNN Encoder using a pre-trained ResNet-50.
    It extracts feature maps from the input image.
    """
    def __init__(self, embed_size, train_cnn=False):
        super(EncoderCNN, self).__init__()
        self.train_cnn = train_cnn
        
        # Use ResNet-50 for its powerful feature extraction
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # The first layer needs to be adapted for grayscale (1 channel) images
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Freeze all layers of the pre-trained ResNet if not training them
        for param in resnet.parameters():
            param.requires_grad = train_cnn
            
        # Remove the final average pool and fully connected layers
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        # Add a convolutional layer to get to the desired embedding size
        self.conv = nn.Conv2d(2048, embed_size, kernel_size=1)

    def forward(self, images):
        # images shape: (batch_size, 1, H, W)
        features = self.resnet(images) # (batch, 2048, H/32, W/32)
        features = self.conv(features) # (batch, embed_size, H/32, W/32)
        
        # Reshape for the Transformer: (batch_size, seq_len, embed_size)
        batch_size, embed_size, height, width = features.shape
        features = features.view(batch_size, embed_size, -1)
        features = features.permute(0, 2, 1) # (batch_size, height*width, embed_size)
        return features


class DecoderTransformer(nn.Module):
    """
    Transformer Decoder that generates the LaTeX sequence.
    """
    def __init__(self, embed_size, vocab_size, num_heads, num_layers, dropout, pad_idx):
        super(DecoderTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(embed_size, dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.pad_idx = pad_idx

    def make_tgt_mask(self, tgt):
        """Creates a subsequent mask to prevent the decoder from 'cheating' and looking ahead."""
        tgt_len = tgt.shape[1]
        # This creates a lower triangular matrix, True for positions that are allowed to attend.
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        return tgt_sub_mask

    def forward(self, features, captions):
        # The target sequence for the decoder should not include the last token (<EOS>)
        captions_input = captions[:, :-1]
        
        # Create a subsequent mask for the decoder's self-attention
        tgt_sub_mask = self.make_tgt_mask(captions_input)
        
        # Create a padding mask to ignore <PAD> tokens in the caption
        tgt_padding_mask = (captions_input == self.pad_idx)

        # Embed the captions and add positional information
        embedded_captions = self.embedding(captions_input)
        embedded_captions = self.positional_encoding(embedded_captions)
        
        # Get the decoder output
        output = self.transformer_decoder(
            tgt=embedded_captions, 
            memory=features, 
            tgt_mask=tgt_sub_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # Pass the output through the final linear layer to get predictions
        predictions = self.fc_out(output)
        return predictions

# Helper class for positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_size)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MathOCRModel(nn.Module):
    """
    The complete Image-to-Sequence model tying the Encoder and Decoder together.
    """
    def __init__(self, embed_size, vocab_size, num_heads, num_layers, dropout, pad_idx, train_cnn=False):
        super(MathOCRModel, self).__init__()
        self.encoder = EncoderCNN(embed_size, train_cnn)
        self.decoder = DecoderTransformer(embed_size, vocab_size, num_heads, num_layers, dropout, pad_idx)

    def forward(self, images, captions):
        encoded_features = self.encoder(images)
        outputs = self.decoder(encoded_features, captions)
        return outputs

