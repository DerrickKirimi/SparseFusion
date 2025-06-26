!pip install mlflow

import base64
import io
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import mlflow

from google.colab import files

#!pip install --force-reinstall sympy

mlflow.set_tracking_uri("file:/content/mlruns")
mlflow.set_experiment("colab_experiment")

# Ensure every computation happens on the GPU when available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#To build the encoding and decoding functions we use the tinyshakespear dataset. However for the sake of brevity we do not pretrain the decoder model on it
#the training function should be able to do it without an issue as well as it could take both images and tex
# Download the dataset
!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespear/input.txt

# Read the file
text_path = "input.txt"
with open(text_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Add characters from the dataframe captions to the vocabulary
caption_text = " ".join(df['caption'].tolist())

# Combine characters from input.txt and captions, and include all alphabet characters
all_chars = sorted(list(set(text + caption_text + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")))

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(all_chars) }
stoi['<pad>']= len(all_chars) # Assign a unique index for padding
itos = { i:ch for i,ch in enumerate(all_chars) }
itos[len(all_chars)] = '<pad>'
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
vocab_size = len(stoi.keys())
print(f"Vocabulary size: {vocab_size}")

print(text[:500])  # preview first 500 characters

class PatchEmbeddings(nn.Module):
    def __init__(self, img_size=96, patch_size=16, hidden_dim=512):
        super().__init__()

        # Store the input image size
        self.img_size = img_size

        # Store the size of each patch
        self.patch_size = patch_size

        # Calculate the total number of patches
        self.num_patches = (img_size // patch_size) ** 2

        # Create a convolutional layer to extract patch embeddings
        # in_channels=3 assumes the input image has 3 color channels (RGB)
        # out_channels=hidden_dim sets the number of output channels to match the hidden dimension
        # kernel_size=patch_size and stride=patch_size ensure each patch is separately embedded
        self.conv = nn.Conv2d(in_channels=3, out_channels=hidden_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, X):
        # Extract patch embeddings from the input image
        X = self.conv(X)

        # Flatten the spatial dimensions (height and width) of the patch embeddings
        # This step flattens the patch dimensions into a single dimension
        X = X.flatten(2)

        # Transpose the dimensions to obtain the shape [batch_size, num_patches, hidden_dim]
        # This step brings the num_patches dimension to the second position
        X = X.transpose(1, 2)

        return X

#testing
img_size, patch_size,  num_hiddens, batch_size = 96, 16, 512, 4
patch_embeddings = PatchEmbeddings(img_size, patch_size, num_hiddens )
X = torch.zeros(batch_size, 3, img_size, img_size)
patch_embeddings(X).shape

#swapping linear for lazy linear for simplicity. Lazylinear can accept any arbitrary input dimension without having it specified

class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1, is_decoder=True):
        super().__init__()

        # Define the layers of the MLP
        layers = [
            # First linear layer that expands the input dimension from n_embd to 4 * n_embd
            nn.Linear(n_embd, 4 * n_embd),

            # Activation function: ReLU if is_decoder is True, else GELU
            nn.ReLU() if is_decoder else nn.GELU(),

            # Second linear layer that projects the intermediate dimension back to n_embd
            nn.Linear(4 * n_embd, n_embd),

            # Dropout layer for regularization
            nn.Dropout(dropout)
        ]

        # Create a sequential container to hold the layers
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input through the MLP layers
        return self.net(x)

#For the sake of this example consider embedding size to be 128
n_embd = 128
testmlp = MLP(n_embd)
mlp_input = torch.zeros(batch_size, 3, n_embd)
testmlp_out = testmlp(mlp_input)
testmlp_out.shape

class Head(nn.Module):
    def __init__(self, n_embd, head_size, dropout=0.1, is_decoder=False):
        super().__init__()

        # Linear layer for key projection
        self.key = nn.Linear(n_embd, head_size, bias=False)

        # Linear layer for query projection
        self.query = nn.Linear(n_embd, head_size, bias=False)

        # Linear layer for value projection
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

        # Flag indicating whether this head is used in the decoder
        self.is_decoder = is_decoder

    def forward(self, x):
        # Get the batch size (B), sequence length (T), and embedding dimension (C) from the input tensor
        B, T, C = x.shape

        # Compute key, query, and value projections
        k = self.key(x)   # Shape: [B, T, head_size]
        q = self.query(x) # Shape: [B, T, head_size]
        v = self.value(x) # Shape: [B, T, head_size]

        # Compute attention scores by taking the dot product of query and key
        # and scaling by the square root of the embedding dimension
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) # Shape: [B, T, T]

        if self.is_decoder:
            # If this head is used in the decoder, apply a causal mask to the attention scores
            # to prevent attending to future positions
            tril = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            wei = wei.masked_fill(tril == 0, float('-inf'))

        # Apply softmax to the attention scores to obtain attention probabilities
        wei = F.softmax(wei, dim=-1) # Shape: [B, T, T]

        # Apply dropout to the attention probabilities for regularization
        wei = self.dropout(wei)

        # Perform weighted aggregation of values using the attention probabilities
        out = wei @ v # Shape: [B, T, head_size]

        return out

#Example values for testing
n_embd, head_size, batch_size = 128, 16, 4

testhead = Head(n_embd, head_size)
head_input = torch.zeros(batch_size, 3, n_embd)
testhead_out = testhead(head_input)
testhead_out.shape # (B, T,H_size)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, dropout=0.1, is_decoder=False):
        super().__init__()

        # Ensure that the embedding dimension is divisible by the number of heads
        assert n_embd % num_heads == 0, "n_embd must be divisible by num_heads"

        # Create a ModuleList of attention heads
        self.heads = nn.ModuleList([
            Head(n_embd, n_embd // num_heads, dropout, is_decoder)
            for _ in range(num_heads)
        ])

        # Linear layer for projecting the concatenated head outputs
        self.proj = nn.Linear(n_embd, n_embd)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply each attention head to the input tensor
        head_outputs = [h(x) for h in self.heads]

        # Concatenate the outputs from all heads along the last dimension
        out = torch.cat(head_outputs, dim=-1)

        # Apply the projection layer to the concatenated outputs
        out = self.proj(out)

        # Apply dropout to the projected outputs for regularization
        out = self.dropout(out)

        return out

#Example values for testing
n_embd, n_head = 128, 8
testmha = MultiHeadAttention(n_embd, n_head)
head_input = torch.zeros(batch_size, 3, n_embd)
testmha_out = testmha(head_input)
testmha_out.shape # (B, T,H_size*n_heads = n_embed)

class Block(nn.Module):
    def __init__(self, n_embd, num_heads, dropout=0.1, is_decoder=False):
        super().__init__()

        # Layer normalization for the input to the attention layer
        self.ln1 = nn.LayerNorm(n_embd)

        # Multi-head attention module
        self.attn = MultiHeadAttention(n_embd, num_heads, dropout, is_decoder)

        # Layer normalization for the input to the FFN
        self.ln2 = nn.LayerNorm(n_embd)

        # Feed-forward neural network (FFN)
        self.ffn = MLP(n_embd)

    def forward(self, x):
        original_x = x  # Save the input for the residual connection

        # Apply layer normalization to the input
        x = self.ln1(x)

        # Apply multi-head attention
        attn_output = self.attn(x)

        # Add the residual connection (original input) to the attention output
        x = original_x + attn_output

        # Apply layer normalization to the input to the FFN
        x = self.ln2(x)

        # Apply the FFN
        ffn_output = self.ffn(x)

        # Add the residual connection (input to FFN) to the FFN output
        x = x + ffn_output

        return x

#Example values for testing
n_embd, head_size, batch_size = 128, 16, 4

testblock = Block(n_embd, n_head)
block_input = torch.zeros(batch_size, 3, n_embd)
testblock_out = testblock(block_input)
testblock_out.shape

"""Now all this can be be put together to implement a Vision Transformer"""

class ViT(nn.Module):
    def __init__(self, img_size, patch_size, num_hiddens, num_heads, num_blks, emb_dropout, blk_dropout):
        super().__init__()

        # Patch embedding layer to convert the input image into patches
        self.patch_embedding = PatchEmbeddings(img_size, patch_size, num_hiddens)

        # Learnable classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))

        # Calculate the number of patches
        num_patches = (img_size // patch_size) ** 2

        # Learnable position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, num_hiddens))

        # Dropout layer for the embeddings
        self.dropout = nn.Dropout(emb_dropout)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([Block(num_hiddens, num_heads, blk_dropout, is_decoder=False) for _ in range(num_blks)])

        # Layer normalization for the final representation
        self.layer_norm = nn.LayerNorm(num_hiddens)

    def forward(self, X):
        # Convert the input image into patch embeddings
        x = self.patch_embedding(X)

        # Expand the classification token to match the batch size
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        # Concatenate the classification token with the patch embeddings
        x = torch.cat((cls_tokens, x), dim=1)

        # Add the position embedding to the patch embeddings
        x += self.pos_embedding

        # Apply dropout to the embeddings
        x = self.dropout(x)

        # Pass the embeddings through the transformer blocks
        for block in self.blocks:
            x = block(x)

        # Apply layer normalization to the final representation
        x = self.layer_norm(x[:, 0])

        return x

#For purposes of testing
img_size, patch_size, num_hiddens, n_head, num_blks, dropout = 96, 16, 512, 8, 3, 0.1

testvit = ViT(img_size, patch_size, num_hiddens, n_head, num_blks, dropout, dropout)
vit_input = torch.zeros(batch_size, 3, img_size, img_size)
testvit_out = testvit(vit_input)
testvit_out.shape

class MultiModalProjector(nn.Module):
    def __init__(self, n_embd, image_embed_dim, dropout=0.1):
        super().__init__()

        # Define the projection network
        self.net = nn.Sequential(
            # Linear layer to expand the image embedding dimension
            nn.Linear(image_embed_dim, 4 * image_embed_dim),

            # GELU activation function
            nn.GELU(),

            # Linear layer to project the expanded image embeddings to the text embedding dimension
            nn.Linear(4 * image_embed_dim, n_embd),

            # Dropout layer for regularization
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pass the input through the projection network
        x = self.net(x)
        return x

#Example values for testing
n_embd,num_hiddens = 128, 512

testmmp = MultiModalProjector(n_embd,num_hiddens)
mmp_input = testvit_out
testmmp_out = testmmp(mmp_input)
testmmp_out.shape

#Expert module
class Expert(nn.Module):
    """ An MLP is a simple linear layer followed by a non-linearity i.e. each Expert """

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

#noisy top-k gating
class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        #layer for router logits
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear =nn.Linear(n_embed, num_experts)


    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)

        #Noise logits
        noise_logits = self.noise_linear(mh_output)

        #Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices

#Now create the sparse mixture of experts module
class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output

class SparseMoEBlock(nn.Module):
    def __init__(self, n_embd, num_heads, num_experts, top_k, dropout=0.1, is_decoder=False):
        super().__init__()

        # Layer normalization for the input to the attention layer
        self.ln1 = nn.LayerNorm(n_embd)

        # Multi-head attention module
        self.attn = MultiHeadAttention(n_embd, num_heads, dropout, is_decoder)

        # Layer normalization for the input to the FFN
        self.ln2 = nn.LayerNorm(n_embd)

        # Feed-forward neural network (FFN)
        self.sparseMoE = SparseMoE(n_embd, num_experts, top_k)

    def forward(self, x):
        original_x = x  # Save the input for the residual connection

        # Apply layer normalization to the input
        x = self.ln1(x)

        # Apply multi-head attention
        attn_output = self.attn(x)

        # Add the residual connection (original input) to the attention output
        x = original_x + attn_output

        # Apply layer normalization to the input to the FFN
        x = self.ln2(x)

        # Apply the FFN
        sparseMoE_output = self.sparseMoE(x)

        # Add the residual connection (input to FFN) to the FFN output
        x = x + sparseMoE_output

        return x

class MoEDecoderLanguageModel(nn.Module):
    def __init__(self, n_embd, image_embed_dim, vocab_size, num_heads, n_layer, num_experts, top_k, use_images=False):
        super().__init__()

        self.use_images = use_images

        # Token embedding table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # Position embedding table
        self.position_embedding_table = nn.Embedding(1000, n_embd)

        if use_images:
            # Image projection layer to align image embeddings with text embeddings
            self.image_projection = MultiModalProjector(n_embd, image_embed_dim)

        # Stack of transformer decoder blocks
        self.sparseMoEBlocks = nn.Sequential(*[SparseMoEBlock(n_embd, num_heads, num_experts, top_k, is_decoder=True) for _ in range(n_layer)])

        # Final layer normalization
        self.ln_f = nn.LayerNorm(n_embd)

        # Language modeling head
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, image_embeds=None, targets=None):
        # Get token embeddings from the input indices
        tok_emb = self.token_embedding_table(idx)

        if self.use_images and image_embeds is not None:
            # Project and concatenate image embeddings with token embeddings
            img_emb = self.image_projection(image_embeds).unsqueeze(1)
            tok_emb = torch.cat([img_emb, tok_emb], dim=1)

        # Get position embeddings
        pos_emb = self.position_embedding_table(torch.arange(tok_emb.size(1), device=device)).unsqueeze(0)

        # Add position embeddings to token embeddings
        x = tok_emb + pos_emb

        # Pass through the transformer decoder blocks
        x = self.sparseMoEBlocks(x)

        # Apply final layer normalization
        x = self.ln_f(x)

        # Get the logits from the language modeling head
        logits = self.lm_head(x)

        if targets is not None:
            if self.use_images and image_embeds is not None:
                # Prepare targets by concatenating a dummy target for the image embedding
                batch_size = idx.size(0)
                targets = torch.cat([torch.full((batch_size, 1), -100, dtype=torch.long, device=device), targets], dim=1)

            # Compute the cross-entropy loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            return logits, loss

        return logits

    def generate(self, idx, image_embeds, max_new_tokens):
        # Get the batch size and sequence length
        B, T = idx.shape

        # Initialize the generated sequence with the input indices
        generated = idx

        if self.use_images and image_embeds is not None:
            # Project and concatenate image embeddings with token embeddings
            img_emb = self.image_projection(image_embeds).unsqueeze(1)
            current_output = torch.cat([img_emb, self.token_embedding_table(idx)], dim=1)
        else:
            current_output = self.token_embedding_table(idx)

        # Generate new tokens iteratively
        for i in range(max_new_tokens):
            # Get the current sequence length
            T_current = current_output.size(1)

            # Get position embeddings for the current sequence length
            current_pos_emb = self.position_embedding_table(torch.arange(T_current, device=device)).unsqueeze(0)

            # Add position embeddings to the current output
            current_output += current_pos_emb

            # Pass through the transformer decoder blocks
            for block in self.sparseMoEBlocks:
                current_output = block(current_output)

            # Get the logits for the last token
            logits = self.lm_head(current_output[:, -1, :])

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next token based on the probabilities
            idx_next = torch.multinomial(probs, num_samples=1)

            # Concatenate the generated token to the generated sequence
            generated = torch.cat((generated, idx_next), dim=1)

            # Get the embeddings for the generated token
            idx_next_emb = self.token_embedding_table(idx_next)

            # Concatenate the generated token embeddings to the current output
            current_output = torch.cat((current_output, idx_next_emb), dim=1)

        return generated

# Using n_layer to represent number of decoder transformer blocks and n_blks for the vision encoder to avoid confusion
model = MoEDecoderLanguageModel(n_embd=128, image_embed_dim=256, vocab_size=1000, num_heads=8, n_layer=6,num_experts=8, top_k=2, use_images=True)
model.to(device)
# Dummy input
B, T = 10, 50
idx = torch.randint(0, 1000, (B, T)).to(device)
image_embeds = torch.randn(B, 256).to(device)  # Assume image_embed_dim is 256

targets = torch.randint(0, vocab_size, (B, T)).to(device)  # Only if you want to compute loss

# Test forward pass
# Check if you need to calculate loss by providing targets
if targets is not None:
    logits, loss = model(idx, image_embeds, targets)
    print(f"Logits shape: {logits.shape}, Loss: {loss}")
else:
    logits = model(idx, image_embeds)  # Call without targets
    print(f"Logits shape: {logits.shape}")

# Test generation
generated = model.generate(idx, image_embeds, max_new_tokens=20)
print(f"Generated sequence shape: {generated.shape}")

class VisionMoELanguageModel(nn.Module):
    def __init__(self, n_embd, image_embed_dim, vocab_size, n_layer, img_size, patch_size, num_heads, num_blks, emb_dropout, blk_dropout, num_experts, top_k):
        super().__init__()

        # Set num_hiddens equal to image_embed_dim
        num_hiddens = image_embed_dim

        # Assert that num_hiddens is divisible by num_heads
        assert num_hiddens % num_heads == 0, "num_hiddens must be divisible by num_heads"

        # Initialize the vision encoder (ViT)
        self.vision_encoder = ViT(img_size, patch_size, num_hiddens, num_heads, num_blks, emb_dropout, blk_dropout)

        # Initialize the language model decoder (DecoderLanguageModel)
        self.decoder = MoEDecoderLanguageModel(n_embd, image_embed_dim, vocab_size, num_heads, n_layer,num_experts, top_k, use_images=True)

    def forward(self, img_array, idx, targets=None):
        # Get the image embeddings from the vision encoder
        image_embeds = self.vision_encoder(img_array)

        # Check if the image embeddings are valid
        if image_embeds.nelement() == 0 or image_embeds.shape[1] == 0:
            raise ValueError("Something is wrong with the ViT model. It's returning an empty tensor or the embedding dimension is empty.")

        if targets is not None:
            # If targets are provided, compute the logits and loss
            logits, loss = self.decoder(idx, image_embeds, targets)
            return logits, loss
        else:
            # If targets are not provided, compute only the logits
            logits = self.decoder(idx, image_embeds)
            return logits

    def generate(self, img_array, idx, max_new_tokens):
        # Get the image embeddings from the vision encoder
        image_embeds = self.vision_encoder(img_array)

        # Check if the image embeddings are valid
        if image_embeds.nelement() == 0 or image_embeds.shape[1] == 0:
            raise ValueError("Something is wrong with the ViT model. It's returning an empty tensor or the embedding dimension is empty.")

        # Generate new tokens using the language model decoder
        generated_tokens = self.decoder.generate(idx, image_embeds, max_new_tokens)
        return generated_tokens

image_embed_dim = num_hiddens

n_layer, block_size, num_experts, top_k =  8, 32, 8, 2


# Initialize the model
model = VisionMoELanguageModel(n_embd, image_embed_dim, vocab_size,  n_layer, img_size, patch_size, n_head, num_blks, dropout, dropout, num_experts, top_k)
model.to(device)

# Create dummy data with correct dimensions
dummy_img = torch.randn(1, 3, img_size, img_size).to(device)  # Correct shape for image input
dummy_idx = torch.randint(0, vocab_size, (1, block_size)).to(device)  # Correct shape for text input

# Forward pass to initialize all parameters
try:
    output = model(dummy_img, dummy_idx)  # Output for debugging
    print("Output from initialization forward pass:", output)
except RuntimeError as e:
    print(f"Runtime Error during forward pass: {str(e)}")
    print("Check layer configurations and input shapes.")

"""Function to convert base64 encoded stringified images in inputs.csv in the repo to pytorch tensors so they can be used for training"""

def base64_to_tensor(base64_str, img_size=96):
    image = Image.open(io.BytesIO(base64.b64decode(base64_str)))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

#Adjusting the data loader from makemore for multimodal data
def get_batch(df, batch_size, split='train', img_size=96, val_batch_size=8):
    # Split data into training and validation sets
    n = int(0.9 * len(df))  # first 90% will be train, rest val
    df_train = df.iloc[:n]
    df_val = df.iloc[n:]
    data = df_train if split == 'train' else df_val
    batch_size = batch_size if split == 'train' else val_batch_size
    replace = False if split == 'train' else True
    batch = data.sample(n=batch_size, replace=replace)

    images = torch.cat([base64_to_tensor(img, img_size) for img in batch['b64string_images']], dim=0).to(device)
    text_indices = [torch.tensor(encode(desc), dtype=torch.long) for desc in batch['caption']]
    max_length = max(len(t) for t in text_indices)

    padded_text = torch.full((batch_size, max_length), fill_value=stoi['<pad>'], dtype=torch.long).to(device)
    for i, text in enumerate(text_indices):
        padded_text[i, :len(text)] = text

    targets = torch.cat([padded_text[:, 1:], torch.full((batch_size, 1), fill_value=stoi['<pad>'], dtype=torch.long, device=device)], dim=1)

    # Truncate or pad targets to match the length of padded_text
    if targets.size(1) > padded_text.size(1):
        targets = targets[:, :padded_text.size(1)]
    elif targets.size(1) < padded_text.size(1):
        targets = torch.cat([targets, torch.full((batch_size, padded_text.size(1) - targets.size(1)), fill_value=stoi['<pad>'], dtype=torch.long, device=device)], dim=1)

    return images, padded_text, targets

#Adjusting the training loop from makemore for multimodal data
def train_model(model, df, epochs, vocab_size, img_size=96):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    with mlflow.start_run():
        for epoch in range(epochs):
            model.train()
            for _ in range(max_iters):
                images, idx, targets = get_batch(df, batch_size, 'train', img_size)
                optimizer.zero_grad()
                logits, loss = model(images, idx, targets)
                loss.backward()
                optimizer.step()
                if _ % eval_interval == 0:
                    #print(f"Loss at iteration {_}: {loss.item()}")
                    metrics = {"train_loss": loss.item()}
                    mlflow.log_metrics(metrics, step=_)
            val_loss = estimate_loss(model, df, 'val', img_size, val_batch_size=8)
            metrics = {"val_loss": val_loss}
            mlflow.log_metrics(metrics, step=epoch)
            #print(f"Validation Loss after epoch {epoch}: {val_loss}")

def estimate_loss(model, df, split, img_size=96, val_batch_size=8):
    losses = []
    model.eval()
    for _ in range(eval_iters):
        images, idx, targets = get_batch(df, batch_size, split, img_size, val_batch_size=val_batch_size)
        _, loss = model(images, idx, targets)
        losses.append(loss.item())
    return sum(losses) / len(losses)

# Step 1: Upload an image
print("📤 Please upload an image file (e.g., PNG, JPEG):")
uploaded = files.upload()

# Step 2: Get uploaded image file path
if not uploaded:
    raise ValueError("❌ No image uploaded.")
image_filename = next(iter(uploaded))

# Step 3: Convert image to base64
def image_to_base64(image_bytes, img_size=96, format="PNG"):
    """Resize image to img_size x img_size and return base64 string."""
    image = Image.open(image_bytes).convert("RGB")
    image = image.resize((img_size, img_size), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

img_base64 = image_to_base64(io.BytesIO(uploaded[image_filename]))

# Step 4: Create a DataFrame with repeated base64 images + captions
num_rows = 10
df = pd.DataFrame({
    'b64string_images': [img_base64] * num_rows,
    'caption': [f"Sample caption {i+1} for testing" for i in range(num_rows)]
})

# Step 5: Expand to simulate larger dataset
df = pd.concat([df] * 30, ignore_index=True)

# Step 6: Save to CSV
csv_path = "/content/inputs.csv"
df.to_csv(csv_path, index=False)
print(f"✅ Saved CSV: {csv_path} with shape {df.shape}")

# Step 7: Create sample input.txt for tokenizer tests
text = "This is a sample text for testing the model.\nHello world!"
text_path = "/content/input.txt"
with open(text_path, 'w', encoding='utf-8') as f:
    f.write(text)
print(f"✅ Saved text file: {text_path}")

df = pd.read_csv("./inputs.csv")
#Expanding dataframe so that there's enough data to test. This is just duplicating data. A real dataset would have more rows
df = pd.concat([df])[['b64string_images', 'caption']]
df.shape

display(df.head())

batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 100
eval_interval = 10
learning_rate = 1e-3
epochs=2 # Increased epochs
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 40
num_blks= 3
head_size = 16
n_embd = 128
n_head = 8
n_layer = 8
dropout = 0.1
img_size=96
patch_size =16
image_embed_dim = 512
emb_dropout = blk_dropout =0.1
num_experts, top_k = 8, 2

def kaiming_init_weights(m):
    if isinstance (m, (nn.Linear)):
        init.kaiming_normal_(m.weight)

# Initialize the model
model = VisionMoELanguageModel(n_embd, image_embed_dim, vocab_size, n_layer, img_size, patch_size, n_head, num_blks, emb_dropout, blk_dropout, num_experts, top_k)
#Appluy thre initialization
model.apply(kaiming_init_weights)
model.to(device)

# Dummy data to initialize lazy modules
dummy_img = torch.randn(1, 3, img_size, img_size).to(device)
dummy_idx = torch.randint(0, vocab_size, (1, block_size)).to(device)
model(dummy_img, dummy_idx)  # Forward pass to initialize all parameters

# Train the model
train_model(model, df, epochs, vocab_size, img_size)

# Test generation
model.eval()
test_img = base64_to_tensor(df['b64string_images'].iloc[0], img_size).to(device)
test_idx = torch.tensor([encode("Sample")], dtype=torch.long).to(device)
generated = model.generate(test_img, test_idx, max_new_tokens=20)
print(f"Generated text: {decode(generated[0].tolist())}")

# Initialize the model
model = VisionMoELanguageModel(n_embd, image_embed_dim, vocab_size, n_layer, img_size, patch_size, n_head, num_blks, emb_dropout, blk_dropout, num_experts, top_k)
#Appluy thre initialization
model.apply(kaiming_init_weights)
model.to(device)

# Dummy data to initialize lazy modules
dummy_img = torch.randn(1, 3, img_size, img_size).to(device)
dummy_idx = torch.randint(0, vocab_size, (1, block_size)).to(device)
model(dummy_img, dummy_idx)  # Forward pass to initialize all parameters

# Train the model
train_model(model, df, epochs, vocab_size, img_size)

# Initialize the model
model = VisionMoELanguageModel(n_embd, image_embed_dim, vocab_size, n_layer, img_size, patch_size, n_head, num_blks, emb_dropout, blk_dropout, num_experts, top_k)
#Appluy thre initialization
model.apply(kaiming_init_weights)
model.to(device)

# Dummy data to initialize lazy modules
dummy_img = torch.randn(1, 3, img_size, img_size).to(device)
dummy_idx = torch.randint(0, vocab_size, (1, block_size)).to(device)
model(dummy_img, dummy_idx)  # Forward pass to initialize all parameters

# Train the model
train_model(model, df, epochs, vocab_size, img_size)

# Test generation using the uploaded image
model.eval()
# The uploaded image is stored in the 'uploaded' variable from the previous upload step
image_filename = next(iter(uploaded))
test_img_bytes = io.BytesIO(uploaded[image_filename])
test_img = base64_to_tensor(image_to_base64(test_img_bytes, img_size), img_size).to(device)

# Start generation with a prompt
prompt = "A photo of"
test_idx = torch.tensor([encode(prompt)], dtype=torch.long).to(device)

generated = model.generate(test_img, test_idx, max_new_tokens=50)
print(f"Generated text: {decode(generated[0].tolist())}")

!pip install wandb -qq

import wandb

# Log in to your wandb account
# You will be prompted to enter your API key
wandb.login()

#Adjusting the training loop from makemore for multimodal data
def train_model(model, df, epochs, vocab_size, img_size=96):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for iter in range(max_iters): 
            images, idx, targets = get_batch(df, batch_size, 'train', img_size)
            optimizer.zero_grad()
            logits, loss = model(images, idx, targets)
            loss.backward()
            optimizer.step()

            if iter % eval_interval == 0:
                train_loss = loss.item()
                val_loss = estimate_loss(model, df, 'val', img_size, val_batch_size=8)
                print(f"Epoch {epoch+1}, Iter {iter}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                # Log metrics to wandb
                wandb.log({"train_loss": train_loss, "val_loss": val_loss}, step=epoch * max_iters + iter)

        # Save model checkpoint at the end of each epoch
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")
        wandb.save(f"model_epoch_{epoch+1}.pt")
        print(f"Checkpoint saved for epoch {epoch+1}")

def estimate_loss(model, df, split, img_size=96, val_batch_size=8):
    losses = []
    model.eval()
    with torch.no_grad(): # Use no_grad for evaluation
        for _ in range(eval_iters):
            images, idx, targets = get_batch(df, batch_size, split, img_size, val_batch_size=val_batch_size)
            _, loss = model(images, idx, targets)
            losses.append(loss.item())
    return sum(losses) / len(losses)

# Initialize a wandb run outside the training loop
run = wandb.init(project="vision-moe-multimodal", config={
    "epochs": epochs,
    "batch_size": batch_size,
    "block_size": block_size,
    "learning_rate": learning_rate,
    "eval_interval": eval_interval,
    "eval_iters": eval_iters,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "dropout": dropout,
    "img_size": img_size,
    "patch_size": patch_size,
    "image_embed_dim": image_embed_dim,
    "emb_dropout": emb_dropout,
    "blk_dropout": blk_dropout,
    "num_experts": num_experts,
    "top_k": top_k,
    "num_blks": num_blks,
    "vocab_size": vocab_size,
})

# Initialize and train the model
model = VisionMoELanguageModel(n_embd, image_embed_dim, vocab_size, n_layer, img_size, patch_size, n_head, num_blks, emb_dropout, blk_dropout, num_experts, top_k)
model.apply(kaiming_init_weights)
model.to(device)
train_model(model, df, epochs, vocab_size, img_size)

# Finish the wandb run
wandb.finish()


model_save_path = "vision_moe_model.pth"
torch.save(model.state_dict(), model_save_path)

## Install hugging face libraries

!pip install transformers huggingface_hub -qq

## Log in to hugging face

from huggingface_hub import notebook_login

notebook_login()

## Create a hugging face repository

from huggingface_hub import HfApi

api = HfApi()
repo_id = "Aptheos/SparseFusion"
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
print(f"Repository '{repo_id}' created or confirmed on Hugging Face Hub.")

## Upload model files

from huggingface_hub import HfApi, upload_file

api = HfApi()
repo_id = "Aptheos/SparseFusion"

# Upload the model state dictionary file
model_path = "vision_moe_model.pth"
upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=repo_id,
    repo_type="model",
)

print(f"Successfully uploaded {model_path} to {repo_id}")

# There might be need to upload vocabulary files if they are not part of the model state dict
# If stoi and itos are saved:
# upload_file(
#     path_or_fileobj="stoi.pkl", # If stoi is saved as a pickle file
#     path_in_repo="stoi.pkl",
#     repo_id=repo_id,
#     repo_type="model",
# )
# upload_file(
#     path_or_fileobj="itos.pkl",
#     path_in_repo="itos.pkl",
#     repo_id=repo_id,
#     repo_type="model",
# )