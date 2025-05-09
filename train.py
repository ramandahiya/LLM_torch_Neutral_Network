import torch
from torch import nn
from torch.nn import functional as F
import os
from pdfReader import ALL_DATA_PATH

# --- Import your PDF processing and data loading functions here ---
# from your_pdf_processing_module import load_and_preprocess_pdfs
 
class BasicLLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_heads, block_size, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, block_size, dropout) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(embedding_dim, vocab_size)
        self.block_size = block_size
        self.apply(self._init_weights)
 
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
 
    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embeddings = self.embedding(idx)
        position_embeddings = nn.Parameter(torch.zeros(1, self.block_size, token_embeddings.shape[-1]))
        position_embeddings = position_embeddings[:, :T, :]
        x = token_embeddings + position_embeddings
        for block in self.transformer:
            x = block(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
 
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
 
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, block_size, dropout):
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads
        self.all_head_size = self.num_heads * self.head_size
        self.query = nn.Linear(embedding_dim, self.all_head_size)
        self.key = nn.Linear(embedding_dim, self.all_head_size)
        self.value = nn.Linear(embedding_dim, self.all_head_size)
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
 
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.all_head_size)
        y = self.resid_dropout(self.proj(y))
        return y
 
class FeedForward(nn.Module):
    def __init__(self, embedding_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
               nn.Linear(embedding_dim, 4 * embedding_dim),
               nn.GELU(),
               nn.Linear(4 * embedding_dim, embedding_dim),
               nn.Dropout(dropout),
            )
 
    def forward(self, x):
        return self.net(x)
 
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attn = MultiHeadAttention(embedding_dim, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ffwd = FeedForward(embedding_dim, dropout)
 
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
 
if __name__ == '__main__':
    # --- Hyperparameters ---
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2
    num_heads = 4
    block_size = 128
    dropout = 0
    batch_size = 8
    sequence_length = 64
    max_new_tokens = 50
    learning_rate = 0.001
    num_epochs = 20
 
    # -- Load and Preprocess PDFs (Placeholder) ---
    #       You'll need to implement a function here to:
    #   - Iterate through your PDF files.
    #   - Extract text content.
    #   - Clean the text.
    # all_text = load_and_preprocess_pdfs("path/to/your/pdfs")

    assert os.path.isfile(ALL_DATA_PATH)
    
    all_text = "This is some example text from a PDF. Another sentence here. And yet another one." # Replace with actual loading

    with open(ALL_DATA_PATH, encoding='utf-8') as file:
        all_text = file.read()


 
    # --- 2. Tokenization and Vocabulary Creation (Basic Example) ---
    tokens = sorted(list(set(all_text)))
    vocab_size = len(tokens)
    stoi = {ch: i for i, ch in enumerate(tokens)}
    itos = {i: ch for i, ch in enumerate(tokens)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
 
    print(f"Vocabulary size: {vocab_size}")
    encoded_text = encode(all_text)
 
    # --- 3. Prepare Training Data (Basic Example) ---
    def get_batch(data, block_size, batch_size):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.tensor(data[i:i+block_size], dtype=torch.long) for i in ix])
        y = torch.stack([torch.tensor(data[i+1:i+block_size+1], dtype=torch.long) for i in ix])
        return x, y
 
    train_data = encoded_text
    train_dataset_size = len(train_data)
    print(f"Training dataset size: {train_dataset_size}")
 
    # --- 4. Instantiate the Model ---
    model = BasicLLM(vocab_size, embedding_dim, hidden_dim, num_layers, num_heads, block_size, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
 
    # --- 5. Training Loop (Simplified) ---
    for epoch in range(num_epochs):
        for step in range(train_dataset_size // (batch_size * block_size)): # Basic iteration
            xb, yb = get_batch(train_data, block_size, batch_size)
            logits, loss = model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss.item():.4f}")
 
    # --- 6. Generation (Example) ---
    context = torch.zeros((1, 1), dtype=torch.long) # Start with a single zero token
    generated_output = model.generate(context, max_new_tokens=max_new_tokens)
    print("Generated text:", decode(generated_output[0].tolist()))
    
    # --- 7. Saving the Model (Optional) ---
    torch.save(model.state_dict(), "basic_llm_trained_on_pdfs.pth")

    print("basic_llm_trained_on_pdfs.pth Generated :")
    
    with open('vocab_size.txt', 'w', encoding='utf-8') as outfile:
        outfile.write(str(vocab_size))