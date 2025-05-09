import torch
# from transformers import AutoTokenizer  # If you used a standard tokenizer
from train import BasicLLM
import numpy
import os
from pdfReader import ALL_DATA_PATH
# import torchtext
# from torchtext.data import get_tokenizer
 
VOCAB_PATH = 'vocab_size.txt'
vocab_size = 0  # Replace with your actual vocabulary size
assert os.path.isfile(VOCAB_PATH)    

with open(VOCAB_PATH, encoding='utf-8') as outfile:
    vocab_size = int(outfile.read())

print("vocab_size : ",vocab_size)
# --- 2. Define Hyperparameters (MUST MATCH TRAINING) ---
embedding_dim = 128
hidden_dim = 256
num_layers = 2
num_heads = 4
block_size = 128
dropout = 0.1
batch_size = 8
sequence_length = 64
max_new_tokens = 50
learning_rate = 0.001
num_epochs = 20

# --- 3. Instantiate the Model ---
model = BasicLLM(vocab_size, embedding_dim, hidden_dim, num_layers, num_heads, block_size, dropout)
 
# --- 4. Load the Saved State Dictionary (.pth file) ---
path_to_saved_model = 'basic_llm_trained_on_pdfs.pth'  # Replace with the actual path to your .pth file
model.load_state_dict(torch.load(path_to_saved_model))
 
# --- 5. Set to Evaluation Mode ---
model.eval()
print(f"Pre-trained model loaded from '{path_to_saved_model}' and in evaluation mode.")
 
# --- 6. Load the Tokenizer (Crucial for input encoding) ---
# If you used a standard tokenizer from Hugging Face Transformers:
#tokenizer_name = 'gpt2'  # Replace with the tokenizer you used during training (if applicable)
# try:
#tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
# except Exception as e:
#     tokenizer = None
#     print(f"Warning: Could not load tokenizer '{tokenizer_name}'. Ensure it's correct or handle tokenization manually.")
 
# If you used a custom tokenizer (e.g., character-level):
# tokens = [...] # Your list of unique tokens
# stoi = {ch: i for i, ch in enumerate(tokens)}
# encode = lambda s: [stoi[c] for c in s]
# decode = lambda l: ''.join([tokens[i] for i, _ in enumerate(l)]) # Corrected decode
 
# --- 7. Prepare Input Prompt ---
prompt_text =  "help"

assert os.path.isfile(ALL_DATA_PATH)    
all_text = "" # Replace with actual loading
with open(ALL_DATA_PATH, encoding='utf-8') as file:
    all_text = file.read()

# --- 2. Tokenization and Vocabulary Creation () ---
all_tokens = sorted(list(set(all_text)))
 

tokenizer = None
if tokenizer:
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
else:
    # Manual encoding if you used a custom tokenizer
    tokens = all_tokens # Your vocabulary list
    stoi = {ch: i for i, ch in enumerate(tokens)}
    encode = lambda s: [stoi.get(c, stoi.get('<unk>', 0)) for c in s] # Handle unknown tokens
    input_ids = torch.tensor([encode(prompt_text)], dtype=torch.long)
 
# --- 8. Generate Information (Text Generation) ---
max_new_tokens = 100
 
with torch.no_grad():
    generated_output = model.generate(input_ids, max_new_tokens=max_new_tokens)
 
# --- 9. Decode the Output to Get Information ---
if tokenizer:
    generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
    print(f"Generated information: '{generated_text}'")
else:
    # Manual decoding if you used a custom tokenizer
    # tokens = all_tokens # Your vocabulary list
    itos = {i: ch for ch, i in stoi.items()}
    decode = lambda l: ''.join([itos.get(i, '<unk>') for i in l])
    generated_text = decode(generated_output[0].tolist())
    print(f"Generated information: '{generated_text}'")