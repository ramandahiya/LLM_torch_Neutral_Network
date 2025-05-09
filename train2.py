import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import random
import os
from pdfReader import ALL_DATA_PATH


# 1. Define a Very Simple Dataset
class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._prepare_data()
 
    def _prepare_data(self):
        data = []
        for text in self.texts:
            token_ids = self.tokenizer.encode(text)
            if len(token_ids) > 1:  # Need at least one input and one target
                for i in range(1, len(token_ids)):
                    input_ids = token_ids[:i]
                    target_id = token_ids[i]
                    if len(input_ids) <= self.max_length:
                        padding_length = self.max_length - len(input_ids)
                        padded_input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                        data.append((torch.tensor(padded_input_ids), torch.tensor(target_id)))
        return data
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        return self.data[idx]
 
# 2. Define a Very Basic LLM Architecture (Simple RNN)
class BasicLLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, pad_token_id):
        super(BasicLLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
 
    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        output, _ = self.rnn(embedded)
        # Only predict the next token based on the last hidden state of the sequence
        prediction = self.fc(output[:, -1, :])
        return prediction
 
# 3. Implement a Basic Tokenizer (Character-Level)
class BasicTokenizer:
    def __init__(self, texts, max_vocab_size=50):
        all_chars = set()
        for text in texts:
            all_chars.update(list(text))
        self.unique_chars = sorted(list(all_chars))
        self.vocab = ['<pad>', '<unk>'] + self.unique_chars[:max_vocab_size - 2]
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.pad_token_id = self.char_to_id['<pad>']
        self.unk_token_id = self.char_to_id['<unk>']
 
    def encode(self, text):
        return [self.char_to_id.get(char, self.unk_token_id) for char in text]
 
    def decode(self, ids):
        return ''.join([self.id_to_char.get(idx, self.unk_token) for idx in ids])
 
    def __len__(self):
        return len(self.vocab)
 
# 4. Prepare Training Data
train_texts = [
    "hello world",
    "how are you",
    "i am fine thank you",
    "what is your name",
    "my name is basicllm",
    "the quick brown fox",
    "jumps over the lazy dog",
    "python is fun",
    "learning is great",
    "this is a simple example"
]

#assert os.path.isfile(ALL_DATA_PATH)    
#train_texts = "This is some example text from a PDF. Another sentence here. And yet another one." # Replace with actual loading
# with open(ALL_DATA_PATH, encoding='utf-8') as file:
#      train_texts = file.read()


 
# Initialize Tokenizer
tokenizer = BasicTokenizer(train_texts)
vocab_size = len(tokenizer)
max_length = 10
 
# Create Dataset and DataLoader
train_dataset = SimpleTextDataset(train_texts, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
 
# 5. Initialize Model, Loss Function, and Optimizer
embedding_dim = 10
hidden_dim = 20
num_layers = 1
pad_token_id = tokenizer.pad_token_id
model = BasicLLM(vocab_size, embedding_dim, hidden_dim, num_layers, pad_token_id)
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=0.01)
 
# 6. Train the Model (Very Few Epochs for Demonstration)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_inputs, batch_targets in train_dataloader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader):.4f}")
 
# 7. Generate Output
def generate_text(model, tokenizer, start_text="h", max_length=20):
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(start_text)).unsqueeze(0)
    generated_ids = input_ids.tolist()[0]
 
    with torch.no_grad():
        for _ in range(max_length - len(start_text)):
            predictions = model(torch.tensor(input_ids).to(torch.long))
            predicted_id = torch.argmax(predictions, dim=-1).item()
            generated_ids.append(predicted_id)
            input_ids = torch.tensor([generated_ids]).to(torch.long)
    return tokenizer.decode(generated_ids)
 
print("\n--- Generated Output ---")
generated_output = generate_text(model, tokenizer, start_text="h")
print(f"Generated text starting with 'h': '{generated_output}'")
generated_output = generate_text(model, tokenizer, start_text="w")
print(f"Generated text starting with 'w': '{generated_output}'")
generated_output = generate_text(model, tokenizer, start_text="t")
print(f"Generated text starting with 't': '{generated_output}'")
 
# 8. Accuracy (Very Rough Estimate on Training Data)
def calculate_accuracy(model, dataloader, tokenizer):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in dataloader:
            outputs = model(batch_inputs)
            predicted_ids = torch.argmax(outputs, dim=-1)
            total_predictions += batch_targets.size(0)
            correct_predictions += (predicted_ids == batch_targets).sum().item()
    return correct_predictions / total_predictions if total_predictions > 0 else 0
 
accuracy = calculate_accuracy(model, train_dataloader, tokenizer)
print(f"\n--- Rough Accuracy on Training Data ---")
print(f"Accuracy: {accuracy:.4f}")
 