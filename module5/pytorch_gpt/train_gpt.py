
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
import nltk
import re
from itertools import islice
import random
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoTokenizer

from gpt.decoder import DecoderOnlyTransformer
from gpt.position_encoder import PositionalEncoding

nltk.download('punkt_tab')

embed_dim = 150
max_len = 75
num_transformers = 6
num_heads = 5
dense_dim = 256
PAD_TOKEN_ID = 0

class NeuralNetwork(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        embed_dim = embed_dim,
        num_transformers = num_transformers,
        num_heads = num_heads,
        dense_dim = dense_dim,
        pad_token_id = PAD_TOKEN_ID
    ):
        super().__init__()
        self.token_embed = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embed_dim,
            padding_idx = pad_token_id,
        )
        self.position_encoding = PositionalEncoding(
            embed_dim = embed_dim,
            max_len = max_len,
        )
        self.transformer_stack = nn.ModuleList([
            DecoderOnlyTransformer(
                embed_dim = embed_dim,
                num_heads = num_heads,
                dense_dim = dense_dim,
        ) for _ in range(num_transformers)])
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(
            in_features = embed_dim,
            out_features = vocab_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        key_padding_mask = (x == PAD_TOKEN_ID)
        x = self.token_embed(x)
        x = self.position_encoding(x)
        for transformer in self.transformer_stack:
            x = transformer(x, key_padding_mask = key_padding_mask)
        x = self.layer_norm(x)
        x = self.linear(x)
        return x #loss will be computed from logits for stability


wiki_dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split='train',streaming=True)
shuffled_stream = wiki_dataset.shuffle(seed=42, buffer_size=100)

TRAIN_SIZE = 100000
TEST_SIZE = 100
batch_size = 256

def clean_wiki_text(text):
    # removes wikipedia headers
    text = re.sub(r'={2,}.*?={2,}', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = ' '.join(text.split())
    return text

class WikiSentenceDataset(IterableDataset):
    def __init__(self, hf_dataset):
        super().__init__()
        self.hf_dataset = hf_dataset
    
    def __iter__(self):
        for example in self.hf_dataset:
            clean_text = clean_wiki_text(example['text'])
            if clean_text:
                sentences = nltk.sent_tokenize(clean_text)
                for sentence in sentences:
                    yield sentence
                    
class ShufflingIterableDataset(IterableDataset):
    def __init__(self, source_dataset, buffer_size, seed):
        super().__init__()
        self.source_dataset = source_dataset
        self.buffer_size = buffer_size
        self.seed = seed
    
    def __iter__(self):
        rng = random.Random(self.seed)
        source_iterator = iter(self.source_dataset)
        shuffle_buffer = list(islice(source_iterator, self.buffer_size))
        # first loop replaces given item with source_iterator addition
        for item in source_iterator:
            idx = rng.randint(0, self.buffer_size-1)
            yield shuffle_buffer[idx]
            shuffle_buffer[idx] = item
        # second loop flushes the current buffer when source_iterator is dry
        rng.shuffle(shuffle_buffer)
        for item in shuffle_buffer:
            yield item
            
            
train_stream = shuffled_stream.take(TRAIN_SIZE)
test_stream = shuffled_stream.skip(TRAIN_SIZE).take(TEST_SIZE)

train_stream = WikiSentenceDataset(train_stream)
test_stream = WikiSentenceDataset(test_stream)

train_dataset = ShufflingIterableDataset(train_stream, buffer_size=1000, seed=42)
test_dataset = ShufflingIterableDataset(test_stream, buffer_size=1000, seed=42)

data_loader_plain = DataLoader(
    train_dataset,
    batch_size = 4
)

for i, batch in enumerate(data_loader_plain):
    print(f"Sentences from batch {i}:")
    for j, sentence in enumerate(batch):
        print(f"  - Item {j}: '{sentence}'")
        
    if i>1:
        break
            

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size

collate_fn = lambda sentences: tokenizer(
        sentences,
        padding='max_length',   
        truncation = True,
        max_length = max_len,
        return_tensors="pt"
    )

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn = collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=collate_fn
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = NeuralNetwork(vocab_size=vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

epochs = 20

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    train_loader_size = 0
    for batch in tqdm(train_loader, desc="Training"):
        train_loader_size += 1
        b = batch['input_ids']
        b = b.to(device)
        inputs = b[:,:-1]
        targets = b[:,1:]
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits.reshape(-1,vocab_size), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / train_loader_size
    print(f"Average Training Loss: {avg_train_loss:.4f}")
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        test_loader_size = 0
        for batch in tqdm(test_loader, desc="Validation"):
            b = batch['input_ids']
            test_loader_size += 1
            b = b.to(device)
            inputs = b[:,:-1]
            targets = b[:,1:]
            logits = model(inputs)
            loss = loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / test_loader_size
    print(f"Average Validation Loss: {avg_val_loss:.4f}")
print("training complete")

# --- ADD THIS CODE FOR EXPORTING ---

import os
import json

# 1. Define a directory to save everything
SAVE_DIR = "my_trained_gpt_model"
os.makedirs(SAVE_DIR, exist_ok=True)

# 2. Save the model's state dictionary
model_path = os.path.join(SAVE_DIR, "model_state_dict.pth")
torch.save(model.state_dict(), model_path)
print(f"Model state_dict saved to {model_path}")

# 3. Save the tokenizer
tokenizer.save_pretrained(SAVE_DIR)
print(f"Tokenizer saved to {SAVE_DIR}")

# 4. Save the model's configuration/hyperparameters
config = {
    "vocab_size": vocab_size,
    "embed_dim": embed_dim,
    "max_len": max_len,
    "num_transformers": num_transformers,
    "num_heads": num_heads,
    "dense_dim": dense_dim,
    "pad_token_id": PAD_TOKEN_ID
}
config_path = os.path.join(SAVE_DIR, "config.json")
with open(config_path, 'w') as f:
    json.dump(config, f)
print(f"Model config saved to {config_path}")

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.1):
    model.eval()
    input_ids_list = tokenizer.encode(prompt, truncation=True, max_length=max_length - 1)
    
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id
    
    device = next(model.parameters()).device
    generated_ids = torch.tensor([input_ids_list], device=device, dtype=torch.long)
    
    with torch.no_grad():
        for _ in range(max_length - len(input_ids_list)):
            current_len = generated_ids.size(1)
            padded_input = torch.full((1, max_len), pad_token_id, device=device, dtype=torch.long)
            padded_input[:, :current_len] = generated_ids
            
            logits = model(padded_input)

            next_token_logits = logits[:, current_len - 1, :]
            
            scaled_logits = next_token_logits / temperature
            
            probabilities = F.softmax(scaled_logits, dim=-1)
            next_token_id = torch.multinomial(probabilities, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            if eos_token_id and next_token_id.item() == eos_token_id:
                break
    generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    
    return generated_text

prompt = "The most prominent figure of the 20th century is"
print("Starting generation...")
print(f"Prompt: {prompt}\n")

generated_paragraph = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_length = 50,
    temperature = .5
)

print(f"Output: {generated_paragraph}")

prompt = "In 2001,"
print("Starting generation...")
print(f"Prompt: {prompt}\n")

generated_paragraph = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_length = 50,
    temperature = .5
)

print(f"Output: {generated_paragraph}")