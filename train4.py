import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import math
import time
from tqdm import tqdm

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len):
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, None, :, :]

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, freqs):
    q_rot = (q * freqs.cos()) + (rotate_half(q) * freqs.sin())
    k_rot = (k * freqs.cos()) + (rotate_half(k) * freqs.sin())
    return q_rot, k_rot

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

class SwiGLUAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.swiglu = SwiGLU(dim, dim * 2, dim)
        self.rotary_emb = RotaryEmbedding(head_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        freqs = self.rotary_emb(N)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, freqs)

        attn = (q_rot @ k_rot.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return self.swiglu(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SwiGLUAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SwiGLU(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., vocab_size=10000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_ratio, qkv_bias, drop, attn_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)

class TextDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.data[index:index+self.seq_length]),
            torch.tensor(self.data[index+1:index+self.seq_length+1])
        )

class Vocabulary:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.word_count = Counter()
        self.total_words = 0
        self.unknown_token = '<unk>'
        self.add_word(self.unknown_token)

    def add_word(self, word):
        if word not in self.word2index:
            index = len(self.word2index)
            self.word2index[word] = index
            self.index2word[index] = word
        self.word_count[word] += 1
        self.total_words += 1

    def __len__(self):
        return len(self.word2index)

def get_data(file_path, vocab_size):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    words = text.split()
    vocab = Vocabulary()
    for word in words:
        vocab.add_word(word)

    common_words = dict(vocab.word_count.most_common(vocab_size - 1))
    new_vocab = Vocabulary()
    for word in common_words:
        new_vocab.add_word(word)

    data = [new_vocab.word2index.get(word, new_vocab.word2index[new_vocab.unknown_token]) for word in words]

    train_data = data[:int(0.8 * len(data))]
    val_data = data[int(0.8 * len(data)):int(0.9 * len(data))]
    test_data = data[int(0.9 * len(data)):]

    return train_data, val_data, test_data, new_vocab

def train(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (batch + 1)})

        val_loss = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start_time
        print(f'Epoch {epoch+1} | Time: {elapsed:.2f}s | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            total_loss += criterion(output.view(-1, output.size(-1)), targets.view(-1)).item()
    return total_loss / len(data_loader)

def generate_text(model, vocab, seq_length, device, num_words=100):
    model.eval()
    context = torch.tensor([vocab.word2index['<unk>']] * seq_length, device=device).unsqueeze(0)
    generated_words = []
    with torch.no_grad():
        for _ in range(num_words):
            output = model(context)
            word_index = output[0, -1].argmax().item()
            generated_words.append(vocab.index2word[word_index])
            context = torch.cat([context[:, 1:], torch.tensor([[word_index]], device=device)], dim=1)
    return ' '.join(generated_words)

def main():
    # Hyperparameters
    file_path = 'alice_in_wonderland.txt'  # Replace with your file path
    vocab_size = 1000
    batch_size = 4
    seq_length = 25
    dim = 128
    depth = 3
    heads = 4
    mlp_ratio = 4.0
    learning_rate = 0.0001
    epochs = 1

    # Get data and create dataloaders
    train_data, val_data, test_data, vocab = get_data(file_path, vocab_size)
    train_dataset = TextDataset(train_data, seq_length)
    val_dataset = TextDataset(val_data, seq_length)
    test_dataset = TextDataset(test_data, seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model, loss function, and optimizer
    model = Transformer(dim, depth, heads, mlp_ratio, vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(model, train_loader, val_loader, criterion, optimizer, device, epochs)

    # Evaluate on test set
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}')

    # Generate some text
    generated_text = generate_text(model, vocab, seq_length, device)
    print("Generated text:")
    print(generated_text)

if __name__ == "__main__":
    main()
