import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

batch_size = 64
learning_rate = 1e-4
num_epochs = 10
vocab_size = 80 # Define your vocabulary size
embedding_dim = 256  # Define the dimension of word embeddings
hidden_dim = 512  # Define the dimension of hidden layers
num_layers = 2  # Number of recurrent layers if using RNN or transformer layers if using a transformer

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("ı],ᴜ ſ͕ɭᴜ j͑ʃᴜꞇ ꞁȷ̀ɔ j͑ʃƽɔƽ")

# Read data from a text file
with open("ſɭɔʞ.txt", "r", encoding="utf-8") as file:
    text_data = file.read()

# Preprocess the data
max_length = 712
tokens = tokenizer(
    text_data,
    max_length=max_length,
    padding="max_length",
    truncation=True
)
input_ids = tokens["input_ids"]
attention_mask = tokens["attention_mask"]

# Split data into training, validation, and test sets
train_size = int(0.8 * len(input_ids))
val_size = int(0.1 * len(input_ids))
test_size = len(input_ids) - train_size - val_size

train_data = input_ids[:train_size]
val_data = input_ids[train_size:train_size + val_size]
test_data = input_ids[train_size + val_size:]

# Check if train_data is not empty
if len(train_data) > 0:
    # Create a custom dataset class
    class CustomDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # Create a DataLoader instance for training
    train_dataset = CustomDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Example: Iterate through the DataLoader
    for batch in train_loader:
        print(batch)
else:
    print("train_data is empty. Please check your data.")

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Create DataLoader instances for training, validation, and test sets
train_dataset = CustomDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

val_dataset = CustomDataset(val_data)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define your custom language model architecture
class CustomLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(CustomLanguageModel, self).__init__()
        
        # Define embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Define recurrent layers (you can also use transformer layers here)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        
        # Define output layer (e.g., for language modeling, it could be a linear layer with vocab_size output)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # Embedding layer
        embedded = self.embedding(x)
        
        # RNN layers
        output, _ = self.rnn(embedded)
        
        # Output layer
        output = self.linear(output)
        return output

# Initialize the model
model = CustomLanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        input_ids = batch

        # Forward pass
        outputs = model(input_ids)

        # Reshape for the loss function
        outputs = outputs.view(-1, vocab_size)
        input_ids = input_ids.view(-1)

        # Calculate loss
        loss = criterion(outputs, input_ids)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation code
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for val_batch in val_loader:
            val_input_ids = val_batch
            val_outputs = model(val_input_ids)
            val_outputs = val_outputs.view(-1, vocab_size)
            val_input_ids = val_input_ids.view(-1)
            val_loss += criterion(val_outputs, val_input_ids).item()
        val_loss /= len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.4f} Validation Loss: {val_loss:.4f}")

    # Checkpointing code (save the model if validation loss decreases)
    if epoch == 0:
        best_val_loss = val_loss
    elif val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "ꞁȷ̀ɜ j͐ʃɹ ŋᷠꞇ j͑ʃᴜꞇ ᶅſɔƴ.pt")

# Save the final trained model
final_model_path = "ᶅſɔⅎ.pt"
torch.save(model.state_dict(), final_model_path)