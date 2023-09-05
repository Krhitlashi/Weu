import json
import torch
from transformers import GPT2Tokenizer
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Load the custom vocabulary from the JSON file
with open("ſȷᴜͷ̗ ſɭɔʞ ꞁȷ̀ᴜꞇ.json", "r", encoding="utf-8") as vocab_file:
    custom_vocab = json.load(vocab_file)

# Define your custom tokenizer function
def custom_tokenizer(text):
    # Tokenize the text using your custom logic
    tokens = word_tokenize(text)
    # Map tokens to their IDs based on your custom vocabulary
    token_ids = [custom_vocab.get(token, custom_vocab["<ſɭſɭſ͕ȷ>"]) for token in tokens]
    return token_ids

# Example text
file_path = "ſɭɔʞ.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# Ensure that all tokens in your text are in the custom vocabulary with unique IDs
custom_tokens = text.split()  # Split the text into tokens
for token in custom_tokens:
    if token not in custom_vocab:
        custom_vocab[token] = len(custom_vocab)

# Tokenize the text
custom_tokens = custom_tokenizer(text)

# Add special tokens programmatically
special_tokens = ["<ɽ͑ʃ'ſ͕ȷƽ>", "<j͑ʃı],>", "<ſ̀ȷſɭſɭ>", "<ſɭſɭſ͕ȷ>", "<ſɭɘſ͕ȷƽ>"]
for special_token in special_tokens:
    if special_token not in custom_vocab:
        custom_vocab[special_token] = len(custom_vocab)

# Split data into training, validation, and test sets
train_size = int(0.8 * len(custom_tokens))
val_size = int(0.1 * len(custom_tokens))
test_size = len(custom_tokens) - train_size - val_size

train_data = custom_tokens[:train_size]
val_data = custom_tokens[train_size:train_size + val_size]
test_data = custom_tokens[train_size + val_size:]

# Initialize the custom tokenizer with the loaded vocabulary
tokenizer = GPT2Tokenizer(vocab_file="ſȷᴜͷ̗ ſɭɔʞ ꞁȷ̀ᴜꞇ.json", merges_file="ſɭɔʞ ſɟɹƽ ꞁȷ̀ᴜꞇ.txt")

tokenizer.pad_token = "<ɽ͑ʃ'ſ͕ȷƽ>"
tokenizer.bos_token = "<j͑ʃı],>"
tokenizer.eos_token = "<ſ̀ȷſɭſɭ>"
tokenizer.unk_token = "<ſɭſɭſ͕ȷ>"
tokenizer.space_token = "<ſɭɘſ͕ȷƽ>"

# Save the custom tokenizer to a directory
tokenizer.save_pretrained("ı],ᴜ ſ͕ɭᴜ j͑ʃᴜꞇ ꞁȷ̀ɔ j͑ʃƽɔƽ")

# Print the results
print("Tokenized Text:", custom_tokens)

# Test
test_text = custom_tokens
encoded = tokenizer.encode(test_text, add_special_tokens=True, return_tensors='pt')

# 1. Convert the encoded tensor to a list of token IDs
encoded_list = encoded.tolist()[0]

# 2. Create a reverse mapping of token IDs to tokens in your custom vocabulary
reverse_vocab = {v: k for k, v in custom_vocab.items()}

# 3. Use the reverse mapping to convert token IDs back to tokens
decoded_tokens = [reverse_vocab[token_id] for token_id in encoded_list]

# 4. Join the tokens together to form the decoded text
decoded_text = ' '.join(decoded_tokens)

# Print the decoded text
print("Decoded Text:", decoded_text)
