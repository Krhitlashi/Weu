import json
import torch
from transformers import GPT2Tokenizer
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Load the custom vocabulary from the JSON file
with open("j͑ʃɹ ſȷɜⅎ ſȷᴜͷ̗.json", "r", encoding="utf-8") as vocab_file:
    custom_vocab = json.load(vocab_file)

# Define your custom tokenizer function
def custom_tokenizer(text):
    # Tokenize the text using spaces
    tokens = text.split()
    # Map tokens to their IDs based on your custom vocabulary
    token_ids = [custom_vocab.get(token, custom_vocab["<ſɭſɭſ͕ȷ>"]) for token in tokens]
    return token_ids

# Add special tokens programmatically
special_tokens = ["<ɽ͑ʃ'ſ͕ȷƽ>", "<j͑ʃı],>", "<ſ̀ȷſɭſɭ>", "<ſɭſɭſ͕ȷ>", "<ſɭɘſ͕ȷƽ>"]
for special_token in special_tokens:
    if special_token not in custom_vocab:
        custom_vocab[special_token] = len(custom_vocab)

# Define a function to tokenize and process text from a file
def tokenize_and_process_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    
    # Ensure that all tokens in the text are in the custom vocabulary with unique IDs
    def split_tokens(text):
        custom_tokens = text.split()  # Split the text into tokens
        for token in custom_tokens:
            if token not in custom_vocab:
                custom_vocab[token] = len(custom_vocab)
    
    split_tokens(text)

    # Tokenize the text
    custom_tokens = custom_tokenizer(text)

    return custom_tokens

# Example text file paths
file_paths = [
    "ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ\ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ.txt",
    "ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ\ꞁȷ̀ꞇ }ʃᴜƽ ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ.txt",
    "ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ\ı],ɹ ŋᷠɔ ſɭᴜꞇ }ʃɔ.txt",
    "ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ\ſɭɔ ſȷɜⅎ.txt",
    "ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ\j͑ʃп́ ꞁȷ̀ꞇ }ʃᴜƽ.txt",
    "ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ\j͑ʃƽᴜ ſɭɔʞ\ſɭɹ j͑ʃᴜ ŋᷠꞇ ɭʃᴜƴ ſɭɹ j͑ʃᴜ ŋᷠꞇ ɭʃᴜƴ ſɭɹ j͑ʃᴜ ŋᷠꞇ ɭʃᴜƴ.txt",
    "ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ\j͑ʃƽᴜ ſɭɔʞ\}ʃɔ ֭ſɭᴜ ı]ɹ ⺓ ſᶘᴜƴ ꞁȷ̀ᴜ }ʃꞇ.txt",
]

all_custom_tokens = []

# Process each text file
for file_path in file_paths:
    custom_tokens = tokenize_and_process_file(file_path)
    all_custom_tokens.extend(custom_tokens)

# Split data into training, validation, and test sets
train_size = int(0.8 * len(all_custom_tokens))
val_size = int(0.1 * len(all_custom_tokens))
test_size = len(all_custom_tokens) - train_size - val_size

train_data = all_custom_tokens[:train_size]
val_data = all_custom_tokens[train_size:train_size + val_size]
test_data = all_custom_tokens[train_size + val_size:]

# Initialize the custom tokenizer with the loaded vocabulary
tokenizer = GPT2Tokenizer(vocab_file="j͑ʃɹ ſȷɜⅎ ſȷᴜͷ̗.json", merges_file="ſɭɔʞ ſɟɹƽ ꞁȷ̀ᴜꞇ.txt")

tokenizer.pad_token = "<ɽ͑ʃ'ſ͕ȷƽ>"
tokenizer.bos_token = "<j͑ʃı],>"
tokenizer.eos_token = "<ſ̀ȷſɭſɭ>"
tokenizer.unk_token = "<ſɭſɭſ͕ȷ>"
tokenizer.space_token = "<ſɭɘſ͕ȷƽ>"

def get_vocab():
    return all_custom_tokens

# Define a function to decode token IDs using the custom vocabulary
def decode_token_ids(encoded_list):
    # Create a reverse mapping of token IDs to tokens in your custom vocabulary
    reverse_vocab = {v: k for k, v in custom_vocab.items()}
    # Use the reverse mapping to convert token IDs back to tokens
    decoded_tokens = [reverse_vocab[token_id] for token_id in encoded_list]
    # Join the tokens together to form the decoded text
    decoded_text = ' '.join(decoded_tokens)
    # Print the decoded text
    
    return decoded_text

# Save the custom tokenizer to a directory
tokenizer.save_pretrained("ı],ᴜ ſ͕ɭᴜ j͑ʃᴜꞇ ꞁȷ̀ɔ j͑ʃƽɔƽ")

# Print the results
print("Tokenized Text:", all_custom_tokens)

# Test
test_text = all_custom_tokens
encoded = tokenizer.encode(test_text, add_special_tokens=True, return_tensors='pt')

# Convert the encoded tensor to a list of token IDs
encoded_list = encoded.tolist()[0]
decoded_text = decode_token_ids(encoded_list)
print("Decoded Text:", decoded_text)
