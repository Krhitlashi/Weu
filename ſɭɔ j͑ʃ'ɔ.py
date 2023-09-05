import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

# Load the custom pretrained language model
model_path = "ᶅſɔⅎ.pt"  # Path to your custom trained model
config_path = "ᶅſɔⅎ.json"  # Path to your configuration file
config = GPT2Config.from_json_file(config_path)
model = GPT2LMHeadModel.from_pretrained(model_path, config=config)

# Load the custom tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("ı],ᴜ ſ͕ɭᴜ j͑ʃᴜꞇ ꞁȷ̀ɔ j͑ʃƽɔƽ")

# Define a list of special tokens to remove
special_tokens_to_remove = [
    "<ɽ͑ʃ'ſ͕ȷƽ>",
    "<ſɭſɭſ͕ȷ>",
    "<j͑ʃı],>",
    "<ſ̀ȷſɭſɭ>",
    "<ſɭɘſ͕ȷƽ>",
]

# Load the custom vocabulary from the JSON file
with open("ſȷᴜͷ̗ ſɭɔʞ ꞁȷ̀ᴜꞇ.json", "r", encoding="utf-8") as vocab_file:
    custom_vocab = json.load(vocab_file)

prompts = [
    "ꞁȷ̀ɔ ſᶘᴜ ɽ͑ʃ'ᴜ",
    "ɭʃᴜ ꞁȷ̀ᴜ ɽ͑ʃ'ᴜȝ",
    "ꞁȷ̀ɜ j͐ʃɹ ŋᷠꞇ"
    ]

# Store generated sequences in a list
generated_token_ids_list = []

# Define a function to decode token IDs using the custom vocabulary
def decode_token_ids(encoded_list, custom_vocab):
    # 1. Create a reverse mapping of token IDs to tokens in your custom vocabulary
    reverse_vocab = {v: k for k, v in custom_vocab.items()}
    
    # 2. Use the reverse mapping to convert token IDs back to tokens
    decoded_tokens = [reverse_vocab[token_id] for token_id in encoded_list]
    
    # 3. Join the tokens together to form the decoded text
    decoded_text = ' '.join(decoded_tokens)
    
    return decoded_text

# Encoding and Generation
def zechawekiif(prompt):
    # Encode Single Input Text
    input_ids = tokenizer.encode(
        prompt,
        add_special_tokens=True,
        return_tensors='pt',
    )

    # Convert input_ids to PyTorch tensor
    input_ids = input_ids.clone().detach()

    # Generate text for the prompt
    max_length = 200
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,  # Number of text samples to generate
        no_repeat_ngram_size=2,  # Avoid repeating the same phrase
        do_sample=True,
        top_p=0.4,  # Adjust the top-p value for randomness
        temperature=0.4,  # Adjust the temperature for creativity
    )

    # Append generated sequences to the list
    generated_token_ids_list.extend(output.tolist())

    # Decode the generated text using decode_token_ids function
    generated_token_ids = output[0].tolist()
    generated_text = decode_token_ids(generated_token_ids, custom_vocab)
    return generated_text


# Debugging Output
for prompt in prompts:
    zechawekiif(prompt)

# print("Generated Output:", generated_token_ids_list)

# Decode the generated sequences
decoded_sequences = []
for generated_token_ids in generated_token_ids_list:
    if generated_token_ids:
        # Decode the token IDs
        decoded_text = decode_token_ids(generated_token_ids, custom_vocab)
    else:
        decoded_text = "ꞁȷ̀ɔȝ ɭl̀ɹƽ"  # Handle empty sequences

    decoded_sequences.append(decoded_text)

for j, decoded_text in enumerate(decoded_sequences):
    for special_token in special_tokens_to_remove:
        decoded_text = decoded_text.replace(special_token, "")
    print(f"|{j} j͑ʃᴜꞇ ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ|")
    print(decoded_text)
    print("")


# Interactive input and generation loop ı],ɹ ſɭ,ɹ
while True:
    print("j͑ʃɹ ʃᴜ }ʃᴜ ſɭɔʞ ꞁȷ̀ɔƽ ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ |ꞁȷ̀ᴜ ŋᷠɹ ꞁȷ̀ɔƽ j͑ʃɹ ʃᴜ }ʃᴜ ⸙ ⸙|")
    user_input = input()
    
    if user_input.lower() == ' ':
        break
    
    generated_text = zechawekiif(user_input)
    for special_token in special_tokens_to_remove:
        generated_text = decoded_text.replace(special_token, "")
    print("")
    print("|ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ|")
    print(generated_text)
    print("")