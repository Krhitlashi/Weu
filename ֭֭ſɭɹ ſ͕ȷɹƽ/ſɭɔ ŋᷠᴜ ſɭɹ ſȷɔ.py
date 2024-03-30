import json
import pandas as pd

# Read the Excel file
df = pd.read_excel('j͑ʃᴜ j͑ʃᴜ ſɭᴜ.xlsx', header=None)

# Convert the dataframe to a dictionary
data_dict = df.set_index(0)[1].to_dict()

# Output the dictionary as JSON
with open('ſɭɔ ŋᷠᴜ.json', 'w', encoding='utf-8') as f:
    json.dump(data_dict, f, indent=4, ensure_ascii=False)
