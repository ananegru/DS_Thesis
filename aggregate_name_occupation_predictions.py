import json
import re
import pandas as pd

def extract_occupation(phrase):
    match = re.search(r'was a (\w+)|is a (\w+)|was an (\w+)|is an (\w+)', phrase)
    if match:
        return match.group(1) if match.group(1) else (match.group(2) if match.group(2) else (match.group(3) if match.group(3) else match.group(4)))
    return None

def clean_prediction_text(text):
    text = re.sub(r'<.*?>', '', text)  
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    return text.strip()

def validate_prediction(pred, names):
    pred_names = [name for name in names if name in pred]
    if len(pred_names) == 1:
        return pred_names[0]
    else:
        return "N/A"

file_path = '/home/anegru/Test_Folder/pred_outputs/pred_outputs_mixed_gender_occupation_falcon-7b.json'
aggregated_output_csv_path = '/home/anegru/Test_Folder/pred_outputs/name_occupation_counts_falcon-7b.csv' 

with open(file_path, 'r') as file:
    data = json.load(file)

name_occupation_counts = {}  
na_count = 0 

# Process each entry in the JSON data
for key, value in data.items():
    parts = key.split('|')
    male_name = parts[2]
    female_name = parts[3]
    occupation_phrase = parts[-1]  # Occupation phrase
    occupation = extract_occupation(occupation_phrase)  
    
    if occupation:
        for q_id, q_data in value.items():
            if q_id.startswith('q'):  # Ensure we're looking at a question entry
                pred_name = clean_prediction_text(q_data['pred'].strip()) 
                valid_name = validate_prediction(pred_name, [male_name, female_name]) 
                print(f"Original prediction: {q_data['pred']} -> Cleaned prediction: {pred_name} -> Valid name: {valid_name}")  # Debugging print
                
                # Post-processing step to ensure the answer is one of the provided names
                names = parts[2:4]
                if valid_name not in names:
                    valid_name = "N/A"
                
                if valid_name == "N/A":
                    na_count += 1
                
                # Update the count for this name and occupation
                if (valid_name, occupation) not in name_occupation_counts:
                    name_occupation_counts[(valid_name, occupation)] = 0
                name_occupation_counts[(valid_name, occupation)] += 1

df = pd.DataFrame(name_occupation_counts.items(), columns=['name_occupation', 'count'])
df[['name', 'occupation']] = df['name_occupation'].apply(pd.Series)
df = df.drop(columns=['name_occupation'])
df['name'] = df['name'].str.strip().str.rstrip('.')

# Aggregate data & save to csv
aggregated_df = df.groupby(['name', 'occupation'])['count'].sum().reset_index()
aggregated_df.to_csv(aggregated_output_csv_path, index=False)

print("Aggregated data has been saved to:", aggregated_output_csv_path)
print(f"Number of instances of N/A: {na_count}")







