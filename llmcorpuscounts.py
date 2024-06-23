import argparse
from collections import Counter
from itertools import product
from nltk.tokenize import word_tokenize 
import pandas as pd
import os
import time
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from queue import Queue
from threading import Thread


# Tokenization using nltk

# Exclude non alphabetic tokens (numbers, special characters, punctuation, whitespace)
def tokenize_nltk(sentence):
    tokens = word_tokenize(sentence)
    return [word for word in tokens]


# Read in word lists

jobs= pd.read_excel('/home/anegru/Test_Folder/Word_Lists/jobs.xlsx', engine='openpyxl')
names= pd.read_excel('/home/anegru/Test_Folder/Word_Lists/names.xlsx', engine='openpyxl')
pronouns= pd.read_excel('/home/anegru/Test_Folder/Word_Lists/pronouns.xlsx', engine='openpyxl')
nationalities= pd.read_excel('/home/anegru/Test_Folder/Word_Lists/nationality.xlsx', engine='openpyxl')
religions= pd.read_excel('/home/anegru/Test_Folder/Word_Lists/religions.xlsx', engine='openpyxl')
ethnicities = pd.read_excel('/home/anegru/Test_Folder/Word_Lists/ethnicity.xlsx', engine='openpyxl') 

negative_words = pd.read_excel('/home/anegru/Test_Folder/Word_Lists/negative_connotations.xlsx', engine='openpyxl')
positive_words = pd.read_excel('/home/anegru/Test_Folder/Word_Lists/positive_connotations.xlsx', engine='openpyxl')


# Unique sets for word lists 

unique_jobs = set(jobs['jobs'])
unique_pronouns = set(pronouns['pronouns'])
unique_names = set(names['names'])
unique_nationalities = set(nationalities['nationality'])
unique_religions = set(religions['religion'])
unique_ethnicities = set(ethnicities['ethnicity'])
unique_negatives = set(negative_words['neg'])
unique_positives = set(positive_words['pos'])


# Single counts & co-occurrence counts with parquet files

def write_combined_counters_to_csv(single_word_results, cooccurrence_counters, directory, file_suffix):
    # Process and write single word counters
    single_word_data = [] 
    for category, counter in single_word_results.items():
        for word, count in counter.items(): 
            single_word_data.append({'category': category, 'word': word, 'count': count})
    single_word_df = pd.DataFrame(single_word_data)
    single_word_file_path = os.path.join(directory, f"{file_suffix}_single_counts.csv")
    single_word_df.to_csv(single_word_file_path, index=False) 

    # Process and write co-occurrence counters
    cooccurrence_data = []
    for category, counter in cooccurrence_counters.items():
        for pair, count in counter.items():
            cooccurrence_data.append({'category': category, 'pair': pair, 'count': count})
    cooccurrence_df = pd.DataFrame(cooccurrence_data)
    cooccurrence_file_path = os.path.join(directory, f"{file_suffix}_cooccurrence_counts.csv")
    cooccurrence_df.to_csv(cooccurrence_file_path, index=False)
    
results_dir = '/home/anegru/Test_Folder/Results'


def process_single_file(file_path, results_dir):

    # Extract file name prefix
    file_name_prefix = os.path.basename(file_path).split('.')[0]
    single_word_file_path = os.path.join(results_dir, f"{file_name_prefix}_single_counts.csv")
    cooccurrence_file_path = os.path.join(results_dir, f"{file_name_prefix}_cooccurrence_counts.csv")

    # If result files already exist
    if os.path.exists(single_word_file_path) and os.path.exists(cooccurrence_file_path):
        print(f"Skipping {file_name_prefix} as results already exist.")
        return
    
    # Counters for single counts
    counter_jobs = Counter()
    counter_pronouns = Counter()
    counter_names = Counter()    
    counter_nationalities = Counter()
    counter_ethnicities = Counter()
    counter_religions = Counter()


    # Categories for single word counters
    categories = {
        'jobs': (unique_jobs, counter_jobs),
        'pronouns': (unique_pronouns, counter_pronouns),
        'names': (unique_names, counter_names),
        'nationalities': (unique_nationalities, counter_nationalities),
        'ethnicities': (unique_ethnicities, counter_ethnicities),
        'religions': (unique_religions, counter_religions)
    }

    # Co-occurrence counters 
    cooccurrence_counters = {
        'jobs_pronouns': Counter(),
        'jobs_names': Counter(),
        'jobs_nationalities': Counter(),
        'jobs_ethnicities': Counter(),
        'negative_nationalities': Counter(),
        'negative_ethnicities': Counter(),
        'negative_religions': Counter(),
        'positive_nationalities': Counter(),
        'positive_ethnicities': Counter(),
        'positive_religions': Counter()
    }

    # Extract the file name prefix
    file_name_prefix = os.path.basename(file_path).split('.')[0]
    print(f'Started process {file_name_prefix}')
    try: 
    
        df = pd.read_parquet(file_path)[['content']]
        for content in df['content']:
            tokens = list(tokenize_nltk(content))
            updated = False
            
            for token in tokens:
                
                for category, (unique_set, counter) in categories.items():
                    if token in unique_set:
                        counter[token] += 1
                        updated = True
                
                # Update co-occurrence counters only if the token belongs to any category
                # 'updated' ensures that cooccurrence counters are only updated if the token is part of a category (avoid unnecessary checks/updates)
                if updated and token in unique_jobs:
                    cooccurrence_counters['jobs_pronouns'].update((token, pronoun) for pronoun in tokens if pronoun in unique_pronouns)
                    cooccurrence_counters['jobs_names'].update((token, name) for name in tokens if name in unique_names)
                    cooccurrence_counters['jobs_nationalities'].update((token, nationality) for nationality in tokens if nationality in unique_nationalities)
                    cooccurrence_counters['jobs_ethnicities'].update((token, ethnicity) for ethnicity in tokens if ethnicity in unique_ethnicities)

                # Update counters for negative and positive words
                if token in unique_negatives:
                    cooccurrence_counters['negative_nationalities'].update((token, nationality) for nationality in tokens if nationality in unique_nationalities)
                    cooccurrence_counters['negative_ethnicities'].update((token, ethnicity) for ethnicity in tokens if ethnicity in unique_ethnicities)
                    cooccurrence_counters['negative_religions'].update((token, religion) for religion in tokens if religion in unique_religions)

                if token in unique_positives:
                    cooccurrence_counters['positive_nationalities'].update((token, nationality) for nationality in tokens if nationality in unique_nationalities)
                    cooccurrence_counters['positive_ethnicities'].update((token, ethnicity) for ethnicity in tokens if ethnicity in unique_ethnicities)
                    cooccurrence_counters['positive_religions'].update((token, religion) for religion in tokens if religion in unique_religions)

        # Single word results & write results to csv
        single_word_results = {k: v for k, (_, v) in categories.items()}
        write_combined_counters_to_csv(single_word_results, cooccurrence_counters, results_dir, file_name_prefix)  

    except Exception as e:
        print(f"An error occurred: {e}")

base_dir = '/scratch-shared/anegru/data/'

# Argument parser
parser = argparse.ArgumentParser(description='Process file names in a directory')
parser.add_argument('files', type=str, nargs='+', help='a file name or names to process in the directory') 
args = parser.parse_args() 

full_file_paths = [os.path.join(base_dir, file_name) for file_name in args.files]
print(full_file_paths)

num_workers = 40

# Run processing concurrently on all Parquet files
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(process_single_file, file_path, results_dir) for file_path in full_file_paths]
    for future in as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"An error occurred: {e}")
          

print("All processing completed.")


