This is the repository for the masters thesis in Data Science at University of Amsterdam. This project is about exploring and quantifying bias in the open-source Falcon Large Language Model.

The files in this repository are as follows:

* word_lists = folder containing words used to perform co-occurrence counts in each bias category
* EDA_Notebook = code for exploratory data analysis
* combined_scores_visualisation = code for barchart plot depicting differences in scores between bias categories
* llmcorpuscounts.py = code to process text data from pre-training corpus to count occurrences & co-occurrences of words in bias categories from parquet files and write results to csv
* llm_predictions_gender.py = code to use Falcon LLM to generate & validate predictions based on given contexts & questions, outputting results to JSON (code shows gender-occupation category with Falcon 7b, other categories are processed in a similar way with the instruct model as well)
* aggregate_name_occupation_predictions.py = code to process predictions to extract & validate, aggregate counts and save to csv
* calculate_score_gender.py & calculate_score_religion.py = code to calculate the bias metric by comparing counts of male/female names with occupations & religions with positive/negative attributes (the same code as for religion applies to the remaining bias categories)
* pretraining_corpus_scores.ipynb = code to calculate the bias metric for all categories for pre-training corpus counts
