import pandas as pd, re
from pathlib import Path
from shared_code import (
    REDDIT_JSON,
    extract_valid_tickers_from_comment_text,
    explode_stock_ticker_to_rows_and_group_columns,
    group_by_column_list_and_calculate_sentiment,
    calculate_total_mentions_and_total_sentiment,
    apply_data_filters, get_correct_column,
)

# Output file location
output_format = "generated_data/LM_data_output.csv"

# Regex for getting alphabetic terms of 2 or more lengths in character
valid_string_regex = re.compile(r"[A-Za-z]{2,}", re.IGNORECASE)

# Regex for detecting common suffix
valid_suffix_found_regex = re.compile(r"(?:'S|ING|ES|ED|S)$", re.IGNORECASE)


# Get positive and negative words out of loaded in dictionary
def load_in_lm_dictionary():
    read_in_lm_dictionary = pd.read_csv("imported_data/Loughran-McDonald_Master_Dictionary.csv")
    lm_dictionary_dataframe = read_in_lm_dictionary[get_correct_column(read_in_lm_dictionary.columns)].astype(str).str.upper()
    return set(lm_dictionary_dataframe.loc[read_in_lm_dictionary["Positive"] > 0]), set(lm_dictionary_dataframe.loc[read_in_lm_dictionary["Negative"] > 0])


# Get count of positive/negative words
def count_positive_negative_words(passed_in_comments: str, positive_words: set, negative_words: set) -> tuple[int, int]:
    positive = negative = 0
    for regex_hit in valid_string_regex.findall(passed_in_comments):
        regex_hit_upper_case = regex_hit.upper()
        if valid_suffix_found_regex.search(regex_hit_upper_case):
            suffix_portion = valid_suffix_found_regex.search(regex_hit_upper_case).group(0)
            final_filtered_word = regex_hit_upper_case[:-len(suffix_portion)] if len(regex_hit_upper_case) > len(suffix_portion) else regex_hit_upper_case
        else:
            final_filtered_word = regex_hit_upper_case
        positive += final_filtered_word in positive_words
        negative += final_filtered_word in negative_words
    return positive, negative


# For each comment body get positive/negative word count
def call_count_positive_negative_words(passed_in_dataframe):
    positive_words, negative_words = load_in_lm_dictionary()
    return [count_positive_negative_words(comment_text_in_body, positive_words, negative_words) for comment_text_in_body in passed_in_dataframe["body"].astype(str)]


# Calculate score and return 0 if denominator is >= 0
def calculate_normalized_sentiment_score(passed_in_dataframe_row):
    denominator = (passed_in_dataframe_row["positive"] + passed_in_dataframe_row["negative"])

    if denominator > 0:
        return (passed_in_dataframe_row["positive"] - passed_in_dataframe_row["negative"]) / denominator
    else:
        return 0.0


# Load Reddit comments dataset from JSON
imported_comment_dataframe = pd.read_json(REDDIT_JSON, lines=True)

# Apply LM sentiment score analyzer
imported_comment_dataframe[["positive", "negative"]] = pd.DataFrame(call_count_positive_negative_words(imported_comment_dataframe), index = imported_comment_dataframe.index)
imported_comment_dataframe["normalized_sentiment_score"] = imported_comment_dataframe.apply(calculate_normalized_sentiment_score, axis = 1)

# Extract and filter valid stock tickers
imported_comment_dataframe["tickers"] = extract_valid_tickers_from_comment_text(imported_comment_dataframe["body"])
imported_comment_dataframe = imported_comment_dataframe[imported_comment_dataframe["tickers"].apply(len) > 0]

# Expand the list column tickers to rows using explode function
exploded_dataframe = explode_stock_ticker_to_rows_and_group_columns(imported_comment_dataframe)

# Generate monthly formatted dataframe
monthly_formatted_dataframe = group_by_column_list_and_calculate_sentiment(exploded_dataframe, "normalized_sentiment_score")

# Calculate mentions/sentiment per subreddit and total overall
mention_and_sentiment_per_sub, mention_and_sentiment_overall = calculate_total_mentions_and_total_sentiment(exploded_dataframe, "normalized_sentiment_score")

# Merge datasets
monthly_sub_merged_dataframe = monthly_formatted_dataframe.merge(mention_and_sentiment_per_sub, on = ["subreddit", "ticker"], how = "left")
final_result = monthly_sub_merged_dataframe.merge(mention_and_sentiment_overall, on = "ticker", how = "left")

# Output files
Path(output_format).parent.mkdir(parents = True, exist_ok = True)
apply_data_filters(final_result, 10).to_csv(output_format, index = False)
