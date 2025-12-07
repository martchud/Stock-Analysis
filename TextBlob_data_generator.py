import pandas as pd
from textblob import TextBlob
from shared_code import REDDIT_JSON
from shared_code import (
    extract_valid_tickers_from_comment_text,
    explode_stock_ticker_to_rows_and_group_columns,
    group_by_column_list_and_calculate_sentiment,
    calculate_total_mentions_and_total_sentiment,
    apply_data_filters,
)
from pathlib import Path

# Output file location
output_format = "generated_data/TextBlob_data_output.csv"

# Load Reddit comments dataset from JSON
imported_comment_dataframe = pd.read_json(REDDIT_JSON, lines=True)

# Apply TextBlob sentiment score analyzer
def assign_sentiment_score(comment_text):
    return TextBlob(comment_text).sentiment.polarity

imported_comment_dataframe["polarity_score"] = imported_comment_dataframe["body"].apply(assign_sentiment_score)

# Extract and filter valid stock tickers
imported_comment_dataframe["tickers"] = extract_valid_tickers_from_comment_text(imported_comment_dataframe["body"])
imported_comment_dataframe = imported_comment_dataframe[imported_comment_dataframe["tickers"].apply(len) > 0]

# Expand the list column tickers to rows using explode function
exploded_dataframe = explode_stock_ticker_to_rows_and_group_columns(imported_comment_dataframe)

# Generate monthly formatted dataframe
monthly_formatted_dataframe = group_by_column_list_and_calculate_sentiment(exploded_dataframe, "polarity_score")

# Calculate mentions/sentiment per subreddit and total overall
mention_and_sentiment_per_sub, mention_and_sentiment_overall = calculate_total_mentions_and_total_sentiment(exploded_dataframe, "polarity_score")

# Merge datasets
monthly_sub_merged_dataframe = monthly_formatted_dataframe.merge(mention_and_sentiment_per_sub, on=["subreddit","ticker"], how="left")
final_result = monthly_sub_merged_dataframe.merge(mention_and_sentiment_overall, on="ticker", how="left")

Path(output_format).parent.mkdir(parents=True, exist_ok=True)
apply_data_filters(final_result, 10).to_csv(output_format, index=False)
