import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from shared_code import REDDIT_JSON
from shared_code import (
    extract_valid_tickers_from_comment_text,
    explode_stock_ticker_to_rows_and_group_columns,
    group_by_column_list_and_calculate_sentiment,
    calculate_total_mentions_and_total_sentiment,
    apply_data_filters,
)

# Output file location
output_format = "generated_data/VADER_data_output.csv"

# Initialize VADER sentiment analyzer
vader_sentiment_analyzer = SentimentIntensityAnalyzer()

# Load Reddit comments dataset from JSON
imported_comment_dataframe = pd.read_json(REDDIT_JSON, lines = True)

# Apply VADER sentiment analyzer and join the results
imported_comment_dataframe = imported_comment_dataframe.join(imported_comment_dataframe["body"].apply(vader_sentiment_analyzer.polarity_scores).apply(pd.Series))

# Extract and filter valid stock tickers
imported_comment_dataframe["tickers"] = extract_valid_tickers_from_comment_text(imported_comment_dataframe["body"])
imported_comment_dataframe = imported_comment_dataframe[imported_comment_dataframe["tickers"].apply(len) > 0]

# Expand the list column tickers to rows using explode function
exploded_dataframe = explode_stock_ticker_to_rows_and_group_columns(imported_comment_dataframe)

# Generate monthly formatted dataframe
monthly_formatted_dataframe = group_by_column_list_and_calculate_sentiment(exploded_dataframe, "compound")

# Calculate mentions/sentiment per subreddit and total overall
mention_and_sentiment_per_sub, mention_and_sentiment_overall = calculate_total_mentions_and_total_sentiment(exploded_dataframe, "compound")

# Merge datasets
monthly_sub_merged_dataframe = monthly_formatted_dataframe.merge(mention_and_sentiment_per_sub, on = ["subreddit", "ticker"], how = "left")
final_result = monthly_sub_merged_dataframe.merge(mention_and_sentiment_overall, on = "ticker", how = "left")

# Output files
Path(output_format).parent.mkdir(parents = True, exist_ok = True)
apply_data_filters(final_result, 10).to_csv(output_format, index = False)
