import pandas as pd
from pathlib import Path

models_and_paths = {
    "TextBlob": "generated_data/TextBlob_data_filtered.csv",
    "VADER":    "generated_data/VADER_data_filtered.csv",
    "LM":       "generated_data/LM_data_filtered.csv",
}

#  Max tickers used per subreddit and per month
max_tickers = 15

# smoothing_factor is used to smooth out weighing's. The higher the values, the more extreme weightings are flatted
smoothing_factor = 1.0

output_path = Path("portfolio_weights")
output_path.mkdir(parents = True, exist_ok = True)

for model_used, path_used in models_and_paths.items():

    # Read data, sort, and limit to top tickers per subreddit/month
    sorted_top_tickers_dataframe = (pd.read_csv(path_used).sort_values(["subreddit", "year", "month"]).
                                    groupby(["subreddit", "year", "month"], group_keys = False).head(max_tickers))

    # Group by subreddit and date
    group_by = sorted_top_tickers_dataframe.groupby(["subreddit", "year", "month"])
    transform_n = group_by["ticker"].transform("size")

    # Formula is based on Laplace smoothing. The formula I use comes out to (ticker sentiment + smoothing)/(total sentiment + smoothing * number of tickers).
    # Rounded to 5 decimal places
    sorted_top_tickers_dataframe["weight"] = ((pd.to_numeric(sorted_top_tickers_dataframe["monthly_mention_sentiment_score"]) + smoothing_factor) /
                                              (group_by["monthly_mention_sentiment_score"].transform("sum") +
                                               smoothing_factor * transform_n)).round(5)

    # Convert weights to percentages and round to 2 decimals
    pct_weights = (sorted_top_tickers_dataframe["weight"] * 100).round(2)
    sorted_top_tickers_dataframe["pct_weights"] = pct_weights

    # For the group ["subreddit", "year", "month"] calculate sum of pct_weights for each group
    transform_weights_sum = sorted_top_tickers_dataframe.groupby(["subreddit", "year", "month"])["pct_weights"].transform("sum")

    # Boolean mask that's only true for last row
    last_row_boolean_mask = ~sorted_top_tickers_dataframe.duplicated(["subreddit", "year", "month"], keep = "last")

    # Calculates the carryover for all rows and applies it to only the last row for each group
    carryover = (100 - transform_weights_sum).round(2).mask(~last_row_boolean_mask, 0)

    # Apply carryover to last group to ensure the sum of weight_pct is 100 for all groups
    sorted_top_tickers_dataframe["weight_pct"] = (pct_weights + carryover).round(2)

    # Final result
    (sorted_top_tickers_dataframe[["subreddit", "year", "month", "ticker", "monthly_mention_sentiment_score", "weight", "weight_pct"]].
     to_csv(output_path / f"{model_used}.csv", index=False))
