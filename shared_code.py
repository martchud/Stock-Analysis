import pandas as pd, re
from pathlib import Path

REDDIT_JSON = "imported_data/reddit_comments.json"
NASDAQ = "imported_data/NASDAQ Stock List.csv"
NYSE = "imported_data/NYSE Stock List.csv"

SUBS = ("stocks","wallstreetbets")


# Precompile regex for tickers with and without $ symbol
extract_ticker_with_cash_symbol = re.compile(r'\$([A-Z]{1,5})\b')
extract_ticker_without_cash_symbol = re.compile(r'\b([A-Z]{1,5})\b')


# Exclude common english words, individual letters, conflicting terms that have other meanings (i.e AI/IT),
# and common slang/abbreviations/phrases
EXCLUDE_SLANG_ABBREVIATIONS_AND_FALSE_POSITIVES = \
    { "GL","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V", "W","X","Y","Z",
      "OPEN", "RDDT", "SNAP", "LINK", "GO", "RH", "IRS", "RSI", "EU", "UK", "ICE", "IP", "CRE", "MS", "UI", "OS", "MAN",
      "NOW","ON","OR", "SI", "SO","SEE","SUN","WAS","YOU","UP","PM", "WTF","IMO","PC","TV","DM","OC", "LOVE", "DD", "OP","AI",
      "ALL","AM","AND","ARE","AS","AT","BE", "BEST","BY","CAN","DO","FOR","HAS","IT", "BB"}


# Helper function for load_in_and_process_tickers
def get_correct_column(passed_column):
    for upper_lower_check_result in ("symbol", "Symbol"):
        if upper_lower_check_result in passed_column:
            return upper_lower_check_result

    return passed_column[0]


 # Load in path and data
def load_in_and_process_tickers(passed_path: str) -> set:
    read_in_dataframe = pd.read_csv(Path(passed_path))
    dataframe_column = get_correct_column(read_in_dataframe.columns)
    string_dataframe = read_in_dataframe[dataframe_column].astype(str).str.strip().str.upper()
    filtered_string_dataframe = string_dataframe.dropna().unique()

    return set(filtered_string_dataframe)

# Load in tickers from NASDAQ and NYSE list and exclude duplicates and invalid tickers
valid_filtered_tickers = (load_in_and_process_tickers(NASDAQ) | load_in_and_process_tickers(NYSE)) - EXCLUDE_SLANG_ABBREVIATIONS_AND_FALSE_POSITIVES

# Extract tickers from text and remove duplicates, keep only known tickers
def extract_valid_tickers_from_comment_text(passed_in_comment_text: pd.Series) -> pd.Series:
    def filter_comments_for_tickers(comment_text):
        return sorted((set(extract_ticker_with_cash_symbol.findall(comment_text)) |
                       set(extract_ticker_without_cash_symbol.findall(comment_text)))
                      & valid_filtered_tickers)

    return passed_in_comment_text.apply(filter_comments_for_tickers)


# Expand the column tickers to rows using explode function
def explode_stock_ticker_to_rows_and_group_columns(passed_in_dataframe: pd.DataFrame) -> pd.DataFrame:
    return passed_in_dataframe.explode("tickers").rename(columns = {"tickers": "ticker"})


# Group by columns and aggregate on ticker. Count mentions and get mean sentiment
def group_by_column_list_and_calculate_sentiment(passed_in_dataframe: pd.DataFrame, column_compound = "compound") -> pd.DataFrame:
    return passed_in_dataframe.groupby(["subreddit", "year", "month", "ticker"]).agg(mentions = ("ticker", "count"), sentiment = (column_compound, "mean")).reset_index()


# Get the totals mentions and sentiment per subreddit/ticker and overall
def calculate_total_mentions_and_total_sentiment(passed_in_dataframe: pd.DataFrame, column_compound = "compound"):
    per_subreddit_totals = passed_in_dataframe.groupby(["subreddit", "ticker"])
    total_mentions_per_sub = per_subreddit_totals["ticker"].count().rename("total_mentions_sub")
    total_sentiment_per_sub = per_subreddit_totals[column_compound].sum().rename("total_sentiment_sub")

    stock_ticker_overall_totals = passed_in_dataframe.groupby("ticker")
    total_mentions_for_all = stock_ticker_overall_totals["ticker"].count().rename("total_mentions_all")
    total_sentiment_for_all = stock_ticker_overall_totals[column_compound].sum().rename("total_sentiment_all")

    return (total_mentions_per_sub.to_frame().join(total_sentiment_per_sub).reset_index(),
            total_mentions_for_all.to_frame().join(total_sentiment_for_all).reset_index())


# Filter data out using min mentions and using only positive sentiment
def apply_data_filters(passed_in_dataframe: pd.DataFrame, minimum_mentions_required: int) -> pd.DataFrame:
    filter_for_mentions_and_sentiment = (passed_in_dataframe["total_mentions_sub"] >= minimum_mentions_required) & (passed_in_dataframe["total_sentiment_sub"] > 0)

    return passed_in_dataframe[filter_for_mentions_and_sentiment]


# Score by mentions * sentiment, keep top 15 stocks per month
def filter_top_15_stocks_per_month(input_csv_file: Path, output_csv_file: Path, keep_only_top_15_stocks: int = 15) -> None:
    read_in_csv_dataframe = pd.read_csv(input_csv_file)

    # Generate a weighing score that considers both mentions and sentiment in its evaluation
    read_in_csv_dataframe["monthly_mention_sentiment_score"] = read_in_csv_dataframe["mentions"] * read_in_csv_dataframe["sentiment"]

    top_15_stocks_dataframe = (read_in_csv_dataframe.groupby(["subreddit", "year", "month"])["monthly_mention_sentiment_score"].
        nlargest(keep_only_top_15_stocks).reset_index(level = ["subreddit", "year", "month"], drop = True)
    )

    output_csv_file.parent.mkdir(parents=True, exist_ok=True)
    (read_in_csv_dataframe.loc[top_15_stocks_dataframe.index]
     .rename(columns={"mentions": "monthly_mentions", "sentiment": "monthly_sentiment"})
     .to_csv(output_csv_file, index=False))
