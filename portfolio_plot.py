import os
# Used for Matplotlib to work without a graphical interface
os.environ["MPLBACKEND"] = "Agg"
import pandas as pd
from yfinance import download
import matplotlib.pyplot as plt
from pathlib import Path
from shared_code import SUBS

# Paths and outputs used
output_path = Path("portfolio_output")
portfolio_weights_path = Path("portfolio_weights")
output_path.mkdir(parents=True, exist_ok=True)

# Benchmark ETF comparisons
benchmark_comparison  = ["SPY", "QQQ", "SMH", "XLK"]

# Investment amount used
investment_amount  = 10000


# Return month bounds for starting month and the following month
def month_bounds(year, month):
    return pd.Timestamp(year, month, 1), pd.Timestamp(year, month + 1, 1)


# Normalize ticker symbols to proper uppercase string format
def load_weights(input_path: Path) -> pd.DataFrame:
    input_data = pd.read_csv(input_path)
    return input_data.assign(ticker=input_data["ticker"].astype(str).str.upper())


# Download adjusted close prices for the set of tickers
def ticker_prices(tickers, start_period, end_period):
    return download((set(tickers)), start_period, end_period, auto_adjust=True, progress=False)["Close"]


def create_portfolio_simulation(subreddit: str, imported_dataframe: pd.DataFrame) -> pd.Series:

    # Create subreddit dataframe on subreddit match with these columns: ["year", "month", "ticker", "weight"]
    subreddit_weights_dataframe = imported_dataframe.loc[imported_dataframe["subreddit"] == subreddit, ["year", "month", "ticker", "weight"]]

    # Convert weights to normalized weights summing to 1
    subreddit_weights_dataframe["normalized_weights"] = (subreddit_weights_dataframe["weight"] /
                                                 subreddit_weights_dataframe.groupby(["year", "month"])["weight"].transform("sum"))

    # Create a MultiIndex dataframe for fast lookups
    # Referenced: https://pandas.pydata.org/docs/user_guide/advanced.html
    normalized_weight_lookup = subreddit_weights_dataframe.set_index(["year", "month", "ticker"])["normalized_weights"]

    sorted_monthly_tickers = sorted(set(normalized_weight_lookup.index.droplevel("ticker")))

    # Used to generate the window of time we are looking at prices. Starts at the beginning of the time
    # period and ends at the month after the end of the last period
    for i in (0, -1):
        if i == 0:
            start_of_period, ignored_value  = month_bounds(*sorted_monthly_tickers[i])
        else:
            last_period_year, last_period_month = sorted_monthly_tickers[i]
            ignored_value, end_of_period = month_bounds(last_period_year, last_period_month + 1)

    # Download prices for all tickers appearing in the subreddit window.
    all_unique_tickers = ticker_prices(subreddit_weights_dataframe["ticker"], start_of_period, end_of_period)

    # Variable storage
    roll_forward_capital = investment_amount
    output = []

    # Used to dictate moving time periods so that we are plotting one month in advance after we consider our data
    plotting_month = sorted_monthly_tickers[1:]
    previous_month = sorted_monthly_tickers
    next_month = [(last_period_year,  last_period_month + 1)]

    # Used to do the computation for each month:
    # 1) Determine time period
    # 2) Gather weights from previous period,
    # 3) Intersection between prices and weights
    # 4) Normalize tickers
    # 5) Determine share count and portfolio value
    for previous, current in zip(previous_month, plotting_month + next_month):
        start, end  = month_bounds(*current)
        price_time_period = all_unique_tickers.query("index >= @start and index < @end")
        previous_period_weights = normalized_weight_lookup.loc[previous]
        weights_and_prices_dataframe = price_time_period.columns.intersection(previous_period_weights.index)
        normalize_weights = previous_period_weights[weights_and_prices_dataframe] / previous_period_weights[weights_and_prices_dataframe].sum()
        share_count = roll_forward_capital * normalize_weights / price_time_period.iloc[0][weights_and_prices_dataframe]
        monthly_portfolio_value = price_time_period[weights_and_prices_dataframe].mul(share_count, axis=1).sum(axis=1)
        output.append(monthly_portfolio_value)
        roll_forward_capital = float(monthly_portfolio_value.iloc[-1])

    # Concatenate month segments and drop duplicates
    portfolio = pd.concat(output)
    duplicate_mask = ~portfolio.index.duplicated()
    return portfolio[duplicate_mask]

def run_model(csv_path: Path):
    # Dictionary for holding portfolio results for each subreddit
    portfolio_curves = {}
    for subreddits in SUBS:
        portfolio_curves[subreddits] = create_portfolio_simulation(subreddits, load_weights(csv_path))
    portfolio_dataframe = pd.DataFrame(portfolio_curves).sort_index()



    # Output directory naming
    out_directory = output_path / csv_path.stem
    out_directory.mkdir(parents=True, exist_ok=True)

    # Plot portfolio
    axis = (portfolio_dataframe / portfolio_dataframe.iloc[0])
    axis = axis.plot(figsize=(12, 6))

    axis.set_title(f"Portfolios vs Benchmarks — {csv_path.stem}")

    # Obtain benchmark price data for same time period as in the portfolios
    benchmark_price_data_df = ticker_prices(benchmark_comparison,
                      portfolio_dataframe.index[0],
                      portfolio_dataframe.index[-1] + pd.Timedelta(days=1)).reindex(portfolio_dataframe.index)

    # Plot benchmarks
    normalize_benchmark_curves = (benchmark_price_data_df  / benchmark_price_data_df.iloc[0])
    normalize_benchmark_curves.plot(ax = axis)

    # Remove legend title for better formatting
    axis.legend(title = None)

    # Set a layout, save plot, close figure
    plt.tight_layout()
    plt.savefig(out_directory / f"portfolio_chart_{csv_path.stem}.png", dpi = 300)
    plt.close()

def main():
    # Run model for each portfolio weighing in order
    for model in sorted(portfolio_weights_path.glob("*.csv")): run_model(model)

if __name__ == "__main__":
    main()
