# Reddit Sentiment Stock Analysis

An end to end data pipeline that mines six months of Reddit comments from **r/stocks** and **r/wallstreetbets**, scores them with three different sentiment analysis models (**VADER**, **TextBlob**, and **Loughran–McDonald**), converts the results into monthly rebalanced stock portfolios, and backtests those portfolios against major market benchmarks (SPY, QQQ, SMH, XLK).

The project answers two questions:

1. **How do general purpose and finance specific sentiment analyzers differ** when applied to the same informal, slang heavy social media text?
2. **Is a trading strategy built purely on Reddit sentiment and ticker popularity viable?**

**Results:** portfolios built from VADER and TextBlob sentiment on r/wallstreetbets returned **~20% over 6 months** (Jan–Jun 2024), outperforming SPY, QQQ, and XLK, and trailing only the semiconductor ETF SMH. The finance oriented Loughran–McDonald dictionary underperformed due to its inability to parse informal language.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Repository Structure](#repository-structure)
- [Pipeline Stages](#pipeline-stages)
- [Methodology Details](#methodology-details)
- [Results](#results)
- [Requirements & Setup](#requirements--setup)
- [Running the Pipeline](#running-the-pipeline)
- [Configuration](#configuration)
- [Limitations](#limitations)
- [Disclaimer](#disclaimer)

---

## How It Works

```
Reddit comments (SFU cluster, PySpark)
        │
        ▼
imported_data/reddit_comments.json  ──►  Ticker extraction (curated NASDAQ + NYSE lists)
        │
        ▼
Three parallel sentiment models: VADER │ TextBlob │ Loughran–McDonald
        │
        ▼
Monthly per ticker scores:  score = mentions × mean sentiment
        │
        ▼
Top 15 tickers per subreddit per month
        │
        ▼
Laplace smoothed portfolio weights (diversification control)
        │
        ▼
Backtest with 1 month trading lag vs. SPY / QQQ / SMH / XLK  ──►  performance charts
```

The core scoring metric **sentiment × mention count** captures both *how positively* a stock is discussed and *how much* it is discussed, mimicking how a retail investor would actually perceive online hype: the most praised **and** most talked about stocks rank highest.

---

## Repository Structure

| Path | Description |
|---|---|
| `reddit_data_retrieval.py` | PySpark job (run on the SFU compute cluster) that extracts and filters raw Reddit comment data |
| `shared_code.py` | Common utilities: ticker list loading, ticker extraction with false positive filtering, aggregation, filtering, and top 15 selection |
| `VADER_data_generator.py` | Scores every comment with VADER's compound polarity score |
| `TextBlob_data_generator.py` | Scores every comment with TextBlob's polarity score |
| `Loughran_McDonald_generator.py` | Scores every comment against the Loughran–McDonald financial dictionary |
| `filter_all_methods.py` | Narrows each model's output to the top 15 tickers per subreddit per month by `mentions × sentiment` |
| `portfolio_generator.py` | Converts filtered picks into smoothed portfolio weights that sum to exactly 100% |
| `portfolio_plot.py` | Simulates a $10,000 portfolio with monthly rebalancing and a one month trading lag; plots results against benchmark ETFs |
| `imported_data/` | Inputs: merged Reddit comments JSON, NASDAQ & NYSE ticker lists, Loughran–McDonald master dictionary |
| `generated_data/` | Intermediate outputs: raw and filtered per model sentiment CSVs |
| `portfolio_weights/` | Final monthly portfolio weightings for each model |
| `portfolio_output/` | Backtest performance charts (PNG, one per model) |

---

## Pipeline Stages

### 0. Data collection (`reddit_data_retrieval.py`)

Runs on the SFU cluster against the class Reddit dataset, using PySpark, it:

- Filters `stocks` and `wallstreetbets` subreddits, January–June 2024
- Drops null, `[deleted]`, and `[removed]` comments and any comment ≤ 10 characters
- Takes a reproducible 7% random sample (`seed=30`) to keep the dataset tractable
- Writes compressed JSON, which is then merged into a single `imported_data/reddit_comments.json`

### 1–3. Sentiment generation (`VADER_data_generator.py`, `TextBlob_data_generator.py`, `Loughran_McDonald_generator.py`)

Each script independently:

1. Loads the merged comment JSON
2. Scores every comment body with its model:
   - **VADER** — compound score from a lexicon tuned for social media (handles slang, emphasis, negation)
   - **TextBlob** — general purpose polarity score; more neutral than VADER, less formal than LM
   - **Loughran–McDonald** — counts positive/negative hits against the LM financial master dictionary (with suffix stripping for `'S/ING/ES/ED/S`) and computes `(pos − neg) / (pos + neg)`
3. Extracts ticker mentions using precompiled regexes for both `$TICKER` and bare `TICKER` forms, validated against **manually curated NASDAQ and NYSE ticker lists** rather than raw regex matching alone
4. Explodes multi ticker comments into one row per ticker, then aggregates per `(subreddit, year, month, ticker)` into monthly mention counts and mean sentiment
5. Filters out tickers with fewer than 10 total subreddit mentions or non-positive cumulative sentiment
6. Writes `generated_data/<MODEL>_data_output.csv`

**Ticker false positive handling:** a curated exclusion set removes single letters, common English words, and ambiguous ticker symbols (`AI`, `IT`, `DD`, `BB`, `OPEN`, `NOW`, etc.) that are overwhelmingly used as ordinary words or Reddit slang rather than stock references. Accurate ticker detection is the foundation of the whole pipeline, so precision was prioritized over recall here.

### 4. Filtering (`filter_all_methods.py`)

For each model, computes `monthly_mention_sentiment_score = mentions × sentiment` and keeps the **top 15 tickers per subreddit per month**, producing `generated_data/<MODEL>_data_filtered.csv`.

### 5. Portfolio construction (`portfolio_generator.py`)

Raw scores would let a single mega cap dominate, NVDA alone could account for ~80% of some months. To enforce a diversification standard, weights are computed with **Laplace (additive) smoothing**:

```
weight(ticker) = (score + α) / (Σ scores + α · n)
```

where `α` is the `smoothing_factor` (default `1.0`) and `n` is the number of tickers that month. Higher `α` flattens extreme weightings toward equal weight. Percentage weights are rounded, with any rounding residue applied to the final ticker so every monthly portfolio sums to exactly 100%. Output: `portfolio_weights/<MODEL>.csv`.

### 6. Backtesting & plotting (`portfolio_plot.py`)

Simulates each subreddit's portfolio from a **$10,000 starting balance** using adjusted close prices from `yfinance`:

- **One month trading lag:** weights derived from month *M*'s comments are traded during month *M+1*, since you cannot trade on a full month of data until that month has ended. This makes the simulation more realistic and stress tests whether the signal survives aging.
- Capital rolls forward month to month (returns compound); portfolios rebalance to the new month's weights at each month's first trading day.
- Results are normalized and plotted against **SPY, QQQ, SMH, and XLK** over the same window, saved as 300 dpi PNGs in `portfolio_output/`.

---

## Methodology Details

**Why sentiment × mentions?** Sentiment alone rewards a ticker praised in three comments; mentions alone rewards a ticker that's heavily discussed but hated. The product requires both popularity and positive conviction, the closest analogue to how retail investors actually interpret online stock buzz.

**Why three models?** They span a spectrum of formality: VADER is built for social media slang, LM is built formal financial text, and TextBlob sits between. Comparing them on identical data isolates how much lexicon choice, not data, drives the outcome.

**Why smoothing?** Portfolio design is where researcher bias is most likely to creep in. The smoothing parameter is deliberately exposed as a single tunable knob: the same picks can produce anywhere from a ~5% to a ~30% six month return depending on concentration tolerance. The reported results use a conservative setting that imitates a well diversified portfolio, and the parameter is documented as arbitrary rather than optimized on outcomes.

---

## Results

| Model | r/wallstreetbets portfolio | r/stocks portfolio | Key takeaway |
|---|---|---|---|
| **VADER** | **~20% / 6 mo** — beat SPY, QQQ, XLK; trailed only SMH | In line with SPY/QQQ | Slang aware lexicon captures WSB language well |
| **Loughran–McDonald** | Slightly beat most benchmarks | Slightly below almost all benchmarks | Formal dictionary misses informal speech; many comments score 0 and are discarded |
| **TextBlob** | **~20% / 6 mo** — beat SPY, QQQ, XLK; trailed only SMH | **~20% / 6 mo** — beat all but SMH | Best overall despite being the "middle-ground" analyzer; small differences in scoring shifted picks/weights favorably |

Additional findings:

- The detected ticker universe differed sharply by model: VADER/TextBlob surfaced high volume retail favorites (NVDA, TSLA, SMCI, AMD, TSM), while LM's formal dictionary surfaced a smaller, different set (ARM, PANW, HPE, AMZN, PLTR) with far lower mention counts, direct evidence of how lexicon choice reshapes the signal.
- VADER and TextBlob produced very similar aggregate dataframes. Their return gap came from marginal differences in monthly picks and weights, and would likely converge with a larger sample.
- The strongest practical use case is not standalone trading but **measuring which stocks are trending and how the crowd feels about them** a signal that can serve as one input among many in a broader strategy.

---

## Requirements & Setup

**Python libraries:**

```bash
pip install pandas numpy yfinance matplotlib pyspark textblob vaderSentiment
python -m textblob.download_corpora
```

**Data inputs (in `imported_data/`):**

- `reddit_comments.json` — merged output of the cluster extraction step
- `NASDAQ Stock List.csv` and `NYSE Stock List.csv` — ticker symbol lists used for validation
- `Loughran-McDonald_Master_Dictionary.csv` — the LM master dictionary (available from the [Loughran–McDonald resources page](https://sraf.nd.edu/loughranmcdonald-master-dictionary/))

`pyspark` is only required for stage 0 on the cluster; the rest of the pipeline runs locally.

---

## Running the Pipeline

Run the scripts in order from the repository root:

```bash
# 0. On the SFU cluster: extract raw comments, then merge output files
#    into imported_data/reddit_comments.json
spark-submit reddit_data_retrieval.py

# 1–3. Generate sentiment scores with each model
python VADER_data_generator.py
python TextBlob_data_generator.py
python Loughran_McDonald_generator.py

# 4. Filter to top 15 tickers per subreddit per month
python filter_all_methods.py

# 5. Generate smoothed portfolio weights
python portfolio_generator.py

# 6. Backtest and plot vs. benchmarks
python portfolio_plot.py
```

Outputs land in `generated_data/`, `portfolio_weights/`, and `portfolio_output/` respectively.

---

## Configuration

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `smoothing_factor` | `portfolio_generator.py` | `1.0` | Higher values flatten weights toward equal weight (more diversification, lower concentration risk) |
| `max_tickers` | `portfolio_generator.py` | `15` | Tickers held per subreddit per month |
| Minimum mentions filter | sentiment generators | `10` | Excludes thinly discussed tickers |
| `investment_amount` | `portfolio_plot.py` | `10000` | Starting capital for the simulation |
| `benchmark_comparison` | `portfolio_plot.py` | `SPY, QQQ, SMH, XLK` | Benchmark ETFs plotted against portfolios |
| Sample fraction / date range | `reddit_data_retrieval.py` | `0.07`, Jan–Jun 2024 | Size and window of the comment dataset |

---

## Limitations

- **Ambiguous tickers.** Symbols like `AI` and `IT` are valid tickers but overwhelmingly used as ordinary words. No reliable meaning differation method was found, so they are excluded entirely. This means genuine discussion of those stocks is invisible to the pipeline.
- **Portfolio design bias.** There is are many ways to turn sentiment rankings into portfolio weights. Any chosen scheme (including this one) embeds assumptions, and it is difficult to fully separate "the method fits the data" from "the data fits the method." The smoothing parameter is explicitly documented as arbitrary for this reason.
- **Lexicon blind spots.** Dictionary based models cannot parse sarcasm, irony, inside jokes, or community specific idioms (common to see on r/wallstreetbets). Each model effectively measures only the subset of commenters whose language matches its lexicon, introducing demographic filtering bias.
- **Sample and window.** Results come from a 7% comment sample over a single six month bull market window; they should not be extrapolated to other market regimes without further testing.

---

## Disclaimer

This project was built for **CMPT 353 (Computational Data Science) at Simon Fraser University** as an exploration of sentiment analysis and data pipelines. It is not financial advice, and the backtested returns should not be interpreted as evidence of a deployable trading strategy.
