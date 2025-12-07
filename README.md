# CMPT-353-Stock-Analysis-Project
CMPT 353 - Stock Analysis Project


# CMPT-353-Stock-Analysis-Project
CMPT 353 - Stock Analysis Project

Required libraries:
import pandas 
import numpy
import yfinance 
import matplotlib 
import pyspark
import textblob
import vaderSentiment
python -m textblob.download_corpora

Order of execution:
0) Obtaining the data: run reddit_data_retrieval.py on the sfu cluster. Merge all files obtained into one json file and proceed to next steps
1) VADER_data_generator.py -> generates sentiment scores using VADER
2) TextBlob_data_generator.py -> generates sentiment scores using TextBlob
3) Loughran_McDonald_generator.py -> generates sentiment scores using LM
4) filter_all_method.py -> Filters all 3 of the above methods to remove unecessary data and narrow down the stock picks per month to the top 25 stocks based on a calculation of sentiment*mentions
5) Portfolio_generator.py -> Generates the portfolio weighing's for the filtered stock picks. This is done so that the weighings aren't a 1 to 1 copy of the top stocks in order to prevent the top few stocks from dominating every portfolio (ensure there is a diversification standard)
6) portfolio_plot.py -> plots the portfolio for each month of data in the next month (i.e January date is plotted in February). The objective behind this decision is to simulate a lag in a trade reacting to obtained data (you can't immediately trade on the data you obtain in the current month). This both simulates a more realistic environment (data collection takes time) and emphasizes the impact of the data (harder to profit with older information).
