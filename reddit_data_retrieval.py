from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('reddit-comment-extractor').getOrCreate()

reddit_comments_path = '/courses/datasets/reddit_comments_repartitioned/'
output = 'reddit-stock-comments'

comments_schema = types.StructType([
    types.StructField('archived', types.BooleanType()),
    types.StructField('author', types.StringType()),
    types.StructField('body', types.StringType()),
    types.StructField('score', types.LongType()),
    types.StructField('created_utc', types.StringType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
])

#Modified code from suggested provided reddit extraction code: https://github.sfu.ca/ggbaker/cluster-datasets/blob/main/reddit/extract.py
def main():
    reddit_comments = spark.read.schema(comments_schema).json(reddit_comments_path)

    subs = ['stocks', 'wallstreetbets']

    filtered = reddit_comments \
        .where(functions.col('subreddit').isin(subs)) \
        .filter(functions.col('year') == 2024) \
        .filter(functions.col('month').between(1, 6)) \
        .where(functions.col('body').isNotNull()) \
        .where(~functions.col('body').isin("[deleted]", "[removed]")) \
        .where(functions.length('body') > 10) \
        .sample(withReplacement=False, fraction=0.07, seed=30) \
        .select('subreddit', 'body', 'created_utc', 'score', 'year', 'month')

    filtered.write.json(output, mode='overwrite', compression='gzip')

main()