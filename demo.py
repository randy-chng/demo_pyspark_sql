import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F

# Get current working directory
cwd = os.getcwd()

# Initialise spark
spark = SparkSession.builder.getOrCreate()

# Read local csv into spark
df = spark.read.csv(
    os.path.join(
        'file://',
        cwd,
        'bank.csv'
    ),
    header=True
)

# Analyse schema
df.printSchema()

# Count records
df.count()

# Sample records
df.limit(3).show(truncate=False)

# Get distinct value of each column
col_distinct_val = {
    col: None for col in df.columns
}

for col in col_distinct_val.keys():

    print('Analysing', col)
    col_distinct_val[col] = [i[col] for i in df.select(col).distinct().collect()]

# Identify categorical and numerical columns
def is_number(value):
    try:
        float(value)
        return True
    except Exception:
        return False

numerical_cols = []
categorical_cols = []

for col, values in col_distinct_val.items():

    print('Analysing', col)
    test = {is_number(value) for value in values}

    if test == {True}:
        numerical_cols.append(col)
    else:
        categorical_cols.append(col)

# Convert numerical column type to numerical
for col in numerical_cols:

    df = df.withColumn(
        col,
        df[col].cast(DoubleType())
    )

# Summary of numerical columns
df.select(numerical_cols).summary().show()

# Summary of categorical columns
for col in categorical_cols:

    print(col, '>>', col_distinct_val[col])

# Standardise value of existing column
df = df.withColumn(
    'poutcome',
    F.when(
        F.col('poutcome') == 'other',
        F.lit('unknown')
    ).otherwise(
        F.col('poutcome')
    )
)

# Grouping and aggregating
df.groupby(
    'job'
).agg(
    F.sum('balance').alias('tot_balance')
).sort(
    'tot_balance',
    ascending=False
).show()

# Create new column for negative balance check
df = df.withColumn(
    'negative_balance',
    F.when(
        F.col('balance') < 0,
        1
    ).otherwise(
        0
    )
)

# Check of negative balance is a one-off or common occurrence
df.groupby(
    'negative_balance'
).count().show()

# Rename created column
df = df.withColumnRenamed(
    'negative_balance',
    'neg_bal'
)

# Drop created column
df = df.drop('neg_bal')

# Create new dataframe
month_dict = [
    {'month': 'jan', 'month_num': 1},
    {'month': 'feb', 'month_num': 2},
    {'month': 'mar', 'month_num': 3},
    {'month': 'apr', 'month_num': 4},
    {'month': 'may', 'month_num': 5},
    {'month': 'jun', 'month_num': 6},
    {'month': 'jul', 'month_num': 7},
    {'month': 'aug', 'month_num': 8},
    {'month': 'sep', 'month_num': 9},
    {'month': 'oct', 'month_num': 10},
    {'month': 'nov', 'month_num': 11},
    {'month': 'dec', 'month_num': 12}
]

month_df = spark.createDataFrame(
    pd.DataFrame(month_dict)
)

# Join dataframes
df = df.join(
    month_df,
    on='month',
    how='inner'
)

# Union dataframes
df = df.union(df)

# Deduplicate dataframe
df = df.dropDuplicates()


