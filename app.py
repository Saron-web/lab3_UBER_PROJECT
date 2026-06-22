from flask import Flask
import pandas as pd

app = Flask(__name__)

# Load the CSV file
df = pd.read_csv("Uber-Jan-Feb-FOIL.csv")

# Clean the data
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Feature extraction
df['Hour'] = df['date'].dt.hour
df['Day'] = df['date'].dt.date
df['DayOfWeek'] = df['date'].dt.day_name()
df['IsWeekend'] = df['DayOfWeek'].isin(['Saturday', 'Sunday'])

# Preview and document data
print(df.head())
print(df.info())
print(df.describe())


