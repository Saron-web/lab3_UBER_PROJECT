from flask import Flask, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import redis
import json

app = Flask(__name__)

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

def run_analysis():
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

    # Aggregate/group data by day and hour
    daily_trips = df.groupby('Day').size()
    hourly_trips = df.groupby('Hour').size()

    print(daily_trips)
    print(hourly_trips)

    # Find busiest day and peak hour
    print("Busiest Day:", daily_trips.idxmax())
    print("Peak Hour:", hourly_trips.idxmax())

    # Bar chart – Trips per Day
    plt.figure(figsize=(10,5))
    daily_trips.plot(kind='bar', color='skyblue')
    plt.title('Trips per Day')
    plt.xlabel('Day')
    plt.ylabel('Number of Trips')
    plt.tight_layout()
    plt.show()

    # Bar chart – Trips per Hour
    plt.figure(figsize=(10,5))
    hourly_trips.plot(kind='bar', color='orange')
    plt.title('Trips per Hour')
    plt.xlabel('Hour')
    plt.ylabel('Number of Trips')
    plt.tight_layout()
    plt.show()

    # Heatmap – Trip Frequency by Day and Hour
    heatmap_data = df.groupby(['DayOfWeek', 'Hour']).size().unstack()
    sns.heatmap(heatmap_data, cmap='viridis')
    plt.title('Trip Frequency by Day and Hour')
    plt.show()

@app.route("/")
def home():
    return "Uber Data Analysis App Running Inside Docker"

@app.route("/predict_traffic")
def predict_traffic():
    timestamp = request.args.get('timestamp')

    cached_value = redis_client.get(timestamp)
    if cached_value:
        return {
            "timestamp": timestamp,
            "prediction": json.loads(cached_value),
            "source": "cache"
        }

    result = {"traffic_level": "High", "hour": 17}

    redis_client.setex(timestamp, 3600, json.dumps(result))

    return {
        "timestamp": timestamp,
        "prediction": result,
        "source": "compute"
    }

if __name__ == "__main__":
    run_analysis()  
    app.run(host="0.0.0.0", port=5000)
