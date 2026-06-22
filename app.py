import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Uber-Jan-Feb-FOIL.csv")

# Subtask 1.1 – Data Cleaning & Exploration
df['date'] = pd.to_datetime(df['date'])

print(df.head())
print(df.info())
print(df.describe())

# Subtask 1.2 – Feature Extraction
df['Hour'] = df['date'].dt.hour
df['Day'] = df['date'].dt.date
df['DayOfWeek'] = df['date'].dt.day_name()

# Traffic analysis
daily_trips = df.groupby('Day').size()
hourly_trips = df.groupby('Hour').size()

# Visualization: Trips per Day
plt.figure(figsize=(10,5))
daily_trips.plot(kind='bar', color='skyblue')
plt.title('Trips per Day')
plt.xlabel('Day')
plt.ylabel('Number of Trips')
plt.tight_layout()
plt.show()

# Visualization: Heatmap (DayOfWeek vs Hour)
heatmap_data = df.groupby(['DayOfWeek', 'Hour']).size().unstack()
sns.heatmap(heatmap_data, cmap='viridis')
plt.title('Trip Frequency by Day and Hour')
plt.show()

