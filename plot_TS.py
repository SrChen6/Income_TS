import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data (replace 'store_income.csv' with your actual file name)
df = pd.read_csv('income_data.csv')

# 2. Prepare the dataset
# Convert the 'Date' string into actual datetime objects
df['Date'] = pd.to_datetime(df['Date'])

# Force chronological order (the "fail-safe")
df = df.sort_values(by='Date')

# Set the Date as the index (crucial for time series plotting)
df.set_index('Date', inplace=True)

# 3. Create the plot
# Set the figure size to be wide (12) and reasonably tall (6)
plt.figure(figsize=(12, 6))

# Plot the index (Dates) on the X-axis and Income on the Y-axis
plt.plot(df.index, df['Income'], color='#1f77b4', linewidth=1.5, label='Daily Income')

# 4. Format the aesthetics
plt.title('Daily Store Income', fontsize=16, pad=15)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Income', fontsize=12)

# Add a subtle grid to make reading values easier
plt.grid(True, linestyle='--', alpha=0.6)

# Rotate the date labels on the X-axis so they don't overlap
plt.xticks(rotation=45)

# Add a legend
plt.legend()

# Automatically adjust the layout so labels aren't cut off
plt.tight_layout()

# 5. Display the graph
plt.show()