import pandas as pd
import matplotlib.pyplot as plt

# Path to your CSV file
file_path = 'data/train_data.csv'  # Replace with your actual file path

# Load the CSV file into a DataFrame
data = pd.read_csv(file_path, header=None, names=["Ground_Truth"])

# Count the frequency of each value (0, 1, or 2)
value_counts = data["Ground_Truth"].value_counts().sort_index()

# Plot a bar graph for the distribution
plt.figure(figsize=(8, 6))
value_counts.plot(kind='bar', width=0.8)
plt.title("Distribution of Values ()", fontsize=14)
plt.xlabel("Value", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()