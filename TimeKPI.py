import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ReadFiles import ActiveCartons, ReserveCartons

# Create datetime columns using positional indices
# ActiveCartons -> Date (col 10), Time (col 11)
ActiveCartons['created_datetime'] = pd.to_datetime(
    ActiveCartons.iloc[:, 10].astype(str) + ' ' + ActiveCartons.iloc[:, 11].astype(str),
    errors='coerce'
)

# ReserveCartons -> Date (col 3), Time (col 4)
ReserveCartons['completed_datetime'] = pd.to_datetime(
    ReserveCartons.iloc[:, 3].astype(str) + ' ' + ReserveCartons.iloc[:, 4].astype(str),
    errors='coerce'
)

# Merge on Carton# (assume col 1 is Carton# in both datasets — adjust if different!)
order_times = ActiveCartons.merge(
    ReserveCartons[[1, 'completed_datetime']],
    left_on=1,
    right_on=1,
    how='left'
)

# Calculate lead time in hours
order_times['lead_time_hours'] = (order_times['completed_datetime'] - order_times['created_datetime']).dt.total_seconds() / 3600

# Clean data: remove negative or missing lead times
order_times = order_times[order_times['lead_time_hours'] >= 0].dropna(subset=['lead_time_hours'])

# KPI: Average order lead time
avg_lead_time = order_times['lead_time_hours'].mean()

print(f"Average Order Lead Time: {avg_lead_time:.2f} hours")

# Optional: distribution plot
plt.figure(figsize=(10,6))
sns.histplot(order_times['lead_time_hours'], bins=50, kde=True, color='skyblue')
plt.xlabel('Order Lead Time (hours)')
plt.ylabel('Number of Orders')
plt.title('Distribution of Order Lead Time')
plt.tight_layout()
plt.show()