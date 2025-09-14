import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ReadFiles import SalesData, SKUChar

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Ensure datetime
SalesData['txndate'] = pd.to_datetime(SalesData['txndate'], errors='coerce')

# 2. Merge SKU dimensions to SalesData
data = SalesData.merge(
    SKUChar[['SKU', 'SellLen', 'SellWid', 'SellHgt']],
    left_on='item', right_on='SKU', how='left'
)

# 3. Compute unit volume in cubic feet
data['UnitVolume_ft3'] = (
    data['SellLen'] * data['SellWid'] * data['SellHgt']
) / 1728  # cubic inches -> cubic feet

# 4. Compute outbound volume per line
data['LineVolume_ft3'] = data['UnitVolume_ft3'] * data['shpqty']

# 5. Aggregate daily
daily = data.groupby(data['txndate'].dt.date).agg(
    total_items=('shpqty', 'sum'),
    total_skus=('item', pd.Series.nunique),
    total_volume=('LineVolume_ft3', 'sum')
).reset_index()

# 6. Add weekday
daily['weekday'] = pd.to_datetime(daily['txndate']).dt.day_name()

# 7. Remove Saturday and Sunday
daily = daily[~daily['weekday'].isin(['Saturday', 'Sunday'])]

# 8. Aggregate per weekday
weekly_stats = daily.groupby('weekday').agg(
    avg_items=('total_items', 'mean'),
    avg_skus=('total_skus', 'mean'),
    avg_volume=('total_volume', 'mean'),
    q25_volume=('total_volume', lambda x: np.percentile(x, 25)),
    q75_volume=('total_volume', lambda x: np.percentile(x, 75))
).reset_index()

# Order weekdays properly
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
weekly_stats['weekday'] = pd.Categorical(weekly_stats['weekday'], categories=weekday_order, ordered=True)
weekly_stats = weekly_stats.sort_values('weekday')

# --- Plot 1: Outbound volume with 25%-75% range ---
plt.figure(figsize=(12,6))
sns.barplot(data=weekly_stats, x="weekday", y="avg_volume", color="skyblue")

lower_err = np.maximum(weekly_stats['avg_volume'] - weekly_stats['q25_volume'], 0)
upper_err = np.maximum(weekly_stats['q75_volume'] - weekly_stats['avg_volume'], 0)

plt.errorbar(
    x=weekly_stats['weekday'],
    y=weekly_stats['avg_volume'],
    yerr=[lower_err, upper_err],
    fmt='o', color='red', capsize=5
)

plt.ylabel("Outbound Volume (cubic feet)")
plt.title("Average Outbound Volume per Weekday (25%-75% range)")
plt.xticks(rotation=45)
plt.show()

# --- Plot 2: Average items and SKUs per weekday as grouped bars on same axis ---
fig, ax = plt.subplots(figsize=(12,6))

x = np.arange(len(weekly_stats['weekday']))
width = 0.35  # width of bars

bars1 = ax.bar(x - width/2, weekly_stats['avg_items'], width, label='Avg Items', color='lightgreen')
bars2 = ax.bar(x + width/2, weekly_stats['avg_skus'], width, label='Avg SKUs', color='skyblue')

ax.set_xticks(x)
ax.set_xticklabels(weekly_stats['weekday'])
ax.set_ylabel("Average Items / SKUs Picked")
plt.title("Average Picked Items and SKUs per Weekday (Grouped)")

ax.legend()
plt.show()

