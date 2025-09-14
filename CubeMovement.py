import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ReadFiles import SalesData, SKUChar

# -------------------------------
# 1. Prepare Data
# -------------------------------

# Merge sales and SKU dimensions
data = SalesData.merge(
    SKUChar[['SKU', 'SellLen', 'SellWid', 'SellHgt']],
    left_on='item',
    right_on='SKU',
    how='left'
)

# Check for missing dimension data
missing_dims = data[data[['SellLen','SellWid','SellHgt']].isnull().any(axis=1)]
if not missing_dims.empty:
    print(f"Warning: {len(missing_dims)} SKUs missing dimension data. They will be ignored.")
    data = data.dropna(subset=['SellLen','SellWid','SellHgt'])

# -------------------------------
# 2. Calculate Volume in Cubic Feet
# -------------------------------

# Volume in cubic inches per unit
data['UnitVolume_in3'] = data['SellLen'] * data['SellWid'] * data['SellHgt']

# Convert to cubic feet
data['UnitVolume_ft3'] = data['UnitVolume_in3'] / 1728  # 12*12*12 = 1728

# Total volume moved per line in cubic feet
data['TotalVolume_ft3'] = data['UnitVolume_ft3'] * data['shpqty']

# -------------------------------
# 3. Aggregate by SKU
# -------------------------------

cube_movement = data.groupby('item').agg(
    total_units=('shpqty', 'sum'),
    total_volume_ft3=('TotalVolume_ft3', 'sum')
).reset_index()

# Sort by total volume
cube_movement = cube_movement.sort_values(by='total_volume_ft3', ascending=False)

# -------------------------------
# 4. Top 20 SKUs by total volume with percentage
# -------------------------------

# Total warehouse volume
total_volume = cube_movement['total_volume_ft3'].sum()

# Top 20 SKUs by total volume
top20 = cube_movement.head(20).copy()

# Calculate percentage of total volume per SKU
top20['pct_total_volume'] = 100 * top20['total_volume_ft3'] / total_volume

# Plot: Bar chart with dual y-axis
fig, ax1 = plt.subplots(figsize=(12,6))

# Bar chart: total volume in cubic feet
sns.barplot(x='item', y='total_volume_ft3', data=top20, palette='viridis', ax=ax1)
ax1.set_xlabel('SKU')
ax1.set_ylabel('Total Picked Volume (cubic feet)')
ax1.set_xticklabels(top20['item'], rotation=45, ha='right')

# Secondary axis: percentage line
ax2 = ax1.twinx()
ax2.plot(top20['item'], top20['pct_total_volume'], color='red', marker='o', linewidth=2)
ax2.set_ylabel('% of Total Volume', color='red')
ax2.set_ylim(0, max(top20['pct_total_volume']) * 1.1)

plt.title('Top 20 SKUs by Cube Movement (Cubic Feet) with % of Total Volume')
plt.tight_layout()
plt.show()

# Total warehouse volume in cubic feet
total_volume = cube_movement['total_volume_ft3'].sum()

print(f"Total volume of all SKUs in the warehouse: {total_volume:,.2f} cubic feet")