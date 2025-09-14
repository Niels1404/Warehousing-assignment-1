import pandas as pd
from ReadFiles import SalesData
import matplotlib.pyplot as plt
import seaborn as sns
# We can calculate the outbound logistics volume used by assuming that each order is sent on the day it is created
# We can calculate turnover in the same way.

# Quality KPI #1: Stockout rate
import pandas as pd

# --- Assumption ---
# SalesData dataframe is already loaded
# Columns: 'item' (SKU), 'ordqty' (ordered), 'shpqty' (shipped)

# 1. Flag stockout lines
SalesData['Stockout'] = SalesData['shpqty'] < SalesData['ordqty']

# 2. Compute stockout rate overall
total_lines = len(SalesData)
stockout_lines = SalesData['Stockout'].sum()
stockout_rate = (stockout_lines / total_lines) * 100

print(f"Total lines: {total_lines:,}")
print(f"Stockout lines: {stockout_lines:,}")
print(f"Overall Stockout Rate: {stockout_rate:.2f}%")

# 3. Breakdown per SKU
sku_stockout = (
    SalesData.groupby('item')
    .agg(
        total_lines=('item', 'count'),
        stockout_lines=('Stockout', 'sum')
    )
    .reset_index()
)

sku_stockout['stockout_rate'] = (
    sku_stockout['stockout_lines'] / sku_stockout['total_lines']
) * 100

# 4. Count how many SKUs have a 100% stockout rate
num_full_stockouts = (sku_stockout['stockout_rate'] == 100).sum()
print(f"Number of SKUs with 100% stockout rate: {num_full_stockouts}")
