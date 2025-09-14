import pandas as pd

# Import all the dataframes from the ReadFiles file
from ReadFiles import SalesData, ReserveCartons, ReserveCases, ActiveCartons, ActiveLocs, SKUChar, SKUInfo, SKUMinMax, PickOrders

# Oké, dit is me een hele opgave. Ik denk dat we moeten beginnen met het beunen van één dataframe met daarin alle relevante info over
# elke SKU, want volgens mij refereer je hier veel aan. 
# Dan moeten we kijken wat er nou bedoeld wordt met cartons en cases en of we daar wat mee moeten.
# Zodra we de SKUs duidelijk hebben, kunnen we wat met de sales data doen en kijken welke SKUs door het DC veel worden ingekocht.
# Hiermee kunnen we volgens mij de activity profiles wel invullen, behalve de heatmap, die we misschien beter kunnen skippen.
# Dan resteren alleen de KPIs nog, waarvan ik zo snel niet weet of dat veel werk is of niet.

