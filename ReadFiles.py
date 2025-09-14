import pandas as pd

# Read al datafiles and place each one in their respective dataframe
ReserveCases = pd.read_excel("DC23CASES AS OF 050210.xls")
ActiveLocs = pd.read_excel("DC23ACTIVE AS OF 050210.xls")
SKUChar = pd.read_excel("Phily_Dim_Status2.xls")
SKUInfo = pd.read_csv("ITEMDATAV2.txt", sep =  "\t")
SKUMinMax = pd.read_excel("DC23MINMAX2.xls")
PickOrders = pd.read_csv("TRCART23.txt", sep = "\t")
ActiveCartons = pd.read_csv("CDCART23.txt", sep = "\t")
ReserveCartons = pd.read_csv("CHCART23.txt", sep = "\t")

# Place the very big sales dataset into a dataframe
SalesData = pd.read_csv("dc23Sales04.txt", sep = "\t")


