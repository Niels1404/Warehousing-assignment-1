import pandas as pd

# headers = ["Item", "Vnid", "Itdesc", "Ordnbr", "Linenbr", "Cusnbr", "Ordqty", "Shpqty", "Uom", "Txndate"]

# Place the very big sales dataset into a dataframe and print the first rows
SalesData = pd.read_csv("dc23Sales04.txt", sep = "\t")
print(SalesData.head(), sep = "\t")
