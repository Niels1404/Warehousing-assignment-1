# item_pareto.py
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def read_sales(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sales file not found: {path.resolve()}")

    df = pd.read_csv(
        path,
        sep=None,
        engine="python",
        dtype={
            "item": str,
            "vnid": str,
            "itdesc": str,
            "ordnbr": str,
            "linenbr": str,
            "cusnbr": str,
            "uom": str
        },
        parse_dates=["txndate"],
        keep_default_na=False
    )

    df.columns = [c.strip().lower() for c in df.columns]
    for c in ["item","vnid","itdesc","ordnbr","linenbr","cusnbr","uom"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    for col in ["ordqty", "shipqty"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df

def aggregate_item_activity(df: pd.DataFrame, metric: str = "picks") -> pd.DataFrame:
    ship_col = None
    for candidate in ["shipqty", "shpqty", "qtyshipped", "shippedqty"]:
        if candidate in df.columns:
            ship_col = candidate
            break
    if ship_col is None:
        raise KeyError(f"No shipped-quantity column found. Columns: {df.columns.tolist()}")

    shipped = df[df[ship_col] > 0].copy()

    if metric == "picks":
        grp = (shipped
               .drop_duplicates(subset=["ordnbr", "linenbr"])
               .groupby("item", as_index=False)
               .size()
               .rename(columns={"size": "activity"}))
    elif metric == "units":
        grp = (shipped
               .groupby("item", as_index=False)[ship_col]
               .sum()
               .rename(columns={ship_col: "activity"}))
    else:
        raise ValueError("metric must be 'picks' or 'units'")

    grp["activity"] = grp["activity"].astype(float)
    grp.sort_values("activity", ascending=False, inplace=True, ignore_index=True)

    total = grp["activity"].sum()
    grp["share_pct"] = 100 * grp["activity"] / total if total > 0 else 0
    grp["cum_share_pct"] = grp["share_pct"].cumsum()
    grp["rank"] = grp.index + 1

    return grp, total

def plot_item_pareto(grp: pd.DataFrame, metric_label: str, out_png: str):
    plt.figure(figsize=(11, 6))
    plt.plot(grp["rank"], grp["cum_share_pct"], linewidth=2, label="Cumulative share")
    plt.axhline(80, linestyle="--", linewidth=1, color="red", label="80% threshold")
    plt.title(f"Item Pareto by {metric_label}")
    plt.xlabel("Item Rank (descending activity)")
    plt.ylabel("Cumulative Share of Activity (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def summarize(grp: pd.DataFrame) -> str:
    def k_for_threshold(th):
        idx = np.searchsorted(grp["cum_share_pct"].values, th, side="left")
        return int(idx + 1) if idx < len(grp) else len(grp)

    n50 = k_for_threshold(50)
    n80 = k_for_threshold(80)
    n_all = len(grp)

    return (f"• Top {n50} items account for 50% of activity "
            f"({n50/n_all:.1%} of items).\n"
            f"• Top {n80} items account for 80% of activity "
            f"({n80/n_all:.1%} of items).\n"
            f"• Total items with shipped activity: {n_all}.")

def main():
    parser = argparse.ArgumentParser(description="Build an Item Pareto from sales file")
    parser.add_argument("--sales", default="dc23Sales04.txt",
                        help="Path to sales file (default: dc23Sales04.txt)")
    parser.add_argument("--metric", choices=["picks","units"], default="picks",
                        help="Pareto metric: 'picks' or 'units'")
    parser.add_argument("--out_csv", default="item_pareto.csv", help="Output CSV path")
    parser.add_argument("--out_png", default="item_pareto.png", help="Output PNG path")
    args = parser.parse_args()

    df = read_sales(args.sales)
    grp, total = aggregate_item_activity(df, metric=args.metric)

    grp.to_csv(args.out_csv, index=False)

    metric_label = "Shipped Lines (Picks)" if args.metric == "picks" else "Units Shipped"
    plot_item_pareto(grp, metric_label, args.out_png)

    print(f"Saved CSV -> {Path(args.out_csv).resolve()}")
    print(f"Saved PNG -> {Path(args.out_png).resolve()}")
    print(f"Total activity ({metric_label}): {int(total):,}")
    print("\nSummary:")
    print(summarize(grp))

    # --- Bar chart: Top 20 SKUs by % of total activity ---
    top_n = 20
    top_items = grp.head(top_n).copy()

    # Compute percentage of total activity
    top_items["pct_total"] = 100 * top_items["activity"] / grp["activity"].sum()

    plt.figure(figsize=(12,6))
    plt.bar(top_items["item"], top_items["pct_total"], color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Percentage of Total Shipped Activity (%)")
    plt.xlabel("Item (SKU)")
    plt.title(f"Top {top_n} SKUs by % of Total Shipped Activity")
    plt.tight_layout()
    plt.savefig("top20_sku_bar.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()
