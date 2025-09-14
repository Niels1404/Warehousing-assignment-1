# customer_pareto.py
# Usage (default picks):  python customer_pareto.py --sales dc23Sales04.txt
# Usage (by units):       python customer_pareto.py --sales dc23Sales04.txt --metric units
# Optional zone filter (if you later merge zones): --zones B C

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def read_sales(path: str) -> pd.DataFrame:
    """
    Robust reader for the 59MB dc23Sales04.txt.
    Tries to auto-detect delimiter; lowercases columns; trims whitespace.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sales file not found: {path.resolve()}")

    # Auto-detect separator
    df = pd.read_csv(
        path,
        sep=None,              # auto-detect
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

    # normalize columns
    df.columns = [c.strip().lower() for c in df.columns]
    for c in ["item","vnid","itdesc","ordnbr","linenbr","cusnbr","uom"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # numeric columns
    if "ordqty" in df.columns:
        df["ordqty"] = pd.to_numeric(df["ordqty"], errors="coerce").fillna(0)
    if "shipqty" in df.columns:
        df["shipqty"] = pd.to_numeric(df["shipqty"], errors="coerce").fillna(0)

    return df

def aggregate_customer_activity(df: pd.DataFrame, metric: str = "picks") -> pd.DataFrame:
    # Try to find the shipped quantity column
    ship_col = None
    for candidate in ["shipqty", "shpqty", "qtyshipped", "shippedqty"]:
        if candidate in df.columns:
            ship_col = candidate
            break
    if ship_col is None:
        raise KeyError(f"No shipped-quantity column found. Columns are: {df.columns.tolist()}")

    if metric not in {"picks", "units"}:
        raise ValueError("metric must be 'picks' or 'units'")

    # filter shipped lines
    shipped = df[df[ship_col] > 0].copy()

    if metric == "picks":
        grp = (shipped
               .drop_duplicates(subset=["ordnbr", "linenbr"])  # defensively
               .groupby("cusnbr", as_index=False)
               .size())
        grp.rename(columns={"size": "activity"}, inplace=True)
    else:  # units
        grp = (shipped
               .groupby("cusnbr", as_index=False)[ship_col]
               .sum()
               .rename(columns={ship_col: "activity"}))

    grp["activity"] = grp["activity"].astype(float)
    grp.sort_values("activity", ascending=False, inplace=True, ignore_index=True)

    # add rank + cumulative share
    total = grp["activity"].sum()
    grp["share_pct"] = 100 * grp["activity"] / total if total > 0 else 0
    grp["cum_share_pct"] = grp["share_pct"].cumsum()
    grp["rank"] = grp.index + 1

    return grp, total

def plot_pareto(grp: pd.DataFrame, metric_label: str, out_png: str):
    plt.figure(figsize=(11, 6))

    plt.plot(grp["rank"], grp["cum_share_pct"], linewidth=2, label="Cumulative share")
    plt.axhline(80, linestyle="--", linewidth=1, color="red", label="80% threshold")

    plt.title(f"Customer Pareto by {metric_label}")
    plt.xlabel("Customer Rank (descending activity)")
    plt.ylabel("Cumulative Share of Activity (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def summarize(grp: pd.DataFrame) -> str:
    """
    Return a brief textual summary: how many customers make up 50% / 80%.
    """
    def k_for_threshold(th):
        idx = np.searchsorted(grp["cum_share_pct"].values, th, side="left")
        return int(idx + 1) if idx < len(grp) else len(grp)

    n50 = k_for_threshold(50)
    n80 = k_for_threshold(80)
    n_all = len(grp)

    return (f"• Top {n50} customers account for 50% of activity "
            f"({n50/n_all:.1%} of customers).\n"
            f"• Top {n80} customers account for 80% of activity "
            f"({n80/n_all:.1%} of customers).\n"
            f"• Total customers with shipped activity: {n_all}.")

def main():
    parser = argparse.ArgumentParser(description="Build a Customer Pareto from dc23Sales04.txt")
    parser.add_argument("--sales", required=True, help="Path to dc23Sales04.txt")
    parser.add_argument("--metric", choices=["picks","units"], default="picks",
                        help="Pareto metric: 'picks' (shipped lines) or 'units' (shipped quantity)")
    parser.add_argument("--out_csv", default="customer_pareto.csv", help="Output CSV path")
    parser.add_argument("--out_png", default="customer_pareto.png", help="Output PNG path")
    args = parser.parse_args()

    df = read_sales(args.sales)
    grp, total = aggregate_customer_activity(df, metric=args.metric)

    # Save results
    grp.to_csv(args.out_csv, index=False)

    metric_label = "Shipped Lines (Picks)" if args.metric == "picks" else "Units Shipped"
    plot_pareto(grp, metric_label, args.out_png)

    print(f"Saved CSV -> {Path(args.out_csv).resolve()}")
    print(f"Saved PNG -> {Path(args.out_png).resolve()}")
    print(f"Total activity ({metric_label}): {int(total):,}")
    print()
    print("Summary:")
    print(summarize(grp))

if __name__ == "__main__":
    import sys
    sys.argv = [
        "CustomerOrderProfiles.py",
        "--sales", "dc23Sales04.txt",   # adjust path if needed
        "--metric", "picks"
    ]
    main()
