# OrderFillRate.py
# Usage:
#   python OrderFillRate.py --sales dc23Sales04.txt
# Optional:
#   --outdir outputs

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Helpers ----------

def _find_col(df, candidates, required=True, name=""):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing required {name or 'column'}; tried {candidates}. Found: {df.columns.tolist()}")
    return None

def _read_sales(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sales file not found: {p.resolve()}")

    df = pd.read_csv(
        p,
        sep=None, engine="python",
        dtype={"item": str, "ordnbr": str, "linenbr": str, "cusnbr": str},
        parse_dates=["txndate"],  # ok if absent; we'll handle below
        keep_default_na=False
    )
    df.columns = [c.strip().lower() for c in df.columns]

    # required cols
    ord_col  = _find_col(df, ["ordqty","orderqty","orderedqty"], name="ordered quantity")
    ship_col = _find_col(df, ["shipqty","shpqty","qtyshipped","shippedqty"], name="shipped quantity")

    # numeric
    df[ord_col]  = pd.to_numeric(df[ord_col], errors="coerce").fillna(0)
    df[ship_col] = pd.to_numeric(df[ship_col], errors="coerce").fillna(0)

    # ensure txndate exists as datetime
    if "txndate" in df.columns:
        # if parse_dates failed (e.g., mixed types), coerce now
        if not np.issubdtype(df["txndate"].dtype, np.datetime64):
            df["txndate"] = pd.to_datetime(df["txndate"], errors="coerce")
    else:
        df["txndate"] = pd.NaT

    # keep handy names
    df = df.rename(columns={ord_col: "ordqty_norm", ship_col: "shipqty_norm"})
    return df

# ---------- KPI calculations ----------

def compute_fill_rates(df: pd.DataFrame):
    # Unit (line) fill rate
    total_ord  = df["ordqty_norm"].sum()
    total_ship = df["shipqty_norm"].sum()
    unit_fill_rate = (total_ship / total_ord * 100) if total_ord > 0 else 0.0

    # Perfect order fill rate: every line in order shipped in full
    line_full = (df["shipqty_norm"] >= df["ordqty_norm"])  # treat overship as full; change to == if needed
    order_full = (
        df.assign(line_full=line_full)
          .groupby("ordnbr")["line_full"]
          .all()
          .mean() * 100
    )

    # By SKU (unit fill rate)
    sku_fill = (
        df.groupby("item", as_index=False)[["ordqty_norm","shipqty_norm"]]
          .sum()
          .assign(unit_fill_rate_pct=lambda x: np.where(x["ordqty_norm"]>0, 100*x["shipqty_norm"]/x["ordqty_norm"], np.nan))
          .sort_values("unit_fill_rate_pct", ascending=True)
    )

    # By customer (unit fill rate)
    cust_fill = (
        df.groupby("cusnbr", as_index=False)[["ordqty_norm","shipqty_norm"]]
          .sum()
          .assign(unit_fill_rate_pct=lambda x: np.where(x["ordqty_norm"]>0, 100*x["shipqty_norm"]/x["ordqty_norm"], np.nan))
          .sort_values("unit_fill_rate_pct", ascending=True)
    )

    # Monthly trend (unit fill rate)
    if df["txndate"].notna().any():
        ts = (df.dropna(subset=["txndate"])
                .assign(month=lambda x: x["txndate"].dt.to_period("M").dt.to_timestamp())
                .groupby("month")[["ordqty_norm","shipqty_norm"]]
                .sum()
                .assign(unit_fill_rate_pct=lambda x: np.where(x["ordqty_norm"]>0, 100*x["shipqty_norm"]/x["ordqty_norm"], np.nan))
                .reset_index())
    else:
        ts = pd.DataFrame(columns=["month","ordqty_norm","shipqty_norm","unit_fill_rate_pct"])

    summary = {
        "total_ordered_units": int(total_ord),
        "total_shipped_units": int(total_ship),
        "unit_fill_rate_pct": round(unit_fill_rate, 2),
        "perfect_order_fill_rate_pct": round(order_full, 2),
        "n_orders": df["ordnbr"].nunique(),
        "n_lines": len(df),
        "n_skus": df["item"].nunique(),
        "n_customers": df["cusnbr"].nunique()
    }

    return summary, sku_fill, cust_fill, ts

# ---------- Plots ----------

def plot_trend(ts: pd.DataFrame, out_png: Path):
    if ts.empty:
        return
    plt.figure(figsize=(10,5))
    plt.plot(ts["month"], ts["unit_fill_rate_pct"], marker="o")
    plt.title("Unit Fill Rate — Monthly Trend")
    plt.ylabel("Unit Fill Rate (%)")
    plt.xlabel("Month")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_bottom_entities(df_ranked: pd.DataFrame, key_col: str, out_png: Path, k=20):
    # show worst k SKUs/customers by unit fill rate
    subset = df_ranked.head(k)
    plt.figure(figsize=(12,6))
    plt.bar(subset[key_col].astype(str), subset["unit_fill_rate_pct"])
    plt.title(f"Worst {k} by Unit Fill Rate (%) — {key_col.upper()}")
    plt.xlabel(key_col.upper())
    plt.ylabel("Unit Fill Rate (%)")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Quality KPI — Order Fill Rate")
    ap.add_argument("--sales", required=True, help="Path to dc23Sales04.txt")
    ap.add_argument("--outdir", default="outputs", help="Output folder for CSV/PNG")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = _read_sales(args.sales)
    summary, sku_fill, cust_fill, ts = compute_fill_rates(df)

    # Save CSVs
    sku_fill.to_csv(outdir / "fillrate_by_sku.csv", index=False)
    cust_fill.to_csv(outdir / "fillrate_by_customer.csv", index=False)
    ts.to_csv(outdir / "fillrate_monthly.csv", index=False)

    # Plots
    plot_trend(ts, outdir / "fillrate_trend.png")
    plot_bottom_entities(sku_fill, "item", outdir / "fillrate_worst_skus.png", k=20)
    plot_bottom_entities(cust_fill, "cusnbr", outdir / "fillrate_worst_customers.png", k=20)

    # Console summary
    print("=== Order Fill Rate Summary ===")
    for k,v in summary.items():
        print(f"{k}: {v}")

    print(f"\nSaved to: {outdir.resolve()}")

if __name__ == "__main__":
    # Convenience default so you can just press Run in VS Code (adjust path as needed):
    import sys
    sys.argv = ["OrderFillRate.py", "--sales", "dc23Sales04.txt", "--outdir", "outputs"]
    main()

