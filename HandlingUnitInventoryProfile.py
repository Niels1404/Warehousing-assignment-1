# HandlingUnitInventoryProfile.py
# Inventory Profile 4.2 — Handling Unit Inventory (Pallet vs Case Reserve)
# Usage:
#   python HandlingUnitInventoryProfile.py --cases "DC23CASES AS OF 050210.xls"
# Optional outputs:
#   --out_csv_overall inv_handling_overall.csv
#   --out_csv_by_zone inv_handling_by_zone.csv
#   --out_png_overall inv_handling_overall.png
#   --out_png_by_zone inv_handling_by_zone.png

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Map WMS location-type codes to handling buckets
HANDLING_MAP = {
    # Case Reserve (case ID per box)
    "BLU": "Case Reserve",
    "KCN": "Case Reserve",
    # Pallet Reserve (case ID at pallet)
    "PLP": "Pallet Reserve",
    "PLC": "Pallet Reserve",
}

def read_cases(path: str) -> pd.DataFrame:
    """Load Cases reserve snapshot and normalize schema."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Cases file not found: {p.resolve()}")

    # Read Excel or CSV
    if p.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(p, dtype=str)
    else:
        df = pd.read_csv(p, dtype=str, sep=None, engine="python")

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Rename known variants
    rename_map = {
        "case number": "case",
        "casenumber": "case",
        "case_id": "case",
        "loc type": "location type",
        "loc_type": "location type",
        "locationtype": "location type",
    }
    df = df.rename(columns=rename_map)

    # Validate required columns
    expected = ["case", "sku", "quantity", "zone", "aisle", "bay", "level", "position", "location type"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing}. Found columns: {df.columns.tolist()}")

    # Types / cleaning
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)

    for c in ["zone", "location type"]:
        df[c] = df[c].astype(str).str.strip().str.upper()

    # Handling unit bucket
    df["handling_unit"] = df["location type"].map(HANDLING_MAP).fillna("Other/Unknown")

    # Zone empty -> UNK
    df["zone"] = df["zone"].replace({"": "UNK", "NA": "UNK"})
    return df

def summarize_overall(df: pd.DataFrame) -> pd.DataFrame:
    overall = (
        df.groupby("handling_unit", as_index=False)["quantity"]
          .sum()
          .sort_values("quantity", ascending=False, ignore_index=True)
    )
    total = overall["quantity"].sum()
    overall["share_pct"] = (overall["quantity"] / total * 100) if total > 0 else 0.0
    return overall, total

def summarize_by_zone(df: pd.DataFrame) -> pd.DataFrame:
    by_zone = df.groupby(["zone", "handling_unit"], as_index=False)["quantity"].sum()
    totals = by_zone.groupby("zone")["quantity"].transform("sum").replace(0, 1)
    by_zone["share_pct"] = by_zone["quantity"] / totals * 100
    # Order zones alphabetically for tidy plots
    by_zone = by_zone.sort_values(["zone", "handling_unit"]).reset_index(drop=True)
    return by_zone

def plot_overall(overall: pd.DataFrame, out_png: str):
    plt.figure(figsize=(8, 5))
    plt.bar(overall["handling_unit"], overall["quantity"], alpha=0.85)
    plt.title("Inventory by Handling Unit (Overall)")
    plt.xlabel("Handling Unit")
    plt.ylabel("Inventory (Selling Units)")
    for i, (qty, pct) in enumerate(zip(overall["quantity"], overall["share_pct"])):
        plt.text(i, qty, f"{qty:,.0f}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_by_zone(by_zone: pd.DataFrame, out_png: str):
    pivot = by_zone.pivot(index="zone", columns="handling_unit", values="quantity").fillna(0)
    pivot = pivot.sort_index()

    ax = pivot.plot(kind="bar", stacked=True, alpha=0.85, figsize=(11, 6))
    ax.set_title("Inventory by Handling Unit and Zone")
    ax.set_xlabel("Zone")
    ax.set_ylabel("Inventory (Selling Units)")
    ax.legend(title="Handling Unit")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def print_summary(overall: pd.DataFrame, total_qty: float):
    lines = [f"Total inventory (selling units): {total_qty:,.0f}"]
    for _, r in overall.iterrows():
        lines.append(f"• {r['handling_unit']}: {r['quantity']:,.0f} units ({r['share_pct']:.1f}%).")
    print("\n".join(lines))

def main():
    ap = argparse.ArgumentParser(description="Handling Unit Inventory Profile (4.2)")
    ap.add_argument("--cases", required=True, help='Path to "DC23CASES AS OF 050210.xls" (or CSV)')
    ap.add_argument("--out_csv_overall", default="inv_handling_overall.csv")
    ap.add_argument("--out_csv_by_zone", default="inv_handling_by_zone.csv")
    ap.add_argument("--out_png_overall", default="inv_handling_overall.png")
    ap.add_argument("--out_png_by_zone", default="inv_handling_by_zone.png")
    args = ap.parse_args()

    df = read_cases(args.cases)

    overall, total_qty = summarize_overall(df)
    by_zone = summarize_by_zone(df)

    # Save artifacts
    overall.to_csv(args.out_csv_overall, index=False)
    by_zone.to_csv(args.out_csv_by_zone, index=False)
    plot_overall(overall, args.out_png_overall)
    plot_by_zone(by_zone, args.out_png_by_zone)

    print(f"Saved: {Path(args.out_csv_overall).resolve()}")
    print(f"Saved: {Path(args.out_csv_by_zone).resolve()}")
    print(f"Saved: {Path(args.out_png_overall).resolve()}")
    print(f"Saved: {Path(args.out_png_by_zone).resolve()}")
    print()
    print_summary(overall, total_qty)

if __name__ == "__main__":
    import sys
    sys.argv = [
        "HandlingUnitInventoryProfile.py",
        "--cases", "DC23CASES AS OF 050210.xls"   # adjust path if needed
    ]
    main()
