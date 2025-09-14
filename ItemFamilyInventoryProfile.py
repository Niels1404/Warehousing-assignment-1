# ItemFamilyInventoryProfile.py
# Inventory Profile 4.1 — Item-Family Inventory (Vendor as Family)
# Usage (typical):
#   python ItemFamilyInventoryProfile.py \
#       --active "DC23ACTIVE AS OF 050210.xls" \
#       --cases  "DC23CASES AS OF 050210.xls" \
#       --items  "ITEMDATAV2.txt" \
#       --top_n 20 --top_n_zone 10

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Helpers
# ---------------------------

def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p.resolve()}")
    if p.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(p, dtype=str)
    else:
        df = pd.read_csv(p, dtype=str, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def _norm_sku(s: pd.Series) -> pd.Series:
    # Normalize SKU for joining: strip, uppercase, remove whitespace
    return s.astype(str).str.strip().str.upper().str.replace(r"\s+", "", regex=True)

# ---------------------------
# Readers & normalizers
# ---------------------------

def read_active(path: str) -> pd.DataFrame:
    df = _read_table(path)

    # Expected schema variants
    rename_map = {
        "sku id": "sku",
        "item": "sku",
        "itemid": "sku",
        "item #": "sku",
        "item#": "sku",
        "qty": "quantity",
        "qty_eaches": "quantity",
        "loc type": "location type",
        "loc_type": "location type",
        "locationtype": "location type",
    }
    df = df.rename(columns=rename_map)

    needed = ["sku", "quantity", "zone"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"[ACTIVE] Missing columns {missing}. Found: {df.columns.tolist()}")

    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["zone"] = df["zone"].astype(str).str.strip().str.upper().replace({"": "UNK", "NA": "UNK"})
    df["sku_norm"] = _norm_sku(df["sku"])

    # Optional vendor passthrough if present
    for cand in ["vendor id", "vendor_id", "vnid", "vendor"]:
        if cand in df.columns:
            df["vendor_id"] = df[cand].astype(str).str.strip()
            break

    df["layer"] = "Active"
    cols = ["sku", "sku_norm", "quantity", "zone", "layer"]
    if "vendor_id" in df.columns:
        cols.append("vendor_id")
    return df[cols]

def read_cases(path: str) -> pd.DataFrame:
    df = _read_table(path)

    rename_map = {
        "case number": "case",
        "casenumber": "case",
        "case_id": "case",
        "loc type": "location type",
        "loc_type": "location type",
        "locationtype": "location type",
        "item": "sku",
        "sku id": "sku",
        "itemid": "sku",
        "item #": "sku",
        "item#": "sku",
        "qty": "quantity",
        "qty_eaches": "quantity",
    }
    df = df.rename(columns=rename_map)

    needed = ["sku", "quantity", "zone", "location type"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"[CASES] Missing columns {missing}. Found: {df.columns.tolist()}")

    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["zone"] = df["zone"].astype(str).str.strip().str.upper().replace({"": "UNK", "NA": "UNK"})
    df["sku_norm"] = _norm_sku(df["sku"])

    for cand in ["vendor id", "vendor_id", "vnid", "vendor"]:
        if cand in df.columns:
            df["vendor_id"] = df[cand].astype(str).str.strip()
            break

    df["layer"] = "Reserve"
    cols = ["sku", "sku_norm", "quantity", "zone", "layer"]
    if "vendor_id" in df.columns:
        cols.append("vendor_id")
    return df[cols]

def read_items(path: str) -> pd.DataFrame:
    df = _read_table(path)
    rename_map = {
    "vendor id": "vendor_id",
    "vendor_id": "vendor_id",
    "vnid": "vendor_id",
    "sku id": "sku",
    "item": "sku",
    "itemid": "sku",
    "item #": "sku",
    "item#": "sku",
    "item number": "sku",   # <-- ADD THIS
    }
    df = df.rename(columns=rename_map)
    if "sku" not in df.columns or "vendor_id" not in df.columns:
        print("[ITEMS] Could not find 'sku' and 'vendor_id' columns. Columns:", df.columns.tolist())
        return pd.DataFrame(columns=["sku", "sku_norm", "vendor_id"])
    df["sku"] = df["sku"].astype(str).str.strip()
    df["vendor_id"] = df["vendor_id"].astype(str).str.strip()
    df["sku_norm"] = _norm_sku(df["sku"])
    return df[["sku", "sku_norm", "vendor_id"]].drop_duplicates()

# ---------------------------
# Aggregation / vendor attach
# ---------------------------

def attach_vendor(inventory: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    inv = inventory.copy()
    has_vendor = "vendor_id" in inv.columns

    if not has_vendor:
        inv = inv.merge(items[["sku_norm", "vendor_id"]], on="sku_norm", how="left")

    total_skus = inv["sku_norm"].nunique()
    matched_skus = inv.loc[inv["vendor_id"].notna(), "sku_norm"].nunique() if "vendor_id" in inv.columns else 0
    rate = (matched_skus / total_skus * 100) if total_skus else 0
    print(f"[MATCH] Vendor matched for {matched_skus}/{total_skus} unique SKUs ({rate:.1f}%).")

    # Fallback: try leading-zero strip if match rate is very low
    if rate < 5 and "sku" in inv.columns and not has_vendor:
        print("[MATCH] Low match rate detected. Trying leading-zero fallback…")
        inv["sku_norm2"] = inv["sku"].astype(str).str.lstrip("0").str.upper()
        items2 = items.copy()
        items2["sku_norm2"] = items2["sku"].astype(str).str.lstrip("0").str.upper()
        inv = inv.merge(items2[["sku_norm2", "vendor_id"]], on="sku_norm2", how="left", suffixes=("", "_lz"))
        inv["vendor_id"] = inv["vendor_id"].fillna(inv["vendor_id_lz"])
        inv.drop(columns=["sku_norm2", "vendor_id_lz"], errors="ignore", inplace=True)
        matched2 = inv["vendor_id"].notna().sum()
        print(f"[MATCH] Fallback match rows with vendor: {matched2}")

    inv["vendor_id"] = inv["vendor_id"].fillna("UNKNOWN_FAMILY")

    # ASCII-safe labels to avoid glyph warnings
    inv["vendor_id_plot"] = (
        inv["vendor_id"].astype(str).str.replace(r"[^\x20-\x7E]", "", regex=True).str.strip()
    )
    return inv

def summarize_family_overall(inv: pd.DataFrame) -> pd.DataFrame:
    overall = (
        inv.groupby("vendor_id", as_index=False)["quantity"]
           .sum()
           .sort_values("quantity", ascending=False, ignore_index=True)
    )
    total = overall["quantity"].sum()
    overall["share_pct"] = (overall["quantity"] / total * 100) if total > 0 else 0.0
    # carry plot label
    labels = (
        inv.drop_duplicates(subset=["vendor_id"])  # any representative label
          .set_index("vendor_id")["vendor_id_plot"]
    )
    overall["vendor_id_plot"] = overall["vendor_id"].map(labels).fillna(overall["vendor_id"])
    return overall, total

def summarize_family_by_zone(inv: pd.DataFrame) -> pd.DataFrame:
    by_zone = inv.groupby(["zone", "vendor_id", "vendor_id_plot"], as_index=False)["quantity"].sum()
    totals = by_zone.groupby("zone")["quantity"].transform("sum").replace(0, 1)
    by_zone["share_pct"] = by_zone["quantity"] / totals * 100
    by_zone = by_zone.sort_values(["zone", "quantity"], ascending=[True, False]).reset_index(drop=True)
    return by_zone

# ---------------------------
# Plots
# ---------------------------

def plot_top_families(overall: pd.DataFrame, out_png: str, top_n: int = 20):
    top = overall.head(top_n).copy()
    if len(overall) > top_n:
        others_qty = overall["quantity"].iloc[top_n:].sum()
        others_share = 100 - top["share_pct"].sum()
        top.loc[len(top)] = {
            "vendor_id": "Others",
            "vendor_id_plot": "Others",
            "quantity": others_qty,
            "share_pct": others_share
        }

    labels = top["vendor_id_plot"]
    plt.figure(figsize=(11, 6))
    plt.bar(labels, top["quantity"], alpha=0.85)
    plt.title(f"Inventory by Item Family (Vendor) — Top {top_n} + Others")
    plt.xlabel("Vendor (Family)")
    plt.ylabel("Inventory (Selling Units)")
    plt.xticks(rotation=45, ha="right")
    for i, (q, pct) in enumerate(zip(top["quantity"], top["share_pct"])):
        plt.text(i, q, f"{q:,.0f}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_family_by_zone(by_zone: pd.DataFrame, out_png: str, top_n: int = 10):
    label_col = "vendor_id_plot"
    fam_order = (
        by_zone.groupby(label_col)["quantity"]
               .sum()
               .sort_values(ascending=False)
               .head(top_n)
               .index.tolist()
    )
    df = by_zone.copy()
    df["family_plot"] = df[label_col].where(df[label_col].isin(fam_order), "Others")

    pivot = df.pivot_table(index="zone", columns="family_plot", values="quantity", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(sorted(pivot.index), axis=0)
    pivot = pivot[pivot.sum().sort_values(ascending=False).index]  # order columns by total

    ax = pivot.plot(kind="bar", stacked=True, alpha=0.88, figsize=(12, 6))
    ax.set_title(f"Inventory by Item Family (Vendor) and Zone — Top {top_n} families")
    ax.set_xlabel("Zone")
    ax.set_ylabel("Inventory (Selling Units)")
    ax.legend(title="Family (Vendor)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Inventory Profile 4.1 — Item-Family Inventory (Vendor as Family)")
    ap.add_argument("--active", required=True, help='Path to "DC23ACTIVE AS OF 050210.xls" (or CSV)')
    ap.add_argument("--cases",  required=True, help='Path to "DC23CASES AS OF 050210.xls" (or CSV)')
    ap.add_argument("--items",  default=None,      help='Path to "ITEMDATAV2.txt" (optional but recommended)')
    ap.add_argument("--top_n",  type=int, default=20, help="Top-N families to show in overall bar")
    ap.add_argument("--top_n_zone", type=int, default=10, help="Top-N families to stack by zone")
    ap.add_argument("--out_csv_overall", default="inv_family_overall.csv")
    ap.add_argument("--out_csv_by_zone", default="inv_family_by_zone.csv")
    ap.add_argument("--out_png_overall", default="inv_family_overall.png")
    ap.add_argument("--out_png_by_zone", default="inv_family_by_zone.png")
    args = ap.parse_args()

    active = read_active(args.active)
    cases  = read_cases(args.cases)
    inv = pd.concat([active, cases], ignore_index=True)

    if args.items:
        items = read_items(args.items)
    else:
        items = pd.DataFrame(columns=["sku", "sku_norm", "vendor_id"])

    inv = attach_vendor(inv, items)

    overall, total_qty = summarize_family_overall(inv)
    by_zone = summarize_family_by_zone(inv)

    overall.to_csv(args.out_csv_overall, index=False)
    by_zone.to_csv(args.out_csv_by_zone, index=False)

    plot_top_families(overall, args.out_png_overall, top_n=args.top_n)
    plot_family_by_zone(by_zone, args.out_png_by_zone, top_n=args.top_n_zone)

    print(f"Saved: {Path(args.out_csv_overall).resolve()}")
    print(f"Saved: {Path(args.out_csv_by_zone).resolve()}")
    print(f"Saved: {Path(args.out_png_overall).resolve()}")
    print(f"Saved: {Path(args.out_png_by_zone).resolve()}")
    print()
    print(f"Total inventory (selling units): {total_qty:,.0f}")
    for _, r in overall.head(5).iterrows():
        print(f"• {r['vendor_id_plot']}: {r['quantity']:,.0f} units ({r['share_pct']:.1f}%).")

if __name__ == "__main__":
    import sys
    sys.argv = [
        "ItemFamilyInventoryProfile.py",
        "--active", "DC23ACTIVE AS OF 050210.xls",
        "--cases",  "DC23CASES AS OF 050210.xls",
        "--items",  "ITEMDATAV2.txt",
        "--top_n",  "20",
        "--top_n_zone", "10",
    ]
    main()
