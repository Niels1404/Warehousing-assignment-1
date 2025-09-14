# LinesPerOrderProfile.py
# Usage: python LinesPerOrderProfile.py --sales dc23Sales04.txt

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def read_sales(path: str) -> pd.DataFrame:
    """
    Load dc23Sales04.txt and normalize column names.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sales file not found: {path.resolve()}")

    df = pd.read_csv(
        path,
        sep=None,
        engine="python",
        dtype={"ordnbr": str, "linenbr": str},
        parse_dates=["txndate"],
        keep_default_na=False
    )

    df.columns = [c.strip().lower() for c in df.columns]
    return df

def build_lines_per_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count number of unique lines per order.
    """
    # Ensure columns exist
    if "ordnbr" not in df.columns or "linenbr" not in df.columns:
        raise KeyError(f"Required columns not found. Columns are: {df.columns.tolist()}")

    # Count distinct lines per order
    lines_per_order = (
        df.groupby("ordnbr")["linenbr"]
          .nunique()
          .reset_index(name="num_lines")
    )

    return lines_per_order

def plot_lines_distribution(lines_per_order: pd.DataFrame, out_png: str):
    """
    Plot histogram of lines per order.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(lines_per_order["num_lines"], bins=50, edgecolor="black", alpha=0.7)

    plt.title("Lines-per-Order Profile")
    plt.xlabel("Number of Lines in Order")
    plt.ylabel("Number of Orders")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def summarize(lines_per_order: pd.DataFrame) -> str:
    avg_lines = lines_per_order["num_lines"].mean()
    max_lines = lines_per_order["num_lines"].max()
    median_lines = lines_per_order["num_lines"].median()

    # % of single-line orders
    single_line_pct = (lines_per_order["num_lines"] == 1).mean() * 100

    return (f"• Average lines per order: {avg_lines:.2f}\n"
            f"• Median lines per order: {median_lines}\n"
            f"• Maximum lines in a single order: {max_lines}\n"
            f"• % of single-line orders: {single_line_pct:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Build Lines-per-Order Profile")
    parser.add_argument("--sales", required=True, help="Path to dc23Sales04.txt")
    parser.add_argument("--out_csv", default="lines_per_order.csv", help="Output CSV path")
    parser.add_argument("--out_png", default="lines_per_order.png", help="Output PNG path")
    args = parser.parse_args()

    df = read_sales(args.sales)
    lines_per_order = build_lines_per_order(df)

    # Save table
    lines_per_order.to_csv(args.out_csv, index=False)

    # Plot distribution
    plot_lines_distribution(lines_per_order, args.out_png)

    print(f"Saved CSV -> {Path(args.out_csv).resolve()}")
    print(f"Saved PNG -> {Path(args.out_png).resolve()}")
    print()
    print("Summary:")
    print(summarize(lines_per_order))

if __name__ == "__main__":
    import sys
    sys.argv = [
        "CustomerOrderProfiles(2).py",
        "--sales", "dc23Sales04.txt"  # adjust path if needed
    ]
    main()
