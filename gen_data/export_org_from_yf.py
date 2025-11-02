import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None


def download_to_org_csv(
    tickers: List[str],
    start: str,
    end: str,
    out_csv: str,
    drop_na_prev_close: bool = True,
) -> pd.DataFrame:
    """Download OHLCV from yfinance and save a CSV compatible with train_predict_data.py.

    Output columns: kdcode, dt, close, open, high, low, prev_close, volume
    """
    if yf is None:
        raise RuntimeError("yfinance is not installed. Please `pip install yfinance` and try again.")

    # Ensure list and non-empty
    tickers = [t.strip() for t in tickers if t and str(t).strip()]
    if not tickers:
        raise ValueError("No tickers provided")

    df = yf.download(tickers, start=start, end=end, auto_adjust=False, group_by="ticker", progress=False)

    rows = []
    if isinstance(df.columns, pd.MultiIndex):
        # Typical multi-ticker layout: columns like ('AAPL','Close'), ('MSFT','Volume') ...
        for t in tickers:
            if (t, "Close") not in df.columns and (t, "close") not in df.columns:
                # skip if ticker missing
                continue
            # Slice by ticker level
            try:
                sub = df[(t,)].copy()
            except Exception:
                # In some regions layout can be ('Price','Ticker'); try swaplevels once
                sub = df.swaplevel(0, 1, axis=1).sort_index(axis=1)
                if (t, "Close") not in sub.columns:
                    continue
                sub = sub[(t,)].copy()

            sub.columns = [c.lower() for c in sub.columns]
            sub["kdcode"] = t
            sub = sub.reset_index().rename(columns={"Date": "dt"})
            rows.append(sub)
    else:
        # Single-ticker layout
        sub = df.copy()
        sub.columns = [c.lower() for c in sub.columns]
        sub["kdcode"] = tickers[0]
        sub = sub.reset_index().rename(columns={"Date": "dt"})
        rows.append(sub)

    if not rows:
        raise ValueError("No data returned from yfinance for the provided tickers/date range")

    tall = pd.concat(rows, ignore_index=True)

    # Required columns check
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(tall.columns)
    for m in list(missing):
        tall[m] = np.nan

    # prev_close per ticker
    tall = tall.sort_values(["kdcode", "dt"])  # dt still datetime here
    tall["prev_close"] = tall.groupby("kdcode")["close"].shift(1)

    if drop_na_prev_close:
        tall = tall.dropna(subset=["prev_close"])  # remove each ticker's first row

    # Cast dt to string YYYY-MM-DD expected by pipeline
    tall["dt"] = pd.to_datetime(tall["dt"]).dt.strftime("%Y-%m-%d")

    out = tall[["kdcode", "dt", "close", "open", "high", "low", "prev_close", "volume"]].copy()
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def main():
    parser = argparse.ArgumentParser(description="Export yfinance OHLCV to *_org.csv for SmartFolio")
    parser.add_argument("--market", required=True, help="Market tag used in filename, e.g., sp500, dji, custom")
    parser.add_argument("--tickers_file", default=None, help="CSV with 'kdcode' or 'ticker' column")
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers if no file provided")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output_dir", default=None, help="Override output dir; default dataset_default")
    args = parser.parse_args()

    if args.tickers_file:
        df_t = pd.read_csv(args.tickers_file)
        col = "kdcode" if "kdcode" in df_t.columns else ("ticker" if "ticker" in df_t.columns else None)
        if col is None:
            raise ValueError("tickers_file must contain 'kdcode' or 'ticker' column")
        tickers = df_t[col].dropna().astype(str).tolist()
    elif args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        raise ValueError("Provide --tickers_file or --tickers")

    base_dir = args.output_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset_default"))
    out_csv = os.path.join(base_dir, f"{args.market}_org.csv")
    df_out = download_to_org_csv(tickers, args.start, args.end, out_csv)
    print(f"Wrote {len(df_out)} rows to {out_csv}")


if __name__ == "__main__":
    main()
