"""Utility script to view and analyze saved portfolio weights.

Usage:
    python utils/view_weights.py --weights-csv logs/monthly/2025-01/final_test_weights_20250109_120000.csv
    python utils/view_weights.py --weights-csv logs/monthly/2025-01/final_test_weights_20250109_120000.csv --top-k 20
    python utils/view_weights.py --weights-csv logs/monthly/2025-01/final_test_weights_20250109_120000.csv --date 2024-06-15
"""
import argparse
import pandas as pd
import os


def load_weights(csv_path):
    """Load portfolio weights CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Weights file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    required_cols = ['date', 'ticker', 'weight']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    
    return df


def show_summary(df, top_k=10):
    """Show overall portfolio summary."""
    print(f"\n{'='*80}")
    print(f"PORTFOLIO WEIGHTS SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nDataset Info:")
    print(f"  Total days: {df['date'].nunique()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Unique tickers: {df['ticker'].nunique()}")
    print(f"  Total weight records: {len(df)}")
    
    # Average weights per ticker
    avg_weights = df.groupby('ticker')['weight'].agg(['mean', 'std', 'min', 'max', 'count']).round(6)
    avg_weights.columns = ['Avg Weight', 'Std Dev', 'Min', 'Max', 'Days Held']
    avg_weights['Avg Weight %'] = (avg_weights['Avg Weight'] * 100).round(2)
    avg_weights = avg_weights.sort_values('Avg Weight', ascending=False)
    
    print(f"\nTop {top_k} Holdings by Average Weight:")
    print(avg_weights.head(top_k).to_string())
    
    # Concentration metrics
    print(f"\nPortfolio Concentration:")
    avg_num_holdings = df.groupby('date')['ticker'].count().mean()
    print(f"  Average number of holdings per day: {avg_num_holdings:.1f}")
    
    top5_weight = avg_weights.head(5)['Avg Weight'].sum()
    top10_weight = avg_weights.head(10)['Avg Weight'].sum()
    print(f"  Top 5 holdings average weight: {top5_weight*100:.2f}%")
    print(f"  Top 10 holdings average weight: {top10_weight*100:.2f}%")


def show_date_weights(df, date):
    """Show portfolio weights for a specific date."""
    date_df = df[df['date'] == date].copy()
    
    if date_df.empty:
        available_dates = sorted(df['date'].unique())
        print(f"No data for date {date}")
        print(f"Available dates: {', '.join(available_dates[:10])}...")
        return
    
    date_df = date_df.sort_values('weight', ascending=False)
    date_df['weight_pct'] = (date_df['weight'] * 100).round(2)
    
    print(f"\n{'='*80}")
    print(f"PORTFOLIO WEIGHTS FOR {date}")
    print(f"{'='*80}")
    print(f"\nTotal holdings: {len(date_df)}")
    print(f"Total weight: {date_df['weight'].sum():.6f} (should be ~1.0)")
    
    print(f"\nTop Holdings:")
    print(date_df[['ticker', 'weight', 'weight_pct']].head(20).to_string(index=False))


def show_ticker_history(df, ticker):
    """Show weight history for a specific ticker."""
    ticker_df = df[df['ticker'] == ticker].copy()
    
    if ticker_df.empty:
        print(f"No data for ticker {ticker}")
        print(f"Available tickers: {', '.join(sorted(df['ticker'].unique())[:20])}...")
        return
    
    ticker_df = ticker_df.sort_values('date')
    ticker_df['weight_pct'] = (ticker_df['weight'] * 100).round(2)
    
    print(f"\n{'='*80}")
    print(f"WEIGHT HISTORY FOR {ticker}")
    print(f"{'='*80}")
    print(f"\nDays held: {len(ticker_df)}")
    print(f"Average weight: {ticker_df['weight'].mean()*100:.2f}%")
    print(f"Min weight: {ticker_df['weight'].min()*100:.2f}%")
    print(f"Max weight: {ticker_df['weight'].max()*100:.2f}%")
    print(f"Std dev: {ticker_df['weight'].std()*100:.2f}%")
    
    print(f"\nWeight Over Time:")
    print(ticker_df[['date', 'weight', 'weight_pct']].to_string(index=False))


def export_summary(df, output_path):
    """Export summary statistics to CSV."""
    avg_weights = df.groupby('ticker')['weight'].agg(['mean', 'std', 'min', 'max', 'count']).round(6)
    avg_weights.columns = ['avg_weight', 'std_weight', 'min_weight', 'max_weight', 'days_held']
    avg_weights['avg_weight_pct'] = (avg_weights['avg_weight'] * 100).round(2)
    avg_weights = avg_weights.sort_values('avg_weight', ascending=False)
    avg_weights.to_csv(output_path)
    print(f"\nExported summary to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="View and analyze portfolio weights")
    parser.add_argument('--weights-csv', required=True, help='Path to weights CSV file')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top holdings to show')
    parser.add_argument('--date', help='Show weights for specific date (YYYY-MM-DD)')
    parser.add_argument('--ticker', help='Show history for specific ticker')
    parser.add_argument('--export-summary', help='Export summary to CSV file')
    
    args = parser.parse_args()
    
    # Load data
    df = load_weights(args.weights_csv)
    
    # Show requested information
    if args.date:
        show_date_weights(df, args.date)
    elif args.ticker:
        show_ticker_history(df, args.ticker)
    else:
        show_summary(df, args.top_k)
    
    # Export if requested
    if args.export_summary:
        export_summary(df, args.export_summary)


if __name__ == '__main__':
    main()
