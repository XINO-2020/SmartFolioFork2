"""Create visualizations from portfolio weights CSV.

Usage:
    python utils/plot_weights.py --weights-csv logs/monthly/2025-01/final_test_weights_20250109_120000.csv --output portfolio_viz.png
    python utils/plot_weights.py --weights-csv logs/monthly/2025-01/final_test_weights_20250109_120000.csv --top-k 15 --plot-type heatmap
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_top_holdings_over_time(df, top_k=10, output_path=None):
    """Plot top K holdings' weights over time."""
    # Get top K tickers by average weight
    top_tickers = df.groupby('ticker')['weight'].mean().nlargest(top_k).index
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for ticker in top_tickers:
        ticker_data = df[df['ticker'] == ticker].sort_values('date')
        ax.plot(ticker_data['date'], ticker_data['weight_pct'], 
               label=ticker, marker='o', markersize=3, alpha=0.7)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Weight (%)', fontsize=12)
    ax.set_title(f'Top {top_k} Holdings Over Time', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()


def plot_weights_heatmap(df, top_k=20, output_path=None):
    """Create heatmap of portfolio weights over time."""
    # Get top K tickers
    top_tickers = df.groupby('ticker')['weight'].mean().nlargest(top_k).index
    
    # Pivot to create matrix: dates x tickers
    df_pivot = df[df['ticker'].isin(top_tickers)].pivot_table(
        index='date', 
        columns='ticker', 
        values='weight_pct',
        fill_value=0
    )
    
    # Sort columns by average weight
    col_order = df_pivot.mean().sort_values(ascending=False).index
    df_pivot = df_pivot[col_order]
    
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(df_pivot.T, cmap='YlOrRd', cbar_kws={'label': 'Weight (%)'}, 
                ax=ax, linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Ticker', fontsize=12)
    ax.set_title(f'Portfolio Weights Heatmap (Top {top_k} Holdings)', 
                fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to: {output_path}")
    else:
        plt.show()


def plot_concentration_metrics(df, output_path=None):
    """Plot portfolio concentration metrics over time."""
    # Calculate metrics per date
    date_metrics = df.groupby('date').agg({
        'ticker': 'count',  # Number of holdings
        'weight': lambda x: x.nlargest(5).sum()  # Top 5 concentration
    }).rename(columns={'ticker': 'num_holdings', 'weight': 'top5_weight'})
    
    date_metrics['top5_weight_pct'] = date_metrics['top5_weight'] * 100
    date_metrics = date_metrics.sort_index()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot number of holdings
    ax1.plot(date_metrics.index, date_metrics['num_holdings'], 
            color='steelblue', marker='o', markersize=4, linewidth=2)
    ax1.set_ylabel('Number of Holdings', fontsize=12)
    ax1.set_title('Portfolio Diversification Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(date_metrics.index, date_metrics['num_holdings'], alpha=0.3, color='steelblue')
    
    # Plot top 5 concentration
    ax2.plot(date_metrics.index, date_metrics['top5_weight_pct'], 
            color='coral', marker='o', markersize=4, linewidth=2)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Top 5 Holdings Weight (%)', fontsize=12)
    ax2.set_title('Portfolio Concentration (Top 5 Holdings)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.fill_between(date_metrics.index, date_metrics['top5_weight_pct'], alpha=0.3, color='coral')
    
    # Rotate x-axis labels
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved concentration plot to: {output_path}")
    else:
        plt.show()


def plot_turnover(df, output_path=None):
    """Plot portfolio turnover over time."""
    dates = sorted(df['date'].unique())
    
    if len(dates) < 2:
        print("Not enough dates to calculate turnover")
        return
    
    turnovers = []
    for i in range(1, len(dates)):
        prev_date = dates[i-1]
        curr_date = dates[i]
        
        # Get weights for both dates
        prev_weights = df[df['date'] == prev_date].set_index('ticker')['weight']
        curr_weights = df[df['date'] == curr_date].set_index('ticker')['weight']
        
        # Align to same tickers (fill missing with 0)
        all_tickers = set(prev_weights.index) | set(curr_weights.index)
        prev_aligned = pd.Series(0.0, index=all_tickers)
        curr_aligned = pd.Series(0.0, index=all_tickers)
        
        prev_aligned.update(prev_weights)
        curr_aligned.update(curr_weights)
        
        # Turnover = sum of absolute weight changes / 2
        turnover = (prev_aligned - curr_aligned).abs().sum() / 2
        turnovers.append({'date': curr_date, 'turnover': turnover * 100})
    
    turnover_df = pd.DataFrame(turnovers)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(turnover_df['date'], turnover_df['turnover'], color='mediumseagreen', alpha=0.7)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Turnover (%)', fontsize=12)
    ax.set_title('Daily Portfolio Turnover', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add average line
    avg_turnover = turnover_df['turnover'].mean()
    ax.axhline(y=avg_turnover, color='red', linestyle='--', 
              label=f'Average: {avg_turnover:.2f}%', linewidth=2)
    ax.legend(fontsize=10)
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved turnover plot to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize portfolio weights")
    parser.add_argument('--weights-csv', required=True, help='Path to weights CSV file')
    parser.add_argument('--plot-type', default='line', 
                       choices=['line', 'heatmap', 'concentration', 'turnover', 'all'],
                       help='Type of plot to generate')
    parser.add_argument('--top-k', type=int, default=10, 
                       help='Number of top holdings to show')
    parser.add_argument('--output', help='Output file path (PNG/PDF)')
    
    args = parser.parse_args()
    
    # Load data
    if not os.path.exists(args.weights_csv):
        print(f"Error: File not found: {args.weights_csv}")
        return
    
    df = pd.read_csv(args.weights_csv)
    print(f"Loaded {len(df)} weight records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique tickers: {df['ticker'].nunique()}")
    
    # Generate plots
    if args.plot_type == 'line' or args.plot_type == 'all':
        output = args.output if args.plot_type != 'all' else args.output.replace('.png', '_line.png')
        plot_top_holdings_over_time(df, args.top_k, output)
    
    if args.plot_type == 'heatmap' or args.plot_type == 'all':
        output = args.output if args.plot_type != 'all' else args.output.replace('.png', '_heatmap.png')
        plot_weights_heatmap(df, args.top_k, output)
    
    if args.plot_type == 'concentration' or args.plot_type == 'all':
        output = args.output if args.plot_type != 'all' else args.output.replace('.png', '_concentration.png')
        plot_concentration_metrics(df, output)
    
    if args.plot_type == 'turnover' or args.plot_type == 'all':
        output = args.output if args.plot_type != 'all' else args.output.replace('.png', '_turnover.png')
        plot_turnover(df, output)


if __name__ == '__main__':
    main()
