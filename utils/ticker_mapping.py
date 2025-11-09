"""Utilities for loading ticker mappings from the dataset.

This module handles loading the actual tickers that were successfully
downloaded and included in the dataset, respecting the stock ordering
used in pickle files.
"""
import os
from typing import Dict, List, Optional
import pandas as pd


def load_ticker_mapping(market: str, date: str, 
                       base_dir: str = "dataset_default") -> Optional[List[str]]:
    """Load ticker list for a specific date from daily_stock CSV.
    
    Args:
        market: Market name (e.g., 'custom', 'hs300')
        date: Date string in format 'YYYY-MM-DD'
        base_dir: Root directory of dataset
        
    Returns:
        List of tickers in the order they appear in features/labels tensors,
        or None if the file doesn't exist or is empty (holiday/market closed)
    """
    csv_path = os.path.join(base_dir, f"daily_stock_{market}", f"{date}.csv")
    
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        # Handle empty CSV (holidays)
        if df.empty or 'kdcode' not in df.columns:
            return None
        
        # Return sorted tickers (matching the order in pickle files)
        # build_dataset_yf.py uses sorted(df_local.kdcode.unique().tolist())
        tickers = sorted(df['kdcode'].unique().tolist())
        return tickers
    except Exception as e:
        print(f"Warning: Could not load tickers from {csv_path}: {e}")
        return None


def get_ticker_mapping_for_period(market: str, 
                                  start_date: str, 
                                  end_date: str,
                                  base_dir: str = "dataset_default") -> Dict[str, List[str]]:
    """Load ticker mappings for all dates in a period.
    
    Args:
        market: Market name
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        base_dir: Root directory of dataset
        
    Returns:
        Dictionary mapping date -> list of tickers
    """
    daily_dir = os.path.join(base_dir, f"daily_stock_{market}")
    
    if not os.path.exists(daily_dir):
        print(f"Warning: Daily stock directory not found: {daily_dir}")
        return {}
    
    date_ticker_map = {}
    
    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(daily_dir) if f.endswith('.csv')]
    
    # Filter by date range
    for csv_file in sorted(csv_files):
        date = csv_file.replace('.csv', '')
        
        # Simple date string comparison works for YYYY-MM-DD format
        if start_date <= date <= end_date:
            tickers = load_ticker_mapping(market, date, base_dir)
            if tickers is not None:
                date_ticker_map[date] = tickers
    
    return date_ticker_map


def infer_tickers_from_dataset(market: str,
                               relation_type: str = "hy",
                               horizon: str = "1",
                               base_dir: str = "dataset_default") -> Optional[List[str]]:
    """Infer ticker list from any available pickle file in the dataset.
    
    This is a fallback method when daily_stock CSV is not available.
    It loads the first available pickle file and extracts tickers from
    the daily_stock CSV for that date.
    
    Args:
        market: Market name
        relation_type: Relation type (default 'hy')
        horizon: Horizon (default '1')
        base_dir: Root directory of dataset
        
    Returns:
        List of tickers, or None if cannot be determined
    """
    data_dir = os.path.join(base_dir, 
                           f"data_train_predict_{market}",
                           f"{horizon}_{relation_type}")
    
    if not os.path.exists(data_dir):
        return None
    
    # Find any pickle file
    pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    
    if not pkl_files:
        return None
    
    # Use the first pickle file's date
    date = pkl_files[0].replace('.pkl', '')
    return load_ticker_mapping(market, date, base_dir)


def format_weights_with_tickers(weights: List[float], 
                                tickers: List[str],
                                threshold: float = 0.0001) -> pd.DataFrame:
    """Format portfolio weights with ticker names.
    
    Args:
        weights: List of portfolio weights (should sum to 1.0)
        tickers: List of ticker symbols
        threshold: Minimum weight to include in output (default 0.01%)
        
    Returns:
        DataFrame with columns ['ticker', 'weight', 'weight_pct']
        sorted by weight descending
    """
    if len(weights) != len(tickers):
        raise ValueError(f"Weights length ({len(weights)}) doesn't match tickers length ({len(tickers)})")
    
    df = pd.DataFrame({
        'ticker': tickers,
        'weight': weights,
        'weight_pct': [w * 100 for w in weights]
    })
    
    # Filter out near-zero allocations
    df = df[df['weight'] >= threshold]
    
    # Sort by weight descending
    df = df.sort_values('weight', ascending=False).reset_index(drop=True)
    
    return df
