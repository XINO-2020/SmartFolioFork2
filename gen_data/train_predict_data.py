import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
import torch
from pandas.tseries.offsets import MonthEnd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
from torch_geometric.data import Data
from tqdm import tqdm

from tools.pathway_features import compute_rolling_mean_std_pathway

STOCK_DATA_PATH = f"../dataset/"

feature_cols = ['close', 'open', 'high', 'low', 'prev_close', 'volume']
feature_cols_normalized = [f'{col}_normalized' for col in feature_cols]


# Set labels based on the desired prediction horizon
def get_label(df, horizon=1):
    df['label'] = None
    df.set_index('kdcode', inplace=True)
    # Group by kdcode and process each stock independently
    for code, group in df.groupby('kdcode'):
        # Use dt as the index and sort chronologically
        group = group.set_index('dt').sort_index()
        # Compute forward returns for the given horizon
        group['return'] = group['close'].shift(-horizon) / group['close'] - 1
        df.loc[code, 'label'] = group['return'].values
    df = df.dropna().reset_index()
    return df


def cal_rolling_mean_std(df, cal_cols=['close'], lookback=5, use_pathway=True):
    """
    Rolling mean/std per ticker. If use_pathway=True, compute via Pathway sliding windows;
    otherwise fallback to pandas groupby/rolling.
    """
    if use_pathway:
        try:
            return compute_rolling_mean_std_pathway(df, cal_cols, lookback)
        except Exception as exc:
            print(f"[warn] Pathway rolling failed ({exc}); falling back to pandas.")

    df = df.sort_values(by=['kdcode', 'dt'])  # Sort by ticker and date
    for col in cal_cols:
        df[f"{col}_mean"] = df.groupby('kdcode')[col].transform(
            lambda x: x.rolling(window=lookback, min_periods=1).mean()
        )
        df[f"{col}_std"] = df.groupby('kdcode')[col].transform(
            lambda x: x.rolling(window=lookback, min_periods=1).std()
        )
    df = df.dropna().reset_index()
    return df


# Cluster samples based on rolling features
def group_and_norm(df, base_cols, n):
    result = []
    kmeans = KMeans(n_clusters=n, random_state=42)
    df = df.sort_values(by=['kdcode', 'dt'])  # Sort by ticker and date
    for date, group in df.groupby('dt'):
        group = group.copy()
        cluster_features = group[base_cols].fillna(0)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(cluster_features)
        group['cluster'] = kmeans.fit_predict(features_scaled)
        group_sizes = group['cluster'].value_counts()
        small_clusters = group_sizes[group_sizes < 2].index
        for cluster in small_clusters:
            # Reassign under-populated clusters to their nearest neighbor
            mask = group['cluster'] == cluster
            cluster_data = group[mask]
            other_data = group[group['cluster'] != cluster]
            # Compute distances to every other cluster
            cluster_features = cluster_data[base_cols].values
            other_features = other_data[base_cols].values
            distances = np.linalg.norm(other_features[:, np.newaxis] - cluster_features, axis=2)  # Euclidean distance
            closest_cluster_indices = np.argmin(distances, axis=0)  # Index of the closest cluster
            # Move samples into the nearest existing cluster
            closest_clusters = other_data.iloc[closest_cluster_indices]['cluster'].values
            group.loc[mask, 'cluster'] = closest_clusters
        # Normalize features within each cluster
        for f in feature_cols:
            group[f'{f}_normalized'] = group.groupby('cluster')[f].transform(
                lambda x: (x - x.mean()) / x.std())
        result.append(group)
    return pd.concat(result)


def filter_code(df):
    dts = set(df['dt'])  # Set of all trade dates
    valid_codes = df.groupby('kdcode')['dt'].apply(set)  # Dates observed for each ticker
    # Keep tickers that trade on every date in the window
    result_codes = valid_codes[valid_codes.apply(lambda x: x == dts)].index.to_list()
    return result_codes


def get_relation_dt(str_year, str_month, stock_trade_dt_s):
    month_dts = [k for k in stock_trade_dt_s
                 if k > str_year + '-' + str_month and k < str_year + '-' + str_month + '-32']
    relation_dt = pd.to_datetime(month_dts[0]) + MonthEnd(1)  # Calendar month-end (not the last trading day)
    relation_dt = relation_dt.strftime('%Y-%m-%d')
    return relation_dt


def gen_mats_by_threshold(corr, threshold=0.2):
    pos_graph = nx.Graph(corr > threshold)
    neg_graph = nx.Graph(corr < -threshold)
    pos_adj = nx.adjacency_matrix(pos_graph)
    pos_adj.data = np.ones(pos_adj.data.shape)
    pos_adj = pos_adj.toarray()
    # Remove diagonal entries to drop self-loops
    pos_adj = pos_adj - np.diag(np.diag(pos_adj))
    neg_adj = nx.adjacency_matrix(neg_graph)
    neg_adj.data = np.ones(neg_adj.data.shape)
    neg_adj = neg_adj.toarray()
    # Remove diagonal entries to drop self-loops
    neg_adj = neg_adj - np.diag(np.diag(neg_adj))
    return pos_adj, neg_adj


def generate_train_predict_data_by_date(dt, df,
                                        relation_dt,
                                        stock_trade_dt_s_all,
                                        filter_code_s,
                                        market,
                                        horizon,
                                        relation_type,
                                        lookback=20,
                                        threshold=0.2,
                                        norm=True):
    # Build both time-series features and point-in-time snapshots
    ts_start = stock_trade_dt_s_all[stock_trade_dt_s_all.index(dt) - (lookback - 1)]
    df_ts = df.loc[df['dt'] <= dt]
    df_ts = df_ts.loc[df_ts['dt'] >= ts_start]

    ind = np.load(f"../dataset/{market}/industry.npy")
    ind = torch.from_numpy(ind).type(torch.float32)

    corr = pd.read_csv(f'../dataset/corr/{market}/{relation_dt}.csv', index_col=0)
    # Threshold correlation matrix to obtain positive/negative adjacency
    pos_adj, neg_adj = gen_mats_by_threshold(corr, threshold)
    corr = torch.from_numpy(np.array(corr)).type(torch.float32)
    pos = torch.from_numpy(np.array(pos_adj)).type(torch.float32)
    neg = torch.from_numpy(np.array(neg_adj)).type(torch.float32)

    ts_features = []
    features = []
    mask = []
    labels = []
    day_last_code = []
    for code in stock_code_s:
        df_ts_code = df_ts.loc[df_ts['kdcode'] == code]
        cols = feature_cols_normalized if norm else feature_cols
        ts_array = df_ts_code[cols].values
        df_code_dt = df_ts_code.loc[df_ts_code['dt'] == dt]
        array = df_code_dt[cols].values
        if ts_array.T.shape[1] == lookback:
            one = []
            ts_features.append(ts_array)
            features.append(array)
            mask.append(True)
            label = df_ts_code.loc[df_ts_code['dt'] == dt]['label'].values
            labels.append(label[0])
            one.append(code)
            one.append(dt)
            day_last_code.append(one)
    ts_features = np.array(ts_features)
    ts_features = torch.from_numpy(ts_features).type(torch.float32)
    features = np.array(features)
    features = torch.from_numpy(features).type(torch.float32)
    mask = [True] * len(labels)
    labels = torch.tensor(labels, dtype=torch.float32)
    # Create edge indices; correlation is symmetric so the upper triangle is sufficient
    edge_index = torch.triu_indices(ind.size(0), ind.size(0), offset=1)
    # Create the PyG Data object
    pyg_data = Data(x=features, edge_index=edge_index)
    # Attach industry/correlation strength as edge attributes
    pyg_data.edge_attr = ind[edge_index[0], edge_index[1]]  # Pull matching entries from the industry matrix

    result = {'corr': Variable(corr),
              'ts_features': Variable(ts_features),
              'features': Variable(features),
              'industry_matrix': Variable(ind),
              'pos_matrix': Variable(pos),
              'neg_matrix': Variable(neg),
              'pyg_data': pyg_data,
              'labels': Variable(labels),
              'mask': mask}

    # Replace NaN values if any slipped through
    for key, value in result.items():
        if isinstance(value, torch.Tensor):
            result[key] = torch.nan_to_num(value, nan=0.0)  # Replace NaNs with zeros
        elif isinstance(value, np.ndarray):
            result[key] = np.nan_to_num(value, nan=0.0)  # Replace NaNs with zeros

    save_path = f'../dataset/data_train_predict_{market}/{horizon}_{relation_type}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(f'../dataset/data_train_predict_{market}/{horizon}_{relation_type}/' + dt + '.pkl', 'wb') as f:
        pickle.dump(result, f)

    code_df = pd.DataFrame(columns=['kdcode', 'dt'], data=day_last_code)
    folder_path = f'../dataset/daily_stock_{market}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    code_df.to_csv(f'../dataset/daily_stock_{market}/' + dt + '.csv', header=True, index=False, encoding='utf_8_sig')


if __name__ == '__main__':
    # 2018-01-01 - 2024-12-12
    market = 'custom'
    train_start_date = '2018-01-01'
    train_end_date = '2022-12-31'
    eval_start_date = '2023-01-01'
    eval_end_date = '2023-12-31'
    test_start_date = '2024-01-01'
    test_end_date = '2024-12-31'
    relation_type = 'hy'

    horizon_s = [1]
    lookback_s = [5, 10, 20]

    df_row = pd.read_csv(f'{STOCK_DATA_PATH}{market}_org.csv')
    # Add supervised labels
    for horizon in horizon_s:
        df = get_label(df=df_row, horizon=horizon)
    # Compute rolling metrics
    df = cal_rolling_mean_std(df, cal_cols=['close', 'volume'], lookback=5)
    # Normalize features within each cluster
    df = group_and_norm(df, base_cols=['close_mean', 'close_std',
                                       'volume_mean', 'volume_std'], n=4)
    df_all = df.copy()
    df = df[(df['dt'] >= train_start_date) & (df['dt'] <= test_end_date)]
    stock_trade_dt_s_all = sorted(df_all['dt'].unique().tolist())
    stock_trade_dt_s = sorted(df['dt'].unique().tolist())
    stock_code_s = filter_code(df)
    for i in tqdm(range(len(stock_trade_dt_s))):
        dt = stock_trade_dt_s[i]
        relation_dt = get_relation_dt(str_year=dt[:4], str_month=dt[5:7], stock_trade_dt_s=stock_trade_dt_s)
        generate_train_predict_data_by_date(dt=dt, df=df_all,
                                            relation_dt=relation_dt,
                                            stock_trade_dt_s_all=stock_trade_dt_s_all,
                                            filter_code_s=stock_code_s,
                                            market=market,
                                            horizon=1,
                                            relation_type=relation_type,
                                            lookback=20,
                                            threshold=0.2,
                                            norm=True)

    print(1)
