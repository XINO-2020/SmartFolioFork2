#!/usr/bin/env python3
"""
FastAPI endpoints for SmartFolio inference and stability inspection.

Endpoints:
- POST /inference: run model inference over a date range, return metrics and weight CSV path.
- POST /stability: compute persistence metrics on a provided weights CSV.

This reuses the existing StockPortfolioEnv and model loading logic. It assumes the
dataset layout matches dataset_default/data_train_predict_{market}/{horizon}_{relation_type}.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from torch_geometric.loader import DataLoader
from stable_baselines3 import PPO

from dataloader.data_loader import AllGraphDataSampler
from env.portfolio_env import StockPortfolioEnv
from trainer.irl_trainer import process_data
from tools.weights_persistence_viz import load_tickers, rank_frequency, summarize, turnover
from main import fine_tune_month
from utils.ticker_mapping import get_ticker_mapping_for_period

import subprocess
import sys


class InferenceRequest(BaseModel):
    model_path: str = Field(..., description="Path to PPO checkpoint (.zip)")
    market: str = "custom"
    horizon: str = "1"
    relation_type: str = "hy"
    test_start_date: str = Field(..., description="YYYY-MM-DD")
    test_end_date: str = Field(..., description="YYYY-MM-DD")
    deterministic: bool = True
    ind_yn: bool = True
    pos_yn: bool = True
    neg_yn: bool = True
    lookback: int = 30
    input_dim: Optional[int] = None
    risk_score: float = 0.5
    output_dir: str = "./logs/api"


class PortfolioValuePoint(BaseModel):
    """Single portfolio value data point."""
    step: int
    portfolio_value: Optional[float]
    daily_return: Optional[float]


class StabilityRequest(BaseModel):
    weights_csv: str = Field(..., description="Path to weights CSV saved from inference")
    tickers_csv: Optional[str] = None
    top_k: int = 20
    top_k_overlap: int = 20


class FinetuneRequest(BaseModel):
    manifest_path: str = Field(..., description="Path to monthly_manifest.json")
    save_dir: str = "./checkpoints"
    device: str = "cuda:0"
    run_monthly_fine_tune: bool = True
    market: str = "custom"
    horizon: str = "1"
    relation_type: str = "hy"
    fine_tune_steps: int = 5000
    baseline_checkpoint: Optional[str] = None
    promotion_min_sharpe: float = 0.5
    promotion_max_drawdown: float = 0.2


def _dataset_dir(req: InferenceRequest) -> Path:
    return Path("dataset_default") / f"data_train_predict_{req.market}" / f"{req.horizon}_{req.relation_type}"


def _load_test_loader(req: InferenceRequest) -> DataLoader:
    data_dir = _dataset_dir(req)
    test_dataset = AllGraphDataSampler(
        base_dir=str(data_dir),
        date=True,
        test_start_date=req.test_start_date,
        test_end_date=req.test_end_date,
        mode="test",
    )
    if len(test_dataset) == 0:
        raise ValueError(f"Empty test dataset for range {req.test_start_date} to {req.test_end_date}")
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), pin_memory=True)
    return test_loader


def _run_inference(req: InferenceRequest) -> Dict[str, Any]:
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SERVER] Using device: {device}", flush=True)
    
    test_loader = _load_test_loader(req)
    print(f"[SERVER] Loaded test loader with {len(test_loader)} batches", flush=True)

    model_path = Path(req.model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"[SERVER] Loading model from {model_path}", flush=True)
    model = PPO.load(str(model_path), env=None, device=device)
    print(f"[SERVER] Model loaded successfully", flush=True)

    records: List[Dict[str, Any]] = []
    all_weights_data: List[Dict[str, Any]] = []
    portfolio_value_data: List[Dict[str, Any]] = []
    
    final_net_value: Optional[float] = None
    peak_value: Optional[float] = None
    current_value: Optional[float] = None

    print(f"[SERVER] Starting inference loop", flush=True)
    
    for batch_idx, data in enumerate(test_loader):
        print(f"[SERVER] Processing batch {batch_idx}", flush=True)
        sys.stdout.flush()
        
        corr, ts_features, features, ind, pos, neg, labels, pyg_data, mask = process_data(data, device=device)
        
        print(f"[SERVER] Data processed. ts_features shape: {ts_features.shape if ts_features is not None else 'None'}", flush=True)
        
        # Extract actual input_dim from ts_features shape
        if ts_features is not None:
            if len(ts_features.shape) == 4:
                actual_input_dim = ts_features.shape[-1]
            elif len(ts_features.shape) == 3:
                actual_input_dim = ts_features.shape[-1]
            else:
                actual_input_dim = features.shape[-1]
        else:
            actual_input_dim = features.shape[-1]
        
        print(f"[SERVER] actual_input_dim: {actual_input_dim}", flush=True)
        
        args_stub = argparse.Namespace(
            risk_score=req.risk_score,
            ind_yn=req.ind_yn,
            pos_yn=req.pos_yn,
            neg_yn=req.neg_yn,
            lookback=req.lookback,
            input_dim=actual_input_dim,
        )
        
        print(f"[SERVER] Creating StockPortfolioEnv...", flush=True)
        sys.stdout.flush()
        
        env_test = StockPortfolioEnv(
            args=args_stub,
            corr=corr,
            ts_features=ts_features,
            features=features,
            ind=ind,
            pos=pos,
            neg=neg,
            returns=labels,
            pyg_data=pyg_data,
            benchmark_return=None,
            mode="test",
            ind_yn=req.ind_yn,
            pos_yn=req.pos_yn,
            neg_yn=req.neg_yn,
            risk_profile={"risk_score": req.risk_score},
        )

        print(f"[SERVER] Environment created successfully", flush=True)
        
        obs_test = env_test.reset()
        print(f"[SERVER] Environment reset. obs shape: {obs_test.shape}", flush=True)
        
        max_step = len(labels)
        print(f"[SERVER] Running {max_step} inference steps", flush=True)

        for step in range(max_step):
            action, _states = model.predict(obs_test, deterministic=req.deterministic)
            obs_test, reward, done, info = env_test.step(action)
            if step % 10 == 0:
                print(f"[SERVER]   Step {step}/{max_step}", flush=True)
            if done:
                print(f"[SERVER]   Done at step {step}", flush=True)
                break

        print(f"[SERVER] Inference complete for batch {batch_idx}", flush=True)
        
        metrics, benchmark_metrics = env_test.evaluate()
        metrics_record = {"batch": batch_idx}
        metrics_record.update(metrics)
        if benchmark_metrics:
            metrics_record.update({f"benchmark_{k}": v for k, v in benchmark_metrics.items()})
        records.append(metrics_record)

        net_value_history = env_test.get_net_value_history()
        daily_returns_history = env_test.get_daily_returns_history()
        
        final_net_value = float(net_value_history[-1]) if len(net_value_history) > 0 else None
        peak_value = float(env_test.peak_value)
        current_value = float(env_test.net_value)

        weights_array = env_test.get_weights_history()
        if weights_array.size > 0:
            num_steps, num_stocks = weights_array.shape
            step_labels = getattr(env_test, "dates", list(range(num_steps)))
            for step_idx in range(num_steps):
                weights = weights_array[step_idx]
                step_value = step_labels[step_idx] if step_idx < len(step_labels) else step_idx
                
                portfolio_value = float(net_value_history[step_idx]) if step_idx < len(net_value_history) else None
                daily_return = float(daily_returns_history[step_idx]) if step_idx < len(daily_returns_history) else None
                
                for idx, weight in enumerate(weights):
                    if weight > 0.0001:
                        all_weights_data.append({
                            "run_id": f"api_run_{batch_idx}",
                            "batch": batch_idx,
                            "date": None,
                            "step": step_value,
                            "ticker": f"stock_{idx}",
                            "weight": float(weight),
                            "weight_pct": float(weight * 100),
                            "portfolio_value": portfolio_value,
                            "daily_return": daily_return,
                        })
                
                portfolio_value_data.append({
                    "run_id": f"api_run_{batch_idx}",
                    "batch": batch_idx,
                    "step": step_value,
                    "portfolio_value": portfolio_value,
                    "daily_return": daily_return,
                })

    out_dir = Path(req.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    weights_csv = out_dir / f"weights_{req.market}_{req.test_start_date}_{req.test_end_date}.csv"
    if all_weights_data:
        pd.DataFrame(all_weights_data).to_csv(weights_csv, index=False)
    
    portfolio_csv = out_dir / f"portfolio_values_{req.market}_{req.test_start_date}_{req.test_end_date}.csv"
    if portfolio_value_data:
        pd.DataFrame(portfolio_value_data).to_csv(portfolio_csv, index=False)

    portfolio_summary = [
        PortfolioValuePoint(
            step=int(pv["step"]),
            portfolio_value=pv["portfolio_value"],
            daily_return=pv["daily_return"]
        )
        for pv in portfolio_value_data
    ]

    return {
        "metrics": records,
        "weights_csv": str(weights_csv) if all_weights_data else None,
        "portfolio_values_csv": str(portfolio_csv) if portfolio_value_data else None,
        "portfolio_values_summary": portfolio_summary,
        "final_portfolio_value": final_net_value,
        "peak_value": peak_value,
        "current_value": current_value,
    }



def _stability_metrics(req: StabilityRequest) -> Dict[str, Any]:
    weights_path = Path(req.weights_csv).expanduser()
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights CSV not found: {weights_path}")
    df = pd.read_csv(weights_path)
    tickers = load_tickers(req.tickers_csv, expected_count=df["ticker"].nunique() if "ticker" in df else None)
    summarize(df, req.top_k, tickers)

    rf = rank_frequency(df, req.top_k)
    rf_path = weights_path.with_name(weights_path.stem + "_rank_frequency.csv")
    if not rf.empty:
        rf.to_csv(rf_path, index_label="rank")

    steps = sorted(df["step"].unique())
    tickers_list = tickers if tickers else sorted(df["ticker"].unique())
    matrix = np.zeros((len(steps), len(tickers_list)), dtype=np.float32)
    ticker_to_idx = {t: i for i, t in enumerate(tickers_list)}
    step_to_idx = {s: i for i, s in enumerate(steps)}
    for _, row in df.iterrows():
        ti = ticker_to_idx.get(row["ticker"], None)
        si = step_to_idx.get(row["step"], None)
        if ti is None or si is None:
            continue
        matrix[si, ti] = row["weight"]

    to = turnover(matrix)
    cos = None
    overlaps = None
    if matrix.shape[0] >= 2:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        normed = matrix / norms
        cos = (normed[1:] * normed[:-1]).sum(axis=1)
        k = min(req.top_k_overlap, matrix.shape[1])
        overlaps_list = []
        for i in range(matrix.shape[0] - 1):
            top_t = set(np.argsort(-matrix[i])[:k])
            top_prev = set(np.argsort(-matrix[i + 1])[:k])
            inter = len(top_t & top_prev)
            union = len(top_t | top_prev)
            overlaps_list.append(inter / union if union > 0 else 0.0)
        overlaps = np.array(overlaps_list)

    return {
        "rank_frequency_csv": str(rf_path) if rf is not None else None,
        "turnover": {
            "mean": float(to.mean()) if to.size else None,
            "std": float(to.std()) if to.size else None,
            "min": float(to.min()) if to.size else None,
            "max": float(to.max()) if to.size else None,
        },
        "cosine": {
            "mean": float(cos.mean()) if cos is not None else None,
            "std": float(cos.std()) if cos is not None else None,
            "min": float(cos.min()) if cos is not None else None,
            "max": float(cos.max()) if cos is not None else None,
        },
        "top_k_overlap": {
            "k": int(min(req.top_k_overlap, matrix.shape[1])) if matrix.size else None,
            "mean": float(overlaps.mean()) if overlaps is not None else None,
            "std": float(overlaps.std()) if overlaps is not None else None,
            "min": float(overlaps.min()) if overlaps is not None else None,
            "max": float(overlaps.max()) if overlaps is not None else None,
        },
        "weights_csv": str(weights_path),
    }


app = FastAPI(title="SmartFolio API", version="0.1.0")


@app.post("/inference")
def inference(req: InferenceRequest):
    try:
        result = _run_inference(req)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/stability")
def stability(req: StabilityRequest):
    try:
        result = _stability_metrics(req)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/finetune")
def finetune(req: FinetuneRequest):
    try:
        args = argparse.Namespace(
            device=req.device,
            model_name="SmartFolio",
            horizon=req.horizon,
            relation_type=req.relation_type,
            ind_yn=True,
            pos_yn=True,
            neg_yn=True,
            multi_reward_yn=True,
            resume_model_path=None,
            reward_net_path=None,
            fine_tune_steps=req.fine_tune_steps,
            save_dir=req.save_dir,
            baseline_checkpoint=req.baseline_checkpoint,
            promotion_min_sharpe=req.promotion_min_sharpe,
            promotion_max_drawdown=req.promotion_max_drawdown,
            run_monthly_fine_tune=True,
            discover_months_with_pathway=False,
            month_cutoff_days=None,
            min_month_days=10,
            expert_cache_path=None,
            irl_epochs=50,
            rl_timesteps=10000,
            risk_score=0.5,
            dd_base_weight=1.0,
            dd_risk_factor=1.0,
            market=req.market,
            seed=123,
            input_dim=6,
            ind=True,
            pos=True,
            neg=True,
            relation="hy",
        )
        ckpt = fine_tune_month(args, manifest_path=req.manifest_path)
        return {"checkpoint": ckpt}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Run SmartFolio FastAPI server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("api.server:app", host=args.host, port=args.port, reload=False)