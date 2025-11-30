import os
import time
import json
import argparse
import warnings
from datetime import datetime
import calendar
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
import torch
print(torch.cuda.is_available())
from dataloader.data_loader import *
from policy.policy import *
# from trainer.trainer import *
from stable_baselines3 import PPO
from trainer.irl_trainer import *
from torch_geometric.loader import DataLoader
from utils.risk_profile import build_risk_profile
from tools.pathway_temporal import discover_monthly_shards_with_pathway
from tools.pathway_monthly_builder import build_monthly_shards_with_pathway

import shutil
import pickle
from stable_baselines3.common.save_util import load_from_zip_file
from trainer.ptr_ppo import PTR_PPO

PATH_DATA = f'./dataset_default/'

def _copy_compatible_policy_weights(policy_module, loaded_state, checkpoint_path):
    """Load only the tensors that match by key and shape to avoid shape-mismatch crashes."""
    if not loaded_state:
        print(f"Warning: {checkpoint_path} did not contain policy weights; skipping weight transfer.")
        return 0

    current_state = policy_module.state_dict()
    matched = 0
    skipped = []

    for key, tensor in loaded_state.items():
        target_tensor = current_state.get(key)
        if target_tensor is None:
            continue
        if target_tensor.shape != tensor.shape:
            skipped.append((key, tuple(tensor.shape), tuple(target_tensor.shape)))
            continue
        if isinstance(tensor, torch.Tensor):
            current_state[key] = tensor.to(target_tensor.device)
        else:
            current_state[key] = torch.as_tensor(tensor, device=target_tensor.device)
        matched += 1

    policy_module.load_state_dict(current_state)

    if matched == 0:
        print(f"Warning: No compatible policy weights found in {checkpoint_path}; starting from scratch.")
    else:
        print(f"Loaded {matched} compatible policy tensors from {checkpoint_path}.")

    if skipped:
        preview = ", ".join(f"{name}: {src}->{dst}" for name, src, dst in skipped[:5])
        if len(skipped) > 5:
            preview += ", ..."
        print(
            f"Skipped {len(skipped)} tensors from {checkpoint_path} due to shape mismatches."
            f" Examples: {preview}"
        )

    return matched
    
def load_weights_into_new_model(
    path,
    env,
    device,
    policy_kwargs,
    ptr_mode=False,
    ptr_coef=0.1,
    prior_policy=None,
):
    def _extract_policy_state_dict():
        try:
            _, params, _ = load_from_zip_file(path, device=device)
        except ValueError as exc:
            print(f"Warning: load_from_zip_file failed ({exc}). Falling back to PPO.load...")
            temp_model = PPO.load(path, env=None, device=device)
            return temp_model.policy.state_dict()

        if params is None:
            return None
        if isinstance(params, dict) and "policy" in params:
            return params["policy"]
        return params

    policy_state = _extract_policy_state_dict()

    model_cls = PTR_PPO if ptr_mode else PPO
    init_kwargs = dict(
        policy=HGATActorCriticPolicy,
        env=env,
        policy_kwargs=policy_kwargs,
        device=device,
        **PPO_PARAMS,
    )
    if ptr_mode:
        init_kwargs["ptr_coef"] = ptr_coef
        init_kwargs["prior_policy"] = prior_policy

    model = model_cls(**init_kwargs)
    _copy_compatible_policy_weights(model.policy, policy_state, path)
    return model

def load_finrag_prior(weights_path, num_stocks, tickers_csv="tickers.csv"):
    """
    Load FinRAG weights from a JSON file and normalize them to a simplex vector.
    Supports payloads shaped as:
      - [w1, w2, ...]
      - {"weights": [...]} or {"scores": [...]}
      - {"TICKER": weight, ...} (ordered by tickers_csv when available)
    """
    if not weights_path:
        print("FinRAG weights path not provided; skipping prior initialization.")
        return None
    if not os.path.exists(weights_path):
        print(f"FinRAG weights path not found: {weights_path}; skipping prior initialization.")
        return None

    with open(weights_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    # Unwrap common container keys
    if isinstance(payload, dict) and ("weights" in payload or "scores" in payload):
        payload = payload.get("weights") or payload.get("scores")

    # Resolve to a list of weights
    weights = None
    if isinstance(payload, dict):
        # Map tickers to weights using the CSV order when available
        if os.path.exists(tickers_csv):
            tickers = pd.read_csv(tickers_csv)["ticker"].tolist()
        else:
            tickers = list(payload.keys())
        weights = [float(payload.get(ticker, 0.0)) for ticker in tickers]
    else:
        weights = list(payload)

    weights_arr = np.asarray(weights, dtype=np.float32)
    if weights_arr.shape[0] != num_stocks:
        print(
            f"FinRAG weights length {weights_arr.shape[0]} does not match num_stocks {num_stocks}; "
            "skipping prior initialization."
        )
        return None

    weights_arr = np.clip(weights_arr, 0.0, None)
    total = float(weights_arr.sum())
    if total <= 0:
        print("FinRAG weights sum to zero; skipping prior initialization.")
        return None

    prior = weights_arr / total
    print(f"Loaded FinRAG prior from {weights_path} (len={len(prior)})")
    return prior


def init_policy_bias_from_prior(model, prior_weights):
    """
    Initialize the policy action head bias so the mean action roughly matches the prior.
    Works for SB3 ActorCriticPolicy subclasses where action_net is a Linear layer.
    """
    if prior_weights is None:
        return
    policy = getattr(model, "policy", None)
    action_net = getattr(policy, "action_net", None)
    if action_net is None or not hasattr(action_net, "bias"):
        print("Policy action_net missing; cannot apply FinRAG prior bias.")
        return
    if action_net.bias.shape[0] != len(prior_weights):
        print(
            f"Action bias shape {action_net.bias.shape[0]} does not match prior length {len(prior_weights)}; "
            "skipping prior bias init."
        )
        return

    prior_logits = torch.log(torch.from_numpy(prior_weights + 1e-8)).to(action_net.bias.device)
    with torch.no_grad():
        action_net.bias.copy_(prior_logits)
    print("Initialized policy action bias from FinRAG prior.")


def _infer_month_dates(shard):
    """Infer month label, start, and end date strings from a manifest shard."""
    month_label = shard.get("month")
    month_start = shard.get("month_start") or shard.get("start_date") or shard.get("train_start_date")
    month_end = shard.get("month_end") or shard.get("end_date") or shard.get("train_end_date")

    # Normalise the month label
    parsed_month = None
    if month_label:
        for fmt in ("%Y-%m", "%Y-%m-%d"):
            try:
                parsed_month = datetime.strptime(month_label, fmt)
                break
            except ValueError:
                continue
    if parsed_month is None and month_start:
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                parsed_month = datetime.strptime(month_start, fmt)
                month_label = parsed_month.strftime("%Y-%m")
                break
            except ValueError:
                continue

    if parsed_month and not month_start:
        month_start = parsed_month.strftime("%Y-%m-01")

    if parsed_month and not month_end and month_start:
        last_day = calendar.monthrange(parsed_month.year, parsed_month.month)[1]
        month_end = parsed_month.replace(day=last_day).strftime("%Y-%m-%d")

    if not (month_label and month_start and month_end):
        raise ValueError(f"Unable to infer complete month information from shard: {shard}")

    return month_label, month_start, month_end

def select_replay_samples(model, env, dataset, k_percent=0.1):
    """
    Select top k% samples based on absolute reward magnitude (proxy for importance).
    """
    print("Selecting replay samples...")
    obs = env.reset()
    rewards = []

    env.reset()

    real_env = env.envs[0]
    max_steps = real_env.max_step
    
    step_rewards = []
    
    for i in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        step_rewards.append((i, abs(reward[0]))) # Store index and abs reward
        if done:
            break
            
    # Sort by absolute reward descending
    step_rewards.sort(key=lambda x: x[1], reverse=True)
    
    # Select top k
    num_to_select = int(len(step_rewards) * k_percent)
    selected_indices = [x[0] for x in step_rewards[:num_to_select]]
    
    # Retrieve data objects from dataset
    # dataset.data_all is the list
    selected_samples = [dataset.data_all[i] for i in selected_indices if i < len(dataset.data_all)]
    
    print(f"Selected {len(selected_samples)} replay samples from {len(dataset)} total.")
    return selected_samples
    
def fine_tune_month(args, manifest_path="monthly_manifest.json", bookkeeping_path=None, replay_buffer=None):
    """Fine-tune the PPO model on the latest unprocessed monthly shard."""
    manifest_file = manifest_path
    
    # --- 1. Manifest Loading & Discovery ---
    if not os.path.exists(manifest_file) and getattr(args, "discover_months_with_pathway", False):
        base_dir_guess = f'dataset_default/data_train_predict_{args.market}/{args.horizon}_{args.relation_type}/'
        try:
            shards = build_monthly_shards_with_pathway(
                base_dir_guess,
                manifest_file,
                min_days=getattr(args, "min_month_days", 10),
                cutoff_days=getattr(args, "month_cutoff_days", None),
            )
            print(f"Built manifest at {manifest_file} with {len(shards)} monthly shards from {base_dir_guess}")
        except Exception as exc:
            raise FileNotFoundError(f"Monthly manifest not found at {manifest_file} and Pathway build failed: {exc}")
    elif not os.path.exists(manifest_file):
        raise FileNotFoundError(f"Monthly manifest not found at {manifest_file}")

    with open(manifest_file, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    shards = manifest.get("monthly_shards", {})
    
    # Optional: Late discovery if manifest exists but is empty
    if (not shards) and getattr(args, "discover_months_with_pathway", False):
        base_dir_guess = manifest.get(
            "base_dir",
            f'dataset_default/data_train_predict_{args.market}/{args.horizon}_{args.relation_type}/',
        )
        try:
            discovered = build_monthly_shards_with_pathway(
                base_dir_guess,
                manifest_file,
                min_days=getattr(args, "min_month_days", 10),
                cutoff_days=getattr(args, "month_cutoff_days", None),
            )
            shards = discovered
            manifest["monthly_shards"] = discovered
            manifest["base_dir"] = base_dir_guess
            with open(manifest_file, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2)
            print(f"Discovered {len(discovered)} monthly shards via Pathway windows.")
        except Exception as exc:
            print(f"Pathway month discovery failed: {exc}")
            
    if not shards:
        raise ValueError("Manifest does not contain any 'monthly_shards'")

    # --- 2. Shard Parsing & Selection ---
    shards_list = []
    if isinstance(shards, dict):
        last_ft = manifest.get("last_fine_tuned_month")
        for idx, (month_label, rel_path) in enumerate(sorted(shards.items())):
            shard = {
                "month": month_label,
                "shard_path": rel_path,
            }
            shard["processed"] = bool(last_ft == month_label)
            shards_list.append(shard)
    else:
        shards_list = list(shards)

    target_month_label = getattr(args, "fine_tune_month", None)
    start_month_label = getattr(args, "fine_tune_start_month", None)
    start_month_dt = None
    if start_month_label:
        try:
            start_month_dt = datetime.strptime(start_month_label, "%Y-%m")
        except ValueError as exc:
            raise ValueError("--fine_tune_start_month must follow YYYY-MM format") from exc

    unprocessed = []
    for idx, shard in enumerate(shards_list):
        if shard.get("processed", False):
            continue
        try:
            month_label, month_start, month_end = _infer_month_dates(shard)
        except ValueError:
            continue
            
        month_dt = datetime.strptime(month_label, "%Y-%m")
        if target_month_label and month_label != target_month_label:
            continue
        if start_month_dt and month_dt < start_month_dt:
            continue
        unprocessed.append((idx, shard, month_label, month_start, month_end))

    if not unprocessed:
        if target_month_label:
            raise RuntimeError(f"Target month {target_month_label} was not found or already processed.")
        raise RuntimeError("No unprocessed monthly shards available for fine-tuning")

    # Pick the earliest unprocessed month (Sequential Continual Learning)
    def _month_sort_key(item):
        _, _, month_label, _, _ = item
        return datetime.strptime(month_label, "%Y-%m")

    shard_idx, shard, month_label, month_start, month_end = min(unprocessed, key=_month_sort_key)

    base_dir = (
        shard.get("data_dir")
        or shard.get("base_dir")
        or manifest.get("base_dir")
        or f'dataset_default/data_train_predict_{args.market}/{args.horizon}_{args.relation_type}/'
    )

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Monthly shard data directory not found: {base_dir}")

    # --- 3. Dataset & Replay Injection ---
    monthly_dataset = AllGraphDataSampler(
        base_dir=base_dir,
        date=True,
        train_start_date=month_start,
        train_end_date=month_end,
        mode="train",
    )

    if len(monthly_dataset) == 0:
        raise ValueError(f"Monthly dataset for {month_label} is empty (start={month_start}, end={month_end})")

    # [FEATURE] Inject replay buffer if available
    if replay_buffer:
        print(f"Injecting {len(replay_buffer)} samples from replay buffer into training data.")
        monthly_dataset.data_all.extend(replay_buffer)

    monthly_loader = DataLoader(
        monthly_dataset,
        batch_size=len(monthly_dataset),
        pin_memory=True,
        collate_fn=lambda x: x,
        drop_last=True,
    )

    env_init = create_env_init(args, data_loader=monthly_loader)

    # --- 4. Checkpoint Resolution ---
    previous_checkpoint = None
    # Find the strictly previous processed shard's checkpoint
    for prev_idx in range(shard_idx - 1, -1, -1):
        prev_shard = shards_list[prev_idx]
        prev_path = prev_shard.get("checkpoint_path") or prev_shard.get("checkpoint")
        if prev_shard.get("processed") and prev_path and os.path.exists(prev_path):
            previous_checkpoint = prev_path
            break

    manifest_last_ckpt = manifest.get("last_checkpoint_path")
    checkpoint_candidates = []
    # Order of preference for loading weights
    for candidate in (
        getattr(args, "resume_model_path", None),
        shard.get("checkpoint"),
        shard.get("checkpoint_path"),
        manifest_last_ckpt,
        previous_checkpoint,
        getattr(args, "baseline_checkpoint", None),
    ):
        if candidate and candidate not in checkpoint_candidates:
            checkpoint_candidates.append(candidate)

    checkpoint_path = next((p for p in checkpoint_candidates if os.path.exists(p)), None)

    if checkpoint_path is None:
        raise FileNotFoundError("No valid base checkpoint found for fine-tuning")

    print(f"Fine-tuning {checkpoint_path} on month {month_label} ({month_start} to {month_end}) for {args.fine_tune_steps} timesteps")
    
    # Determine lookback
    lookback = getattr(env_init, 'lookback', getattr(args, 'lookback', 30))
    if hasattr(env_init, 'envs'):
         lookback = getattr(env_init.envs[0], 'lookback', lookback)
    
    policy_kwargs = dict(
        last_layer_dim_pi=args.num_stocks,
        last_layer_dim_vf=args.num_stocks,
        n_head=8,
        hidden_dim=128,
        no_ind=(not args.ind_yn),
        no_neg=(not args.neg_yn),
        lookback=lookback,
    )

    # --- 5. Model Loading (PTR Logic) ---
    if getattr(args, "ptr_mode", False):
        print(f"Using PTR (Policy Transfer via Regularization) with coef={args.ptr_coef}")
        # Load the "prior" (frozen old policy)
        prior_model = load_weights_into_new_model(
            checkpoint_path,
            env_init,
            args.device,
            policy_kwargs,
            ptr_mode=False,
        )
        prior_policy = prior_model.policy
        # Load the "current" (trainable new policy)
        model = load_weights_into_new_model(
            checkpoint_path,
            env_init,
            args.device,
            policy_kwargs,
            ptr_mode=True,
            ptr_coef=args.ptr_coef,
            prior_policy=prior_policy,
        )
    else:
        # Standard PPO loading
        model = load_weights_into_new_model(
            checkpoint_path,
            env_init,
            args.device,
            policy_kwargs,
            ptr_mode=False,
        )

    model.set_env(env_init)
    model.learn(total_timesteps=getattr(args, "fine_tune_steps", 5000))

    # --- 6. New Replay Selection ---
    new_replay_samples = []
    if getattr(args, "ptr_mode", False):
        # Select high-value samples from the current month to carry forward
        new_replay_samples = select_replay_samples(model, env_init, monthly_dataset, k_percent=0.1)

    # --- 7. Save Result ---
    os.makedirs(args.save_dir, exist_ok=True)
    month_slug = month_label.replace("/", "-")
    out_path = os.path.join(args.save_dir, f"{args.model_name}_{month_slug}.zip")
    model.save(out_path)
    print(f"Saved fine-tuned checkpoint to {out_path}")

    # --- 8. Update Manifest (FIXED: Now before return) ---
    shard.update({
        "processed": True,
        "checkpoint_path": out_path,
        "processed_at": datetime.utcnow().isoformat(timespec="seconds"),
    })
    manifest["monthly_shards"][shard_idx] = shard
    manifest["last_fine_tuned_month"] = month_label
    manifest["last_checkpoint_path"] = out_path
    
    output_manifest = bookkeeping_path or manifest_file
    with open(output_manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Updated manifest at {output_manifest}")

    return out_path, new_replay_samples

def train_predict(args, predict_dt):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    data_dir = f'dataset_default/data_train_predict_{args.market}/{args.horizon}_{args.relation_type}/'
    train_dataset = AllGraphDataSampler(base_dir=data_dir, date=True,
                                        train_start_date=args.train_start_date, train_end_date=args.train_end_date,
                                        mode="train")
    val_dataset = AllGraphDataSampler(base_dir=data_dir, date=True,
                                      val_start_date=args.val_start_date, val_end_date=args.val_end_date,
                                      mode="val")
    test_dataset = AllGraphDataSampler(base_dir=data_dir, date=True,
                                       test_start_date=args.test_start_date, test_end_date=args.test_end_date,
                                       mode="test")
    
    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, collate_fn=lambda x: x,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), pin_memory=True)
    print(len(train_loader), len(val_loader), len(test_loader))

    # create or load model
    env_init = create_env_init(args, dataset=train_dataset)
    
    if args.policy == 'MLP':
        if getattr(args, 'resume_model_path', None) and os.path.exists(args.resume_model_path):
            print(f"Loading PPO model from {args.resume_model_path}")
            model = PPO.load(args.resume_model_path, env=env_init, device=args.device)
        else:
            model = PPO(policy='MlpPolicy',
                        env=env_init,
                        **PPO_PARAMS,
                        seed=args.seed,
                        device=args.device)
    elif args.policy == 'HGAT':
        # Determine lookback from environment or args to ensure consistency
        lookback = getattr(env_init, 'lookback', getattr(args, 'lookback', 30))
        if hasattr(env_init, 'envs'):
             lookback = getattr(env_init.envs[0], 'lookback', lookback)

        policy_kwargs = dict(
            last_layer_dim_pi=args.num_stocks,  
            last_layer_dim_vf=args.num_stocks,
            n_head=8,
            hidden_dim=128,
            no_ind=(not args.ind_yn),
            no_neg=(not args.neg_yn),
            lookback=lookback,
        )
        if getattr(args, 'resume_model_path', None) and os.path.exists(args.resume_model_path):
            print(f"Loading PPO model from {args.resume_model_path}")
            model = PPO.load(args.resume_model_path, env=env_init, device=args.device)
        else:
            model = PPO(policy=HGATActorCriticPolicy,
                        env=env_init,
                        policy_kwargs=policy_kwargs,
                        **PPO_PARAMS,
                        seed=args.seed,
                        device=args.device)
                        
    # Initialize policy bias with FinRAG prior if available
    init_policy_bias_from_prior(model, getattr(args, "finrag_prior", None))
    
    # Train
    train_model_and_predict(model, args, train_loader, val_loader, test_loader)

    # [FEATURE] Select initial replay samples from pre-training data if PTR mode is enabled
    if getattr(args, "ptr_mode", False):
        print("Selecting initial replay samples from pre-training data...")
        # We need to recreate the env with the full training set for selection
        env_selection = create_env_init(args, dataset=train_dataset)
        model.set_env(env_selection)
        
        initial_buffer = select_replay_samples(model, env_selection, train_dataset, k_percent=0.1)
        
        os.makedirs(args.save_dir, exist_ok=True)
        buffer_path = os.path.join(args.save_dir, f"replay_buffer_{args.market}.pkl")
        with open(buffer_path, "wb") as f:
            pickle.dump(initial_buffer, f)
        print(f"Saved initial replay buffer ({len(initial_buffer)} samples) to {buffer_path}")

    checkpoint_path = None
    try:
        os.makedirs(args.save_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        checkpoint_path = os.path.join(
            args.save_dir,
            f"ppo_{args.policy.lower()}_{args.market}_{ts}.zip",
        )
        model.save(checkpoint_path)
        print(f"Saved pre-training checkpoint to {checkpoint_path}")

        baseline_path = getattr(args, "baseline_checkpoint", None)
        if baseline_path:
            os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
            shutil.copy2(checkpoint_path, baseline_path)
            print(f"Updated baseline checkpoint at {baseline_path}")
    except Exception as exc:
        print(f"Failed to save pre-training checkpoint: {exc}")

    return model, checkpoint_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transaction ..")
    parser.add_argument("-device", "-d", default="cuda:0", help="gpu")
    parser.add_argument("-model_name", "-nm", default="SmartFolio", help="Model name used in checkpoints and logs")
    parser.add_argument("-horizon", "-hrz", default="1", help="Return prediction horizon in trading days")
    parser.add_argument("-relation_type", "-rt", default="hy", help="Correlation relation type label (default: hy)")
    parser.add_argument("-ind_yn", "-ind", default="y", help="Enable industry relation graph")
    parser.add_argument("-pos_yn", "-pos", default="y", help="Enable momentum relation graph")
    parser.add_argument("-neg_yn", "-neg", default="y", help="Enable reversal relation graph")
    parser.add_argument("-multi_reward_yn", "-mr", default="y", help="Enable multi-reward IRL head")
    parser.add_argument("-policy", "-p", default="MLP", help="Policy architecture identifier")
    
    # Continual learning / Resume
    parser.add_argument("--resume_model_path", default=None, help="Path to previously saved PPO model to resume from")
    parser.add_argument("--reward_net_path", default=None, help="Path to saved IRL reward network state_dict to resume from")
    parser.add_argument("--fine_tune_steps", type=int, default=5000, help="Timesteps for monthly fine-tuning when resuming")
    parser.add_argument("--save_dir", default="./checkpoints", help="Directory to save trained models")
    parser.add_argument("--baseline_checkpoint", default="./checkpoints/baseline.zip",
                        help="Destination checkpoint promoted after passing gating criteria")
    parser.add_argument("--promotion_min_sharpe", type=float, default=0.5,
                        help="Minimum Sharpe ratio required to promote a fine-tuned checkpoint")
    parser.add_argument("--promotion_max_drawdown", type=float, default=0.2,
                        help="Maximum acceptable drawdown (absolute fraction, e.g. 0.2 for 20%) for promotion")
    
    # Monthly Fine-tuning flags
    parser.add_argument("--run_monthly_fine_tune", action="store_true",
                        help="Run monthly fine-tuning using the manifest instead of full training")
    parser.add_argument("--discover_months_with_pathway", action="store_true",
                        help="When manifest lacks shards, group daily pickle files into monthly windows using Pathway")
    parser.add_argument("--month_cutoff_days", type=int, default=None,
                        help="Optional cutoff (days) to drop late daily files when building monthly shards via Pathway")
    parser.add_argument("--min_month_days", type=int, default=10,
                        help="Minimum number of daily files required to keep a discovered month window")
    parser.add_argument("--fine_tune_month", default=None,
                        help="Optional explicit month label (YYYY-MM) to fine-tune, overriding automatic selection")
    parser.add_argument("--fine_tune_start_month", default=None,
                        help="Skip shards earlier than this month label (YYYY-MM) when selecting the next month")
    
    # Expert & FinRAG
    parser.add_argument("--expert_cache_path", default=None,
                        help="Optional path to cache expert trajectories for reuse")
    parser.add_argument("--num_expert_trajectories", type=int, default=100,
                        help="Number of expert trajectories to generate for IRL pretraining")
    parser.add_argument("--finrag_weights_path", default=None,
                        help="Path to FinRAG weights JSON used to initialize the policy prior")
    
    # Training Hyperparameters
    parser.add_argument("--max_epochs", type=int, default=1, help="Number of IRL+RL epochs to run")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size for loaders and IRL")
    parser.add_argument("--n_steps", type=int, default=2048, help="Rollout horizon (environment steps) per PPO update cycle")
    parser.add_argument("--irl_epochs", type=int, default=50, help="Number of IRL training epochs")
    parser.add_argument("--rl_timesteps", type=int, default=10000, help="Number of RL timesteps for training")
    parser.add_argument(
        "--disable-tensorboard",
        action="store_true",
        help="Skip configuring TensorBoard logging to avoid importing the optional dependency.",
    )
    
    # Risk-adaptive reward parameters
    parser.add_argument("--risk_score", type=float, default=0.5, help="User risk score: 0=conservative, 1=aggressive")
    parser.add_argument("--dd_base_weight", type=float, default=1.0, help="Base weight for drawdown penalty")
    parser.add_argument("--dd_risk_factor", type=float, default=1.0, help="Risk factor k in β_dd(ρ) = β_base*(1+k*(1-ρ))")
    
    # PTR (Policy Transfer Regularization) parameters
    parser.add_argument("--ptr_mode", action="store_true", help="Enable Policy Transfer via Regularization (PTR) for continual learning")
    parser.add_argument("--ptr_coef", type=float, default=0.1, help="Coefficient for PTR loss (KL divergence penalty)")
    parser.add_argument("--use_ptr", action="store_true", help="Backward-compatible alias for --ptr_mode")
    parser.add_argument("--ptr_memory_size", type=int, default=500, help="Maximum number of samples retained in the PTR replay buffer")
    parser.add_argument("--ptr_priority_type", type=str, default="max", help="Replay buffer priority aggregation strategy")
    
    # Date ranges
    parser.add_argument("--train_start_date", default="2020-01-06", help="Start date for training")
    parser.add_argument("--train_end_date", default="2023-01-31", help="End date for training")
    parser.add_argument("--val_start_date", default="2023-02-01", help="Start date for validation")
    parser.add_argument("--val_end_date", default="2023-12-29", help="End date for validation")
    parser.add_argument("--test_start_date", default="2024-01-02", help="Start date for testing")
    parser.add_argument("--test_end_date", default="2024-12-26", help="End date for testing")
    
    args = parser.parse_args()
    args.market = 'custom' # Force market to custom as per your setup

    # Handle aliases and config
    if getattr(args, "use_ptr", False):
        args.ptr_mode = True

    PPO_PARAMS["batch_size"] = args.batch_size
    PPO_PARAMS["n_steps"] = args.n_steps

    if getattr(args, "disable_tensorboard", False):
        PPO_PARAMS["tensorboard_log"] = None
        print("TensorBoard logging disabled (--disable-tensorboard); PPO will not attempt to import tensorboard.")

    # Defaults
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.model_name = 'SmartFolio'
    args.relation_type = getattr(args, "relation_type", "hy") or "hy"
    args.seed = 123
    
    # Auto-detect input_dim
    try:
        data_dir_detect = f'dataset_default/data_train_predict_{args.market}/{args.horizon}_{args.relation_type}/'
        sample_files_detect = [f for f in os.listdir(data_dir_detect) if f.endswith('.pkl')]
        if sample_files_detect:
            import pickle
            sample_path_detect = os.path.join(data_dir_detect, sample_files_detect[0])
            with open(sample_path_detect, 'rb') as f:
                sample_data_detect = pickle.load(f)
            feats = sample_data_detect.get('features')
            if feats is not None:
                try:
                    shape = feats.shape
                except Exception:
                    try:
                        shape = feats.size()
                    except Exception:
                        shape = None
                if shape and len(shape) >= 2:
                    args.input_dim = shape[-1]
                    print(f"Auto-detected input_dim: {args.input_dim}")
                else:
                    print("Warning: could not determine input_dim from sample; falling back to 6")
                    args.input_dim = 6
            else:
                print("Warning: 'features' not found in sample; falling back to input_dim=6")
                args.input_dim = 6
        else:
            print(f"Warning: No sample files found in {data_dir_detect}; falling back to input_dim=6")
            args.input_dim = 6
    except Exception as e:
        print(f"Warning: input_dim auto-detection failed ({e}); falling back to 6")
        args.input_dim = 6
        
    args.ind_yn = True
    args.pos_yn = True
    args.neg_yn = True
    args.multi_reward = True
    
    # Hyperparams overrides
    args.irl_epochs = getattr(args, 'irl_epochs', 50)
    args.rl_timesteps = getattr(args, 'rl_timesteps', 10000)
    args.risk_score = getattr(args, 'risk_score', 0.5)
    args.dd_base_weight = getattr(args, 'dd_base_weight', 1.0)
    args.dd_risk_factor = getattr(args, 'dd_risk_factor', 1.0)
    args.risk_profile = build_risk_profile(args.risk_score)
    
    if not getattr(args, "expert_cache_path", None):
        args.expert_cache_path = os.path.join("dataset_default", "expert_cache")
    
    os.makedirs(args.save_dir, exist_ok=True)

    # Auto-detect num_stocks
    data_dir = f'dataset_default/data_train_predict_{args.market}/{args.horizon}_{args.relation_type}/'
    sample_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    if sample_files:
        import pickle
        sample_path = os.path.join(data_dir, sample_files[0])
        with open(sample_path, 'rb') as f:
            sample_data = pickle.load(f)
        args.num_stocks = sample_data['features'].shape[0]
        print(f"Auto-detected num_stocks for custom market: {args.num_stocks}")
    else:
        raise ValueError(f"No pickle files found in {data_dir} to determine num_stocks")

    # Load FinRAG prior
    args.finrag_prior = load_finrag_prior(args.finrag_weights_path, args.num_stocks)

    print("market:", args.market, "num_stocks:", args.num_stocks)
    
    # --- Execution Logic ---
    if args.run_monthly_fine_tune:
        # NOTE: Adjust this path if your manifest is located elsewhere
        manifest_path = "dataset_default/data_train_predict_custom/1_corr/monthly_manifest.json"
        replay_buffer = []
        
        # Load initial buffer if available
        buffer_path = os.path.join(args.save_dir, f"replay_buffer_{args.market}.pkl")
        if os.path.exists(buffer_path):
            with open(buffer_path, "rb") as f:
                replay_buffer = pickle.load(f)
            print(f"Loaded initial replay buffer with {len(replay_buffer)} samples.")
            
        while True:
            try:
                # Call fine_tune_month (returns path AND new samples)
                checkpoint, new_samples = fine_tune_month(args, manifest_path=manifest_path, replay_buffer=replay_buffer)
                print(f"Monthly fine-tuning complete. Checkpoint: {checkpoint}")
                
                # Update replay buffer with new samples
                if new_samples:
                    replay_buffer.extend(new_samples)
                    max_buffer = getattr(args, "ptr_memory_size", 500)
                    if len(replay_buffer) > max_buffer:
                        # Keep the most recent ones
                        replay_buffer = replay_buffer[-max_buffer:]
                    print(f"Replay buffer updated. Current size: {len(replay_buffer)}")
                    with open(buffer_path, "wb") as f:
                        pickle.dump(replay_buffer, f)
                    print(f"Persisted replay buffer to {buffer_path}")
                
                # Update resume_model_path so the next iteration picks up this model
                args.resume_model_path = checkpoint

                # Exit if user requested only a single specific month
                if getattr(args, "fine_tune_month", None):
                    print("Requested single-month fine-tune complete; exiting loop.")
                    break
            except RuntimeError as e:
                # Stop when no more months are left
                if "No unprocessed monthly shards" in str(e):
                    print("All months processed.")
                    break
                if getattr(args, "fine_tune_month", None) and "Target month" in str(e):
                    print(str(e))
                    break
                else:
                    raise e
    else:
        # Standard Train/Predict
        trained_model, checkpoint_path = train_predict(args, predict_dt='2024-12-30')
        if checkpoint_path is None:
            print("Warning: training completed but checkpoint could not be saved.")
        print(1)
