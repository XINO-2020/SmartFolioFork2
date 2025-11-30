import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from policy.policy import HGATNetwork

def test_raw_desire_viz():
    print("--- 🧠 Visualizing Part A: Raw Neural Desires (Amplified) ---")
    
    # 1. Setup Model
    num_stocks = 20
    lookback = 30
    input_dim = 6
    hidden_dim = 64
    
    adj_size = num_stocks * num_stocks
    # Feature Dim: Graphs + TimeSeries + PrevWeights
    feature_dim = (3 * adj_size) + (num_stocks * lookback * input_dim) + num_stocks
    
    print(f"1. Initializing Policy Network...")
    torch.manual_seed(42)
    policy = HGATNetwork(
        feature_dim=feature_dim,
        last_layer_dim_pi=num_stocks,
        last_layer_dim_vf=num_stocks,
        lookback=lookback,
        hidden_dim=hidden_dim,
        n_head=4
    )
    
    # --- CRITICAL STEP: SIMULATE TRAINING ---
    # We manually scale up the weights of the final output layer (The "Mouth")
    # This proves the network allows large values to pass through.
    print("   -> 💉 Injecting 'Experience' (Scaling weights by 100x)...")
    with torch.no_grad():
        policy.policy_net.output_head.weight *= 100.0
        policy.policy_net.output_head.bias += 0.0 # No bias change needed
    
    # 2. Create Dummy Input
    dummy_input = torch.randn(1, feature_dim)
    
    # 3. Forward Pass
    with torch.no_grad():
        raw_scores = policy.forward_actor(dummy_input)
        
    scores = raw_scores[0].numpy()
    
    print(f"2. Raw Scores range: [{scores.min():.2f}, {scores.max():.2f}]")
    
    # Verification Logic
    if scores.max() > 1.0 or scores.min() < 0.0:
        print("✅ PASS: The Policy can output strong opinions (Negative/Positive values).")
    else:
        print("❌ FAIL: The Policy is still constrained between 0-1 (Hidden Sigmoid detected).")

    # 4. Generate Visualization
    print("3. Generating Charts...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Chart 1: Distribution
    sns.histplot(scores, kde=True, ax=axes[0], color='purple', bins=15)
    axes[0].set_title("Distribution of Raw Desire (Simulated Trained Model)", fontsize=12)
    axes[0].set_xlabel("Logit Value (Score)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(0, color='black', linestyle='--', alpha=0.5, label='Neutral')
    
    # Chart 2: Conviction
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    colors = ['red' if x < 0 else 'green' for x in sorted_scores]
    
    axes[1].bar(range(num_stocks), sorted_scores, color=colors, alpha=0.8)
    axes[1].set_title("Conviction Levels (Green=Buy, Red=Short/Avoid)", fontsize=12)
    axes[1].set_xlabel("Stocks")
    axes[1].set_ylabel("Raw Score")
    axes[1].axhline(0, color='black', linewidth=1)
    
    plt.tight_layout()
    plt.savefig("test/policy_raw_desire.png")
    print("✅ Visualization Saved: test/policy_raw_desire.png")

if __name__ == "__main__":
    test_raw_desire_viz()