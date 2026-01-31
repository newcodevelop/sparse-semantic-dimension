import torch
print(str(torch.__version__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE
import math
from tqdm import tqdm
import gc

# import torch

# ==========================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================
CONFIG = {
    "N_STEPS": [1000, 10000, 20000, 30000, 50000, 100000, 500000], # Scale this up to 100k+ for the paper
    "ALPHA": 0.5,
    "DELTA": 0.05,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    
    # Define the Suite of Models to Test
    "MODELS": {
        "GPT-2 Small": {
            "name": "gpt2-small",
            "sae_release": "gpt2-small-res-jb", 
            "sae_id": "blocks.6.hook_resid_pre",
            "batch_size": 8
        },
        "Gemma-2B": {
            "name": "gemma-2b",
            "sae_release": "gemma-2b-res-jb", # check sae_lens availability
            "sae_id": "blocks.12.hook_resid_post", # Mid-layer
            "batch_size": 4
        },
        # Uncomment for Llama-3 (Requires High VRAM)
        "Llama-3-8B": {
            "name": "meta-llama/Meta-Llama-3-8B",
            "sae_release": "llama-3-8b-res-jb", # hypothetical release name
            "sae_id": "blocks.16.hook_resid_post",
            "batch_size": 1
        }
    }
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def smoothed_bpd_loss(logits, tokens, alpha, vocab_size):
    probs = torch.softmax(logits, dim=-1)
    probs_shifted = probs[:, :-1, :]
    tokens_shifted = tokens[:, 1:]
    true_probs = torch.gather(probs_shifted, -1, tokens_shifted.unsqueeze(-1)).squeeze(-1)
    smoothed_probs = (1 - alpha) * true_probs + (alpha / vocab_size)
    return -torch.log2(smoothed_probs).mean()

def run_experiment_for_model(model_key, config):
    """
    Runs the full experiment for a single model and returns the plot data.
    """
    print(f"\n🚀 STARTING EXPERIMENT: {model_key}")
    
    # 1. Load Model & SAE
    # Note: We clear memory between runs to avoid OOM
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        model = HookedTransformer.from_pretrained(config["name"], device=CONFIG["DEVICE"])
        sae, _, _ = SAE.from_pretrained(release=config["sae_release"], sae_id=config["sae_id"], device=CONFIG["DEVICE"])
        sae.eval()
    except Exception as e:
        print(f"Skipping {model_key} due to load error: {e}")
        return []

    vocab_size = model.cfg.d_vocab
    bounded_loss_cap = math.log2(vocab_size / CONFIG["ALPHA"])
    
    # 2. Stream Data
    dataset = load_dataset("NeelNanda/c4-code-20k", split="train", streaming=True)
    iterator = iter(dataset)
    
    # 3. Data Collection
    results = {"proxy": [], "gap": [], "k": [], "C": []}
    plot_points = []
    total_tokens = 0
    current_step_idx = 0
    
    pbar = tqdm(total=max(CONFIG["N_STEPS"]))
    
    while total_tokens < max(CONFIG["N_STEPS"]):
        try:
            batch = next(iterator)
            tokens = model.to_tokens(batch['text'])[:, :128]
            if tokens.shape[1] < 128: continue
            
            # --- METRICS CALCULATION START ---
            with torch.no_grad():
                # Original
                hook_name = config["sae_id"]
                orig_logits, cache = model.run_with_cache(tokens, names_filter=[hook_name])
                orig_loss = smoothed_bpd_loss(orig_logits, tokens, CONFIG["ALPHA"], vocab_size)
                
                # SAE
                original_act = cache[hook_name]
                flat_act = original_act.reshape(-1, original_act.shape[-1])
                
                feature_acts = sae.encode(flat_act)
                recons_act = sae.decode(feature_acts)
                
                # Metrics
                k = (feature_acts > 0).float().sum(dim=-1).mean().item()
                c = torch.norm(recons_act, p=2, dim=-1).max().item()
                
                # Proxy
                recons_reshaped = recons_act.reshape(original_act.shape)
                def hook_fn(activations, hook): return recons_reshaped
                proxy_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
                proxy_loss = smoothed_bpd_loss(proxy_logits, tokens, CONFIG["ALPHA"], vocab_size)
            # --- METRICS CALCULATION END ---

            # Store
            gap = abs(orig_loss.item() - proxy_loss.item())
            results["proxy"].append(proxy_loss.item())
            results["gap"].append(gap)
            results["k"].append(k)
            results["C"].append(c)
            
            num_toks = tokens.numel()
            total_tokens += num_toks
            pbar.update(num_toks)
            
            # Checkpoint
            target_N = CONFIG["N_STEPS"][current_step_idx]
            if total_tokens >= target_N:
                N = total_tokens
                avg_proxy = np.mean(results["proxy"])
                avg_gap = np.mean(results["gap"])
                avg_k = np.mean(results["k"])
                m = sae.cfg.d_sae
                
                # Calculate Bound Terms
                ssd = avg_k * math.log((math.e * m) / avg_k)
                complexity = 4 * bounded_loss_cap * math.sqrt(ssd / N)
                delta_k = (6 * CONFIG["DELTA"]) / (math.pi**2 * avg_k**2)
                srm = bounded_loss_cap * math.sqrt(math.log(1/delta_k) / (2*N))
                
                total_bound = avg_proxy + avg_gap + complexity + srm
                
                plot_points.append({
                    "Model": model_key,
                    "N": N,
                    "Total Bound": total_bound,
                    "Empirical Risk": avg_proxy + avg_gap,
                    "Sparsity (k)": avg_k
                })
                
                print(f"Snapshot {model_key} @ N={N}: Bound={total_bound:.4f} (k={avg_k:.1f})")
                current_step_idx += 1
                if current_step_idx >= len(CONFIG["N_STEPS"]):
                    break
                    
        except StopIteration:
            break
            
    pbar.close()
    
    # Cleanup memory
    del model, sae
    torch.cuda.empty_cache()
    gc.collect()
    
    return plot_points

# ==========================================
# 3. RUN ALL AND PLOT
# ==========================================
all_plot_data = []

for model_name, model_cfg in CONFIG["MODELS"].items():
    model_data = run_experiment_for_model(model_name, model_cfg)
    all_plot_data.extend(model_data)

# Create DataFrame
df = pd.DataFrame(all_plot_data)

# PLOTTING
plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")

# Plot Total Bound for each model
sns.lineplot(
    data=df, 
    x="N", 
    y="Total Bound", 
    hue="Model", 
    style="Model", 
    markers=True, 
    dashes=False, 
    linewidth=2.5
)

# Add Random Baselines (Rough approximation for standard vocab sizes)
plt.axhline(y=15.6, color='grey', linestyle=':', label="Random Baseline (GPT-2)")

plt.xscale("log")
plt.xlabel("Sample Size (N)", fontsize=12)
plt.ylabel("Generalization Bound (Bits)", fontsize=12)
plt.title("Scaling of Sparse Semantic Bound Across Models", fontsize=14)
plt.legend()
plt.savefig("multimodel_convergence.png", dpi=300)
print("Plot saved as multimodel_convergence.png")





