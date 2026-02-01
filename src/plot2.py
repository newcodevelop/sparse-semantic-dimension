import torch
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

# ==========================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================
CONFIG = {
    "N_STEPS": [1000, 5000, 10000, 25000, 50000, 100000], 
    "ALPHA": 0.5,
    "DELTA": 0.05,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    
    "MODELS": {
        "GPT-2 Small": {
            "name": "gpt2-small",
            "sae_release": "gpt2-small-res-jb", 
            "sae_id": "blocks.6.hook_resid_pre",
            "batch_size": 8,
            "color": "#1f77b4" # Blue
        },
        "Gemma-2B": {
            "name": "gemma-2b",
            "sae_release": "gemma-2b-res-jb", 
            "sae_id": "blocks.12.hook_resid_post", 
            "batch_size": 2,
            "color": "#d62728" # Red
        }
    }
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def smoothed_bpd_loss(logits, tokens, alpha, vocab_size, reduction='mean'):
    probs = torch.softmax(logits, dim=-1)
    probs_shifted = probs[:, :-1, :]
    tokens_shifted = tokens[:, 1:]
    true_probs = torch.gather(probs_shifted, -1, tokens_shifted.unsqueeze(-1)).squeeze(-1)
    smoothed_probs = (1 - alpha) * true_probs + (alpha / vocab_size)
    # return -torch.log2(smoothed_probs).mean()
    # Log2 Loss (per token)
    log_probs = -torch.log2(smoothed_probs)
    
    # Average over sequence dimension (dim=1) to get loss per sequence
    loss_per_seq = log_probs.mean(dim=-1) 
    
    if reduction == 'mean':
        return loss_per_seq.mean()
    else:
        return loss_per_seq # Returns [batch_size]

def run_experiment_for_model(model_key, config):
    print(f"\n🚀 STARTING EXPERIMENT: {model_key}")
    
    # Clean memory
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        model = HookedTransformer.from_pretrained(config["name"], device=CONFIG["DEVICE"])
        # Gemma vocab is ~256k, GPT-2 is ~50k. We fetch dynamically.
        vocab_size = model.cfg.d_vocab 
        random_baseline = math.log2(vocab_size)
        bounded_loss_cap = math.log2(vocab_size / CONFIG["ALPHA"])
        
        print(f"  Vocab Size: {vocab_size} | Random Baseline: {random_baseline:.2f} bits")

        sae, _, _ = SAE.from_pretrained(release=config["sae_release"], sae_id=config["sae_id"], device=CONFIG["DEVICE"])
        sae.eval()
    except Exception as e:
        print(f"Skipping {model_key} due to load error: {e}")
        return []

    # Stream Data
    dataset = load_dataset("NeelNanda/c4-code-20k", split="train", streaming=True)
    iterator = iter(dataset)
    
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
            
            with torch.no_grad():
                hook_name = config["sae_id"]
                orig_logits, cache = model.run_with_cache(tokens, names_filter=[hook_name])
                # NEW (104)
                orig_loss_vec = smoothed_bpd_loss(orig_logits, tokens, CONFIG["ALPHA"], vocab_size, reduction='none')
                # SAE
                original_act = cache[hook_name]
                flat_act = original_act.reshape(-1, original_act.shape[-1])
                feature_acts = sae.encode(flat_act)
                recons_act = sae.decode(feature_acts)
                
                k = (feature_acts > 0).float().sum(dim=-1).mean().item()
                c = torch.norm(recons_act, p=2, dim=-1).max().item()
                
                # Proxy
                recons_reshaped = recons_act.reshape(original_act.shape)
                def hook_fn(activations, hook): return recons_reshaped
                proxy_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
                # proxy_loss = smoothed_bpd_loss(proxy_logits, tokens, CONFIG["ALPHA"], vocab_size)
                proxy_loss_vec = smoothed_bpd_loss(proxy_logits, tokens, CONFIG["ALPHA"], vocab_size, reduction='none')

            # --- THE CORRECTION ---
            # Calculate gap for each sequence, THEN average
            gap_vec = torch.abs(orig_loss_vec - proxy_loss_vec)
            gap = gap_vec.mean().item()
            
            # We still track the empirical proxy risk (mean)
            proxy_loss = proxy_loss_vec.mean()
            # gap = abs(smoothed_bpd_loss(orig_logits, tokens, CONFIG["ALPHA"], vocab_size).item() - proxy_loss.item())
            
            results["proxy"].append(proxy_loss.item())
            results["gap"].append(gap)
            results["k"].append(k)
            results["C"].append(c)
            
            total_tokens += tokens.numel()
            pbar.update(tokens.numel())
            
            target_N = CONFIG["N_STEPS"][current_step_idx]
            if total_tokens >= target_N:
                N = total_tokens
                avg_proxy = np.mean(results["proxy"])
                avg_gap = np.mean(results["gap"])
                avg_k = np.mean(results["k"])
                m = sae.cfg.d_sae
                
                # Bound Terms
                ssd = avg_k * math.log((math.e * m) / avg_k)
                complexity = 4 * bounded_loss_cap * math.sqrt(ssd / N)
                delta_k = (6 * CONFIG["DELTA"]) / (math.pi**2 * avg_k**2)
                srm = bounded_loss_cap * math.sqrt(math.log(1/delta_k) / (2*N))
                hoeffdings = 1 * bounded_loss_cap * math.sqrt(math.log(1/CONFIG["DELTA"]) / (2*N))
                total_bound = avg_proxy + avg_gap + complexity + srm + hoeffdings
                
                plot_points.append({
                    "Model": model_key,
                    "N": N,
                    "Total Bound": total_bound,
                    "Random Baseline": random_baseline,
                    "Relative Bound": random_baseline - total_bound, # Positive is good
                    "Color": config["color"]
                })
                
                print(f"  N={N}: Bound={total_bound:.2f} (Baseline={random_baseline:.2f})")
                current_step_idx += 1
                if current_step_idx >= len(CONFIG["N_STEPS"]): break
                    
        except StopIteration: break
            
    pbar.close()
    del model, sae
    return plot_points

# ==========================================
# 3. RUN AND AGGREGATE
# ==========================================
all_data = []
for key, cfg in CONFIG["MODELS"].items():
    all_data.extend(run_experiment_for_model(key, cfg))

df = pd.DataFrame(all_data)

# ==========================================
# 4. PROFESSIONAL PLOTTING
# ==========================================
plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
fig, ax = plt.subplots(figsize=(10, 6))

# Define colors
colors = {m: c["color"] for m, c in CONFIG["MODELS"].items()}

# 1. Plot the Main Bound Curves
sns.lineplot(
    data=df, x="N", y="Total Bound", hue="Model", palette=colors, 
    style="Model", markers=True, markersize=8, linewidth=2.5, ax=ax
)

# 2. Add "Vacuous Lines" per model
# We define the x-range for the lines
x_vals = np.logspace(np.log10(min(CONFIG["N_STEPS"])), np.log10(max(CONFIG["N_STEPS"])), 100)

for model_name, group in df.groupby("Model"):
    baseline = group["Random Baseline"].iloc[0]
    color = colors[model_name]
    ax.axhline(y=baseline, color=color, linestyle='--', alpha=0.6, linewidth=1.5)
    ax.text(max(CONFIG["N_STEPS"]), baseline + 0.1, f"Random ({model_name})", 
            color=color, va="bottom", ha="right", fontsize=9, fontweight='bold')

# 3. Create the "Grey Area" (Vacuous Zone)
# Since baselines differ, we shade above the *highest* baseline to show the region 
# where ANY model would definitely be vacuous, or we shade between the curves and their baselines.
# Per your request, let's shade the area above the GPT-2 baseline (lowest) as "Potentially Vacuous"
# Or better: Shade above the specific baseline for the first model (GPT-2) as a reference.
# Let's shade the top of the graph starting from the lowest baseline (GPT-2 ~15.6)
min_baseline = df["Random Baseline"].min()
ax.fill_between(x_vals, min_baseline, 25, color='grey', alpha=0.1)
ax.text(min(CONFIG["N_STEPS"]), min_baseline + 0.3, "Vacuous Zone (for GPT-2)", 
        color='grey', fontsize=10, style='italic')

# 4. Formatting
ax.set_xscale("log")
ax.set_xlabel("Number of Samples (N)", fontweight='bold')
ax.set_ylabel("Generalization Bound (Bits)", fontweight='bold')
ax.set_title("Sparse Semantic Generalization Bound Convergence", fontsize=14, pad=15)
ax.grid(True, which="both", linestyle='-', alpha=0.2)
ax.set_ylim(5, 20) # Adjust based on your data range

# Save
plt.tight_layout()
plt.savefig("professional_bound_plot.png", dpi=300)
print("Plot saved as professional_bound_plot.png")

# Optional: Relative Plot (Cleaner comparison)
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.lineplot(data=df, x="N", y="Relative Bound", hue="Model", palette=colors, marker="o", ax=ax2)
ax2.axhline(0, color='black', linestyle='--', linewidth=2)
ax2.fill_between(x_vals, -5, 0, color='grey', alpha=0.1) # Shading the FAILURE zone (below 0)
ax2.text(min(CONFIG["N_STEPS"]), -0.5, "Vacuous Zone (Worse than Random)", color='grey')
ax2.set_xscale("log")
ax2.set_ylabel("Bits Saved over Random Guess (Higher is Better)")
ax2.set_title("Relative Generalization Performance")
plt.tight_layout()
plt.savefig("relative_bound_plot.png", dpi=300)
print("Relative plot saved.")
