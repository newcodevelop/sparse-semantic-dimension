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
    "N_STEPS": [1000, 5000, 10000, 25000, 50000, 100000, 500000, 1000000], 
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
    
    # Log2 Loss (per token)
    log_probs = -torch.log2(smoothed_probs)
    
    # Average over sequence dimension (dim=1) to get loss per sequence
    loss_per_seq = log_probs.mean(dim=-1) 
    
    if reduction == 'mean':
        return loss_per_seq.mean()
    else:
        return loss_per_seq # Returns [batch_size]

def run_ablation_random_decoder(model_key, config):
    print(f"\n🧪 STARTING ABLATION (Random Decoder): {model_key}")
    
    # Clean memory
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        model = HookedTransformer.from_pretrained(config["name"], device=CONFIG["DEVICE"])
        vocab_size = model.cfg.d_vocab 
        random_baseline = math.log2(vocab_size)
        bounded_loss_cap = math.log2(vocab_size / CONFIG["ALPHA"])
        
        print(f"  Vocab Size: {vocab_size} | Random Baseline: {random_baseline:.2f} bits")

        # Load SAE just to get the Encoder and Dimensions
        sae, _, _ = SAE.from_pretrained(release=config["sae_release"], sae_id=config["sae_id"], device=CONFIG["DEVICE"])
        sae.eval()
        
        # --- ABLATION SETUP: RANDOM DECODER ---
        # We create a random matrix of shape [d_sae, d_model]
        # We normalize columns to unit norm to simulate a "well-behaved" but semantic-less dictionary
        d_sae = sae.cfg.d_sae
        d_model = model.cfg.d_model
        
        print(f"  Initializing Random Decoder ({d_sae} x {d_model})...")
        W_rand = torch.randn(d_sae, d_model, device=CONFIG["DEVICE"])
        # Normalize columns (dim=1 here because of how we multiply, or dim=0? 
        # Typically dictionary atoms are columns. W_dec usually maps feature -> output.
        # sae.decode(f) is roughly f @ W_dec. So rows of W_dec are the features? 
        # In sae_lens: W_dec is [d_sae, d_model]. 
        # So each row i is the vector for feature i.
        # We normalize each row to unit norm.
        W_rand = torch.nn.functional.normalize(W_rand, p=2, dim=1)
        
        # Scaling factor: Real SAE features often have specific magnitudes. 
        # To avoid the "Scale" critique, let's multiply by the average norm of the real decoder?
        # Or just leave it unit norm. Unit norm is standard for "dictionaries".
        
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
                
                # 1. Original Loss Vector
                orig_loss_vec = smoothed_bpd_loss(orig_logits, tokens, CONFIG["ALPHA"], vocab_size, reduction='none')
                
                # 2. Get Sparse Features (Using REAL Encoder)
                original_act = cache[hook_name]
                flat_act = original_act.reshape(-1, original_act.shape[-1])
                feature_acts = sae.encode(flat_act) # [Batch*Seq, d_sae]
                
                # 3. ABLATION: Random Decoding
                # Instead of sae.decode(), we use our random matrix
                recons_act = feature_acts @ W_rand # [Batch*Seq, d_model]
                # Add bias if the real SAE has a decoder bias (sae.b_dec)
                # For strict random ablation, we can ignore bias or use random bias. 
                # Let's add the REAL bias to be charitable (so we only ablate the directions).
                if hasattr(sae, 'b_dec'):
                    recons_act += sae.b_dec
                
                # Metrics
                k = (feature_acts > 0).float().sum(dim=-1).mean().item()
                c = torch.norm(recons_act, p=2, dim=-1).max().item()
                
                # 4. Proxy Run (Abutting)
                recons_reshaped = recons_act.reshape(original_act.shape)
                def hook_fn(activations, hook): return recons_reshaped
                proxy_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
                
                # 5. Proxy Loss Vector
                proxy_loss_vec = smoothed_bpd_loss(proxy_logits, tokens, CONFIG["ALPHA"], vocab_size, reduction='none')

            # Gap Calculation
            gap_vec = torch.abs(orig_loss_vec - proxy_loss_vec)
            gap = gap_vec.mean().item()
            proxy_loss = proxy_loss_vec.mean().item()
            
            results["proxy"].append(proxy_loss)
            results["gap"].append(gap)
            results["k"].append(k)
            results["C"].append(c)
            
            total_tokens += tokens.numel()
            pbar.update(tokens.numel())
            
            # Checkpoint
            target_N = CONFIG["N_STEPS"][current_step_idx]
            if total_tokens >= target_N:
                N = total_tokens
                avg_proxy = np.mean(results["proxy"])
                avg_gap = np.mean(results["gap"])
                avg_k = np.mean(results["k"])
                m = sae.cfg.d_sae
                
                # Bound Terms (Same formulas)
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
                    "Color": config["color"]
                })
                
                print(f"  N={N}: Bound={total_bound:.2f} (Baseline={random_baseline:.2f})")
                current_step_idx += 1
                if current_step_idx >= len(CONFIG["N_STEPS"]): break
                    
        except StopIteration: break
            
    pbar.close()
    del model, sae, W_rand
    return plot_points

# ==========================================
# 3. RUN AND AGGREGATE
# ==========================================
all_data = []
for key, cfg in CONFIG["MODELS"].items():
    all_data.extend(run_ablation_random_decoder(key, cfg))

df = pd.DataFrame(all_data)

# ==========================================
# 4. PLOTTING (Ablation Style)
# ==========================================
plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
fig, ax = plt.subplots(figsize=(10, 6))

colors = {m: c["color"] for m, c in CONFIG["MODELS"].items()}

# 1. Plot the Random Decoder Bounds
sns.lineplot(
    data=df, x="N", y="Total Bound", hue="Model", palette=colors, 
    style="Model", markers=True, markersize=8, linewidth=2.5, ax=ax
)

# 2. Add Baselines
x_vals = np.logspace(np.log10(min(CONFIG["N_STEPS"])), np.log10(max(CONFIG["N_STEPS"])), 100)

for model_name, group in df.groupby("Model"):
    baseline = group["Random Baseline"].iloc[0]
    color = colors[model_name]
    ax.axhline(y=baseline, color=color, linestyle='--', alpha=0.6, linewidth=1.5)
    # Label
    ax.text(max(CONFIG["N_STEPS"]), baseline - 0.5, f"Random ({model_name})", 
            color=color, va="top", ha="right", fontsize=9, fontweight='bold')

# 3. Formatting
ax.set_xscale("log")
ax.set_xlabel("Number of Samples (N)", fontweight='bold')
ax.set_ylabel("Generalization Bound (Bits)", fontweight='bold')
ax.set_title("Ablation: Random Decoder (Testing Semantic Necessity)", fontsize=14, pad=15)
ax.grid(True, which="both", linestyle='-', alpha=0.2)

# Important: Adjust Y-limit to show how bad it is. 
# Random decoder bounds will likely be 20-30 bits or more.
# We want to show they are ABOVE the baseline.
# Let's auto-scale but ensure baseline is visible.
y_min = df["Random Baseline"].min() - 2
y_max = df["Total Bound"].max() + 2
ax.set_ylim(y_min, y_max)

# Shading the "Non-Vacuous" zone (which we fail to reach)
# Anything BELOW the baseline is good. We are likely above.
# Let's shade the area BELOW the lowest baseline as "Success Zone" (unreached)
min_baseline = df["Random Baseline"].min()
ax.fill_between(x_vals, 0, min_baseline, color='green', alpha=0.05)
ax.text(min(CONFIG["N_STEPS"]), min_baseline - 1.0, "Non-Vacuous Zone (Unreached)", 
        color='green', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig("ablation_random_decoder_plot.png", dpi=300)
print("Plot saved as ablation_random_decoder_plot.png")