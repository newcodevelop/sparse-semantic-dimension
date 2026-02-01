import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE
from tqdm import tqdm
import gc

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "N_TOKENS": 50_000, 
    "ALPHA": 0.5,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    
    "MODELS": {
        "GPT-2 Small": {
            "name": "gpt2-small",
            "sae_release": "gpt2-small-res-jb", 
            "sae_id": "blocks.6.hook_resid_pre",
            "batch_size": 16,
            "color": "#1f77b4"
        },
        "Gemma-2B": {
            "name": "gemma-2b",
            "sae_release": "gemma-2b-res-jb", 
            "sae_id": "blocks.12.hook_resid_post", 
            "batch_size": 4, 
            "color": "#d62728"
        }
    }
}

# ==========================================
# 2. HELPER FUNCTION
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

def run_shuffling_experiment(model_key, config):
    print(f"\n🧪 STARTING ABLATION (Shuffling): {model_key}")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        model = HookedTransformer.from_pretrained(config["name"], device=CONFIG["DEVICE"])
        vocab_size = model.cfg.d_vocab
        sae, _, _ = SAE.from_pretrained(release=config["sae_release"], sae_id=config["sae_id"], device=CONFIG["DEVICE"])
        sae.eval()
    except Exception as e:
        print(f"Skipping {model_key}: {e}")
        return []

    dataset = load_dataset("NeelNanda/c4-code-20k", split="train", streaming=True)
    iterator = iter(dataset)
    
    results = []
    total_tokens = 0
    pbar = tqdm(total=CONFIG["N_TOKENS"])
    
    batch_size = config["batch_size"]

    while total_tokens < CONFIG["N_TOKENS"]:
        # --- FIXED BATCHING LOGIC ---
        batch_texts = []
        try:
            for _ in range(batch_size):
                item = next(iterator)
                text = item['text'] if 'text' in item else item['content']
                batch_texts.append(text)
        except StopIteration:
            if not batch_texts: break
            
        # Tokenize list of strings -> [Batch, Seq]
        tokens = model.to_tokens(batch_texts)
        if tokens.shape[1] < 128: continue
        tokens = tokens[:, :128]
        # -----------------------------
            
        with torch.no_grad():
            hook_name = config["sae_id"]
            
            # 1. Original Loss
            orig_logits, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            orig_loss_vec = smoothed_bpd_loss(orig_logits, tokens, CONFIG["ALPHA"], vocab_size, reduction='none')
            
            # 2. Get Real Features
            original_act = cache[hook_name]
            flat_act = original_act.reshape(-1, original_act.shape[-1])
            feature_acts = sae.encode(flat_act) 
            
            # --- A: REAL SEMANTICS ---
            recons_real = sae.decode(feature_acts).reshape(original_act.shape)
            def hook_real(activations, hook): return recons_real
            proxy_logits_real = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_real)])
            loss_real_vec = smoothed_bpd_loss(proxy_logits_real, tokens, CONFIG["ALPHA"], vocab_size, reduction='none')
            
            # --- B: SHUFFLED SEMANTICS ---
            # Shuffle feature indices (permute columns)
            perm_idx = torch.randperm(feature_acts.shape[1], device=CONFIG["DEVICE"])
            feature_acts_shuffled = feature_acts[:, perm_idx]
            
            recons_shuffled = sae.decode(feature_acts_shuffled).reshape(original_act.shape)
            def hook_shuffled(activations, hook): return recons_shuffled
            proxy_logits_shuff = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_shuffled)])
            loss_shuff_vec = smoothed_bpd_loss(proxy_logits_shuff, tokens, CONFIG["ALPHA"], vocab_size, reduction='none')
            
            # 3. Calculate Gaps (Vectors of size [Batch])
            gap_real_vec = torch.abs(orig_loss_vec - loss_real_vec).cpu().numpy()
            gap_shuff_vec = torch.abs(orig_loss_vec - loss_shuff_vec).cpu().numpy()
            print(gap_real_vec.shape, gap_shuff_vec.shape)
            # Store
            for g_r, g_s in zip(gap_real_vec, gap_shuff_vec):
                results.append({"Model": model_key, "Condition": "Real SAE", "Gap (Bits)": g_r})
                results.append({"Model": model_key, "Condition": "Shuffled Features", "Gap (Bits)": g_s})
        
        total_tokens += tokens.numel()
        pbar.update(tokens.numel())
            
    pbar.close()
    del model, sae
    return results

# ==========================================
# 3. RUN EXPERIMENT
# ==========================================
all_results = []
for key, cfg in CONFIG["MODELS"].items():
    res = run_shuffling_experiment(key, cfg)
    all_results.extend(res)

df = pd.DataFrame(all_results)

# ==========================================
# 4. PLOTTING
# ==========================================
print("\nGenerating Histogram Plot...")
plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
sns.set_theme(style="white", context="paper", font_scale=1.2)

g = sns.FacetGrid(df, col="Model", hue="Condition", height=5, aspect=1.3,
                  palette={"Real SAE": "#2ca02c", "Shuffled Features": "#d62728"},
                  sharex=False) 

g.map(sns.kdeplot, "Gap (Bits)", fill=True, alpha=0.4, linewidth=2, clip=(0, None))

g.set_titles("{col_name}", fontweight='bold', fontsize=14)
g.set_axis_labels("Reconstruction Gap (Bits)", "Density", fontsize=12)
g.add_legend(title="Semantic Condition")

plt.subplots_adjust(top=0.85)
g.fig.suptitle("Ablation B: Semantic Specificity Test\n(Same Sparsity, Random Meaning)", 
               fontsize=16, fontweight='bold')

save_path = "ablation_shuffled_histogram.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"✅ Plot saved as {save_path}")