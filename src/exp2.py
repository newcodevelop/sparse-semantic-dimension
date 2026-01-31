import torch
import numpy as np
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE
from tqdm import tqdm
import matplotlib.pyplot as plt # For histogram data

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "N_TOKENS": 10_000,          # 10k tokens per dataset is enough for distribution
    "SAE_RELEASE": "gpt2-small-res-jb",
    "SAE_ID": "blocks.6.hook_resid_pre",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "CONTEXT_LEN": 128
}

print(f"Device: {CONFIG['DEVICE']}")

# ==========================================
# 2. LOAD MODEL & SAE
# ==========================================
model = HookedTransformer.from_pretrained("gpt2-small", device=CONFIG["DEVICE"])
sae, _, _ = SAE.from_pretrained(release=CONFIG["SAE_RELEASE"], sae_id=CONFIG["SAE_ID"], device=CONFIG["DEVICE"])
sae.eval()

# ==========================================
# 3. DATASET PREPARATION
# ==========================================
def get_tokens_from_dataset(dataset_name, split, config_name=None):
    """Generates a stream of tokens from a huggingface dataset"""
    print(f"Loading {dataset_name}...")
    ds = load_dataset(dataset_name, config_name, split=split, streaming=True)
    iterator = iter(ds)
    
    collected_tokens = []
    count = 0
    
    pbar = tqdm(total=CONFIG["N_TOKENS"])
    
    while count < CONFIG["N_TOKENS"]:
        try:
            item = next(iterator)
            
            try:
                text = item['text'] if 'text' in item else item['code']
            except:
                print(item)

            # Tokenize
            batch_tokens = model.to_tokens(text)[:, :CONFIG["CONTEXT_LEN"]]
            if batch_tokens.shape[1] < CONFIG["CONTEXT_LEN"]: continue
            
            collected_tokens.append(batch_tokens)
            count += batch_tokens.numel()
            pbar.update(batch_tokens.numel())
            
        except StopIteration:
            break
            
    pbar.close()
    return torch.cat(collected_tokens, dim=0)[:CONFIG["N_TOKENS"] // CONFIG["CONTEXT_LEN"] + 1]

# 1. In-Distribution: C4 (Web Text)
tokens_id = get_tokens_from_dataset("NeelNanda/c4-code-20k", "train")

# 2. Shifted: CodeParrot (Python Code)
tokens_shifted = get_tokens_from_dataset("codeparrot/github-code", "train")

# 3. Far-OOD: Random Tokens
# Generate random integers within vocab range
print("Generating Random Tokens...")
tokens_ood = torch.randint(
    0, model.cfg.d_vocab, 
    (CONFIG["N_TOKENS"] // CONFIG["CONTEXT_LEN"] + 5, CONFIG["CONTEXT_LEN"])
).to(CONFIG["DEVICE"])

datasets = {
    "In-Distribution (English)": tokens_id,
    "Shifted (Code)": tokens_shifted,
    "Far-OOD (Random)": tokens_ood
}

# ==========================================
# 4. RUN EXPERIMENT: MEASURE SPARSITY (k)
# ==========================================
results = {}

for name, tokens in datasets.items():
    print(f"\nProcessing {name}...")
    sparsity_scores = []
    
    # Process in batches
    batch_size = 8
    num_batches = len(tokens) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            batch = tokens[i*batch_size : (i+1)*batch_size].to(CONFIG["DEVICE"])
            
            # Run Model Cache
            _, cache = model.run_with_cache(batch, names_filter=[CONFIG["SAE_ID"]])
            activations = cache[CONFIG["SAE_ID"]] # [Batch, Seq, d_model]
            
            # Flatten
            flat_act = activations.reshape(-1, activations.shape[-1])
            
            # SAE Encode
            feature_acts = sae.encode(flat_act)
            
            # Calculate k (L0 norm) per token
            k_per_token = (feature_acts > 0).float().sum(dim=-1).cpu().numpy()
            sparsity_scores.extend(k_per_token)
            
    results[name] = np.array(sparsity_scores)

# ==========================================
# 5. PRINT STATISTICS & HISTOGRAM DATA
# ==========================================
print("\n" + "="*40)
print("FINAL RESULTS: OOD ORACLE CHECK")
print("="*40)

for name, k_vals in results.items():
    mean_k = np.mean(k_vals)
    std_k = np.std(k_vals)
    max_k = np.max(k_vals)
    
    print(f"DATASET: {name}")
    print(f"  Mean Sparsity (k): {mean_k:.2f}")
    print(f"  Std Dev:           {std_k:.2f}")
    print(f"  Max k Observed:    {max_k}")
    print("-" * 20)

print("\n(Copy these numbers for your 'Experiment 2' table)")


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
    "N_TOKENS": 20_000, 
    "BATCH_SIZE": 8,
    "CONTEXT_LEN": 128,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    
    "MODELS": {
        "GPT-2 Small": {
            "name": "gpt2-small",
            "sae_release": "gpt2-small-res-jb", 
            "sae_id": "blocks.6.hook_resid_pre",
            "batch_size": 16
        },
        "Gemma-2B": {
            "name": "gemma-2b",
            "sae_release": "gemma-2b-res-jb", 
            "sae_id": "blocks.12.hook_resid_post", 
            "batch_size": 4
        }
    }
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_dataset_tokens(model, dataset_name, split, n_tokens):
    """Streams tokens from a HuggingFace dataset."""
    ds = load_dataset(dataset_name, split=split, streaming=True)
    iterator = iter(ds)
    collected = []
    count = 0
    pbar = tqdm(total=n_tokens, desc=f"  Loading {dataset_name}", leave=False)
    
    while count < n_tokens:
        try:
            item = next(iterator)
            text = item['text'] if 'text' in item else item['code']
            tokens = model.to_tokens(text)[:, :CONFIG["CONTEXT_LEN"]]
            if tokens.shape[1] < CONFIG["CONTEXT_LEN"]: continue
            collected.append(tokens)
            count += tokens.numel()
            pbar.update(tokens.numel())
        except StopIteration:
            break
    pbar.close()
    full_tensor = torch.cat(collected, dim=0)
    return full_tensor[:(n_tokens // CONFIG["CONTEXT_LEN"]) + 1]

def generate_random_tokens(model, n_tokens):
    """Generates random noise tokens."""
    vocab_size = model.cfg.d_vocab
    n_batches = (n_tokens // CONFIG["CONTEXT_LEN"]) + 1
    return torch.randint(0, vocab_size, (n_batches, CONFIG["CONTEXT_LEN"]))

# ==========================================
# 3. EXPERIMENT LOOP
# ==========================================
all_results = []
stats_report = {}

print("Starting Multi-Model OOD Oracle Check...")

for model_name, cfg in CONFIG["MODELS"].items():
    print(f"\n🚀 MODEL: {model_name}")
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        model = HookedTransformer.from_pretrained(cfg["name"], device=CONFIG["DEVICE"])
        sae, _, _ = SAE.from_pretrained(release=cfg["sae_release"], sae_id=cfg["sae_id"], device=CONFIG["DEVICE"])
        sae.eval()
        
        # Prepare Data
        tokens_id = get_dataset_tokens(model, "NeelNanda/c4-code-20k", "train", CONFIG["N_TOKENS"])
        tokens_shifted = get_dataset_tokens(model, "codeparrot/github-code", "train", CONFIG["N_TOKENS"])
        tokens_ood = generate_random_tokens(model, CONFIG["N_TOKENS"])
        
        datasets = {
            "In-Distribution (English)": tokens_id,
            "Shifted (Code)": tokens_shifted,
            "Far-OOD (Random)": tokens_ood
        }
        
        model_stats = []
        
        for data_type, tokens in datasets.items():
            sparsity_list = []
            batch_size = cfg["batch_size"]
            num_batches = len(tokens) // batch_size
            
            with torch.no_grad():
                for i in tqdm(range(num_batches), desc=f"  Scanning {data_type}", leave=False):
                    batch = tokens[i*batch_size : (i+1)*batch_size].to(CONFIG["DEVICE"])
                    _, cache = model.run_with_cache(batch, names_filter=[cfg["sae_id"]])
                    acts = cache[cfg["sae_id"]]
                    flat_acts = acts.reshape(-1, acts.shape[-1])
                    
                    feature_acts = sae.encode(flat_acts)
                    k = (feature_acts > 0).float().sum(dim=-1).cpu().numpy()
                    sparsity_list.extend(k)
            
            # --- STATISTICS CALCULATION ---
            k_arr = np.array(sparsity_list)
            stats = {
                "name": data_type,
                "mean": np.mean(k_arr),
                "std": np.std(k_arr),
                "max": np.max(k_arr)
            }
            model_stats.append(stats)
            
            # Store for Plotting (Downsample for speed if needed)
            if len(sparsity_list) > 10000:
                sparsity_list = np.random.choice(sparsity_list, 10000, replace=False)
            
            for val in sparsity_list:
                all_results.append({
                    "Model": model_name,
                    "Condition": data_type,
                    "Sparsity (k)": val
                })

        stats_report[model_name] = model_stats

    except Exception as e:
        print(f"❌ Error on {model_name}: {e}")

# ==========================================
# 4. PRINT REPORT (YOUR FORMAT)
# ==========================================
print("\n" + "="*40)
print("FINAL STATISTICS REPORT")
print("="*40)

for model_name, stats_list in stats_report.items():
    print(f"\n>>> MODEL: {model_name}")
    print("="*40)
    for stat in stats_list:
        print(f"DATASET: {stat['name']}")
        print(f"  Mean Sparsity (k): {stat['mean']:.2f}")
        print(f"  Std Dev:           {stat['std']:.2f}")
        print(f"  Max k Observed:    {stat['max']:.1f}")
        print("-" * 20)

# ==========================================
# 5. GENERATE PLOT
# ==========================================
print("\nGenerating Plots...")
df = pd.DataFrame(all_results)

sns.set_theme(style="white", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'

g = sns.FacetGrid(df, col="Model", hue="Condition", height=5, aspect=1.2, 
                  sharex=False, 
                  palette={"In-Distribution (English)": "#1f77b4", 
                           "Shifted (Code)": "#2ca02c", 
                           "Far-OOD (Random)": "#d62728"})

g.map(sns.kdeplot, "Sparsity (k)", fill=True, alpha=0.3, linewidth=2)

# --- INSERT HERE ---
g.set(xlim=(0, 300)) 
# -------------------

g.set_titles("{col_name}", fontweight='bold', fontsize=14)
g.set_axis_labels("Active Features (k)", "Density", fontsize=12)
g.add_legend(title="Data Distribution")

plt.subplots_adjust(top=0.85)
g.fig.suptitle("The Complexity Shift: Runtime Sparsity as an Uncertainty Metric", fontsize=16, fontweight='bold')

plt.savefig("sparsity_shift_histogram.png", dpi=300, bbox_inches="tight")
print("✅ Plot saved as sparsity_shift_histogram.png")