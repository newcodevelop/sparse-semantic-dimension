# sparse-semantic-dimension

# Sparse Semantic Generalization Bound (White Paper)

This repository contains a short white paper exploring a learning-theoretic explanation of non-vacuous LLM generalization via **sparse semantic features**, using **Sparse Autoencoders (SAEs)** as a proxy/compression model. We argue that LLM having extreme high parameter counts actually capture all their knowledge in a low dimensional sparse manifold, which is captured by sparse activation on SAEs (S) trained on LLM (M) activation.

Key ingredients in the manuscript:
- A **proxy decomposition** that compares the original predictor to a proxy predictor $S\circ M$ built from an SAE reconstruction.
- A bounded **language-modeling loss** based on **prediction-smoothed bits-per-token (BPD)**.
- A generaization bound in which the dominant complexity term scales like $k\log(em/k)$.


## Notes

- The theory sections may include several results labeled as “sketch”; they are meant as a starting point for refinement.

## License

This is work of Dibyanayan Bandyopadhyay and subsequently copyrighted by him. Can be used in Apache 2.0 license terms.
