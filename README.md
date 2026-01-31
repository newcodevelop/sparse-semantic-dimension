# sparse-semantic-dimension

# Sparse Semantic Generalization Bound (White Paper)

This repository contains a short LaTeX white paper exploring a learning-theoretic explanation of LLM generalization via **sparse semantic features**, using **Sparse Autoencoders (SAEs)** as a proxy/compression model.

Key ingredients in the manuscript:
- A **proxy decomposition** that compares the original predictor to a proxy predictor $S\circ M$ built from an SAE reconstruction.
- A bounded **language-modeling loss** based on **prediction-smoothed bits-per-token (BPD)**.
- A sketch SRM-style bound in which the dominant complexity term scales like $k\log(em/k)$.


## Notes

- The theory sections may include several results labeled as “sketch”; they are meant as a starting point for refinement.

## License

This is work of Dibyanayan Bandyopadhyay and subsequently copyrighted by him. Can be used in Apache 2.0 license terms.
