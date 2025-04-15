## Problem Descriptions

### Q1 – Rain Probability Estimation

**Goal:**  
Given an array of daily rain probabilities `p` (length 365), compute the probability that it rains more than `n` days in a year.

- Uses Dynamic Programming for computation optimization.

---

### Q2 – Phoneme-to-Word Reconstruction

**Goal:**  
Given a sequence of phonemes, find all possible combinations of dictionary words that can produce this exact phoneme sequence.

- Uses backtracking.
- Uses memoization for faster computations.

---

### Q4 – Manual CTC Loss Implementation

**Goal:**  
Implement the CTC loss (as introduced by Graves et al., ICML 2006) with both forward and backward support using PyTorch.

- Supports only batch size of 1 (for simplicity).
- In this assignment, I assume that pytorch's `autograd` functionality can be used.
