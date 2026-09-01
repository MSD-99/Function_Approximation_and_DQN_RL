# Deep RL & Function Approximation: Linear TD(0) & DQN

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![RL](https://img.shields.io/badge/Reinforcement%20Learning-Deep%20RL%20%26%20Function%20Approximation-0052CC?style=for-the-badge)

This repository transitions from tabular Reinforcement Learning to continuous state spaces using Function Approximation. It covers both linear value prediction and non-linear Deep Q-Networks (DQN).

## 📌 Project Overview

**Notebook**: [`function_approx_and_dqn.ipynb`](function_approx_and_dqn.ipynb)

The project is divided into two major components:

### 1. Linear Function Approximation (Value Prediction)
- **Environment**: Random Walk.
- **Method**: State aggregation (binning) combined with Linear TD(0).
- **Experiments**: Comparing generalization performance using different resolutions of state aggregation (e.g., 3 bins vs. 9 bins).

### 2. Deep Q-Network (DQN)
- **Environment**: `CartPole-v1` (OpenAI Gym).
- **Method**: A minimal implementation of DQN using PyTorch, incorporating Experience Replay and a Target Network.
- **Experiments**: Evaluating agent convergence and performing ablation studies on the effect of architectural/hyperparameter choices.

---

## 📊 Visualizations & Convergence Results

The outputs demonstrate the learned value functions under linear approximation and the learning curves (reward per episode) of the DQN agent.

### Sample Outputs
<p align="center">
  <img src="assets/func_approx_output_1.png" width="90%" alt="Output 1" />
</p>
<p align="center">
  <img src="assets/func_approx_output_2.png" width="45%" alt="Output 2" />
  <img src="assets/func_approx_output_3.png" width="45%" alt="Output 3" />
</p>

---

## 🚀 Quickstart

```bash
git clone https://github.com/MSD-99/Function_Approximation_and_DQN_RL.git
cd Function_Approximation_and_DQN_RL
pip install -r requirements.txt
jupyter lab
```
