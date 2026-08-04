# Multi-Agent Reinforcement Learning for Warehouse Optimization

Scalable MARL system for optimizing order fulfillment, robot coordination, and inventory management in dynamic warehouses.

## Overview

Modern warehouses require intelligent coordination between multiple autonomous robots for:

- Order picking and batching optimization
- Collision free navigation
- Dynamic inventory restocking
- Scalability to 50+ heterogeneous agents

## Tech Stack

- RLlib (Ray) for multi-agent training
- Gymnasium custom warehouse environment
- PyTorch for policy networks
- PPO and QMIX algorithms
- GCP Vertex AI for distributed training
- MLflow for experiment tracking

<!-- CONFIRM: train_iac.py currently implements a custom Independent Actor-Critic (IAC) training loop in raw PyTorch, not RLlib, and not PPO or QMIX. Let me know if RLlib/PPO/QMIX are used elsewhere in the project (not visible from the files I could access) or if the tech stack section should be updated to reflect the current IAC implementation. -->

## Current Progress

- Custom Gymnasium warehouse environment (complete)
- Single-agent PPO baseline (complete)
- Multi-agent RLlib configuration (in progress)
- GCP Vertex AI pipeline (planned)

<!-- CONFIRM: eval_baseline.py evaluates a ManagerAgent plus WorkerPolicy setup (manager/worker architecture), and train_iac.py trains 4 independent actor-critic agents directly, separate from what's described above. Let me know if "Current Progress" should be updated to mention these two, or if they represent earlier work superseded by the RLlib direction. -->

## Repository Structure

```
marl-warehouse-optimizer/
├── src/
│   ├── env/
│   │   └── warehouse_env.py     # Custom Gymnasium warehouse environment
│   └── agents/
│       └── manager_worker.py    # ManagerAgent and WorkerPolicy classes
├── logs/                        # Training run logs (e.g. iac.csv)
├── train_iac.py                 # Independent Actor-Critic training script
├── eval_baseline.py             # Manager/worker baseline evaluation script
└── README.md
```

<!-- CONFIRM: this only reflects the files I was able to access (train_iac.py, eval_baseline.py, and their imports). If there are more files under src/ (for example an RLlib training script), paste them or the full file list and I will add them. -->
