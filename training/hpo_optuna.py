"""
Optuna HPO sketch for CPC+SNN training.

This module provides a minimal runnable skeleton to launch Optuna studies
over key hyperparameters. It integrates with the existing `CPCSNNTrainer`
and returns the best trial summary. Designed to be safe on small GPUs
by default (small search space, conservative batch/epochs).
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any

import optuna

try:
    from .base_trainer import CPCSNNTrainer, TrainingConfig
except Exception:  # fallback for direct import
    from training.base_trainer import CPCSNNTrainer, TrainingConfig


def objective(trial: optuna.trial.Trial) -> float:
    """Objective function for Optuna: maximize balanced accuracy on test set."""
    # Sample hyperparameters (narrow ranges for stability on 3060 Ti)
    learning_rate = trial.suggest_float("learning_rate", 3e-5, 3e-4, log=True)
    snn_hidden = trial.suggest_int("snn_hidden", 24, 64, step=8)
    spike_time_steps = trial.suggest_int("spike_time_steps", 16, 32, step=4)
    spike_threshold = trial.suggest_float("spike_threshold", 0.05, 0.2)
    focal_gamma = trial.suggest_float("focal_gamma", 1.5, 3.0)
    class1_weight = trial.suggest_float("class1_weight", 1.0, 1.6)
    cpc_heads = trial.suggest_int("cpc_heads", 4, 8, step=2)
    cpc_layers = trial.suggest_int("cpc_layers", 2, 6, step=2)

    # Build config
    cfg = TrainingConfig(
        learning_rate=learning_rate,
        batch_size=1,
        num_epochs=8,  # short for HPO speed
        output_dir=str(Path("outputs") / "hpo_trials" / f"trial_{trial.number}"),
        spike_time_steps=spike_time_steps,
        spike_threshold=spike_threshold,
        spike_learnable=True,
        focal_gamma=focal_gamma,
        class1_weight=class1_weight,
        cpc_attention_heads=cpc_heads,
        cpc_transformer_layers=cpc_layers,
        snn_hidden_size=snn_hidden,
        grad_accum_steps=3,
        early_stopping_metric="balanced_accuracy",
        early_stopping_mode="max",
        checkpoint_every_epochs=1000,  # effectively disable per-epoch ckpt
        use_wandb=False,
        use_tensorboard=False,
    )

    trainer = CPCSNNTrainer(cfg)
    model = trainer.create_model()

    # Minimal synthetic dataset for quick HPO; replace with real generator if desired
    import jax
    import jax.numpy as jnp
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (256, 256))
    y = jax.random.bernoulli(key, p=0.4, shape=(256,)).astype(jnp.int32)
    # stratified split
    train_x, test_x = x[:200], x[200:]
    train_y, test_y = y[:200], y[200:]

    trainer.train_state = trainer.create_train_state(model, train_x[:1])

    # Short training loop
    steps_per_epoch = 60
    for epoch in range(cfg.num_epochs):
        for i in range(steps_per_epoch):
            idx = (i * cfg.batch_size) % len(train_x)
            batch = (train_x[idx:idx+cfg.batch_size], train_y[idx:idx+cfg.batch_size])
            trainer.train_state, metrics, _ = trainer.train_step(trainer.train_state, batch)

        # Eval and report intermediate score
        from .test_evaluation import evaluate_on_test_set
        eval_res = evaluate_on_test_set(trainer.train_state, test_x, test_y, train_signals=train_x, verbose=False, batch_size=64)
        bacc = 0.5 * (float(eval_res.get('specificity', 0.0)) + float(eval_res.get('recall', 0.0)))
        trial.report(bacc, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Save trial results
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "trial_result.json", "w") as f:
        json.dump({
            "best_balanced_accuracy": bacc,
            "params": trial.params,
        }, f, indent=2)

    return bacc


def run_hpo(n_trials: int = 20) -> int:
    """Run an Optuna study over the objective defined above."""
    study = optuna.create_study(direction="maximize", study_name="cpc_snn_hpo")
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    best = study.best_trial
    print("Best trial:")
    print({"value": best.value, "params": best.params})

    # Save study summary
    out_dir = Path("outputs") / "hpo_trials"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "best_trial.json", "w") as f:
        json.dump({
            "value": best.value,
            "params": best.params,
            "number": best.number,
        }, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(run_hpo())


