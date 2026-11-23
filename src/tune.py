"""Tune learning rate and batch size without evaluating the held-out test set."""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import optuna
import torch
import yaml

from src.train import initializer, train_model
from src.train import device
from src.utils.enums.model_name_enums import Models
from src.utils.load_config import load_config
from src.utils.load_data import close_loaders, load_data


def tune_hyperparameters(
    model_name: str,
    output_dir: Path,
    *,
    config_path: Path | None = None,
    embedding_choice: str | None = None,
    loss_choice: str | None = None,
    n_trials: int = 20,
    learning_rate_min: float = 1e-5,
    learning_rate_max: float = 1e-2,
    batch_sizes: tuple[int, ...] = (16, 32, 64),
    study_name: str = "hyperparameters",
) -> optuna.Study:
    """Run additional fixed-budget trials, resuming only an identical setup."""
    if model_name not in (Models.CNN_EMBED, Models.CNN_NOEMBED):
        raise ValueError(f"Unsupported model: {model_name!r}")
    if model_name == Models.CNN_EMBED and (
        embedding_choice not in ("1", "2") or loss_choice not in ("1", "2")
    ):
        raise ValueError("Embedding tuning requires embedding and loss choices of '1' or '2'.")
    if n_trials < 1 or not batch_sizes or any(size < 1 for size in batch_sizes):
        raise ValueError("Trials and batch sizes must be positive.")
    if not (0 < learning_rate_min <= learning_rate_max < math.inf):
        raise ValueError("Learning-rate bounds must be finite, positive, and ordered.")

    config = load_config(config_path or f"{model_name}_config")
    if config["training_args"]["epochs"] < 1:
        raise ValueError("training_args.epochs must be positive.")
    context = {
        "model": model_name,
        "embedding": embedding_choice,
        "loss": loss_choice,
        "config": config,
        "learning_rate_bounds": [learning_rate_min, learning_rate_max],
        "batch_sizes": list(batch_sizes),
        "objective": "final_validation_loss",
    }
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{(output_dir / 'study.db').as_posix()}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.get("seed", 541)),
        load_if_exists=True,
    )
    previous_context = study.user_attrs.get("context")
    if previous_context is not None and previous_context != context:
        raise ValueError("Study setup changed. Use a new output directory or study name.")
    if previous_context is None and study.trials:
        raise ValueError("Cannot resume a study without matching setup metadata.")
    study.set_user_attr("context", context)

    def objective(trial: optuna.Trial) -> float:
        learning_rate = trial.suggest_float(
            "learning_rate", learning_rate_min, learning_rate_max, log=True
        )
        batch_size = trial.suggest_categorical("batch_size", list(batch_sizes))
        criterion, model, trial_config, _, _ = initializer(
            model_name, embedding_choice, loss_choice, config_path
        )
        if trial_config != config:
            raise ValueError("Configuration changed during tuning; start a new study.")
        model = model.to(device)
        train_loader, test_loader, val_loader = load_data(
            trial_config["data"], batch_size, trial_config.get("seed", 541)
        )
        try:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            _, val_losses, val_accuracies = train_model(
                model, train_loader, val_loader, criterion, optimizer,
                trial_config["training_args"]["epochs"],
            )
        finally:
            close_loaders(train_loader, test_loader, val_loader)
        value = val_losses[-1]
        if not math.isfinite(value):
            raise ValueError(f"Trial {trial.number} produced a non-finite validation loss.")
        trial.set_user_attr("validation_accuracy", val_accuracies[-1])
        return value

    study.optimize(objective, n_trials=n_trials)
    best_config = deepcopy(config)
    best_config["hyperparameters"]["learning_rate"] = study.best_params["learning_rate"]
    best_config["training_args"]["batch_size"] = study.best_params["batch_size"]
    with (output_dir / "best_config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(best_config, file, sort_keys=False)
    with (output_dir / "best_trial.json").open("w", encoding="utf-8") as file:
        json.dump({
            "study_name": study.study_name,
            "trial": study.best_trial.number,
            "validation_loss": study.best_value,
            "validation_accuracy": study.best_trial.user_attrs["validation_accuracy"],
            "params": study.best_params,
        }, file, indent=2)
    study.trials_dataframe().to_csv(output_dir / "trials.csv", index=False)
    print(f"Best validation loss: {study.best_value:.6f}; parameters: {study.best_params}")
    print(f"Saved study, trial history, and best configuration to {output_dir}")
    return study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=[Models.CNN_EMBED.value, Models.CNN_NOEMBED.value])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--embedding", choices=["1", "2"])
    parser.add_argument("--loss", choices=["1", "2"])
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--lr-max", type=float, default=1e-2)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--study-name", default="hyperparameters")
    args = parser.parse_args()
    tune_hyperparameters(
        args.model, args.output_dir, config_path=args.config,
        embedding_choice=args.embedding, loss_choice=args.loss, n_trials=args.trials,
        learning_rate_min=args.lr_min, learning_rate_max=args.lr_max,
        batch_sizes=tuple(args.batch_sizes), study_name=args.study_name,
    )


if __name__ == "__main__":
    main()
