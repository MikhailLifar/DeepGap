"""Train and evaluate an LSTM on multi-stock return sequences."""

from __future__ import annotations

import argparse
import glob
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


@dataclass
class Config:
    data_dir: str
    sequence_length: int = 60
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    num_base_stocks: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 50
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    model_dir: str = "models"
    checkpoint_dir: str = "checkpoints"
    model_name: str = "lstm_multi_stock_predictor"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    content_clip: float = 20.0
    seed: int = 42
    plots_dir: str = "plots"
    use_tqdm: bool = True
    verbose: bool = True
    monthly: bool = False
    news_csv: str | None = None


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_all_stock_data(
    data_dir: str, sequence_length: int, verbose: bool = True
) -> pd.DataFrame:
    """Load all CSVs and return combined DataFrame with returns."""
    all_data: List[pd.DataFrame] = []
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    missing_columns: List[str] = []
    too_short: List[Tuple[str, int]] = []
    load_errors: List[str] = []
    if verbose:
        print(f"Found {len(csv_files)} CSV files in {data_dir}")

    for file_path in csv_files:
        ticker = Path(file_path).stem
        try:
            df = pd.read_csv(file_path)
            if "Date" in df.columns and "Close" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.sort_values("Date")
                df["Close_Return"] = df["Close"].pct_change() * 100
                df["Ticker"] = ticker
                df = df[["Ticker", "Date", "Close", "Close_Return"]].copy()
                df = df.dropna().reset_index(drop=True)
                if len(df) > sequence_length:
                    all_data.append(df)
                else:
                    too_short.append((ticker, len(df)))
            else:
                missing_columns.append(ticker)
        except Exception as exc:
            if verbose:
                print(f"Error loading {ticker}: {exc}")
            load_errors.append(ticker)
            continue

    if not all_data:
        raise ValueError("No data loaded.")

    combined_df = pd.concat(all_data, ignore_index=True)
    if verbose:
        print(f"Total records: {len(combined_df)}")
        print(f"Unique tickers: {combined_df['Ticker'].nunique()}")
        print(f"Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
        if missing_columns:
            print(f"Skipped (missing Date/Close): {len(missing_columns)}")
        if too_short:
            print(f"Skipped (<= sequence_length): {len(too_short)}")
            preview = ", ".join(f"{t}:{n}" for t, n in sorted(too_short)[:20])
            print(f"Short tickers (sample): {preview}")
        if load_errors:
            print(f"Skipped (load errors): {len(load_errors)}")
    return combined_df


def load_all_stock_data_monthly(
    data_dir: str, sequence_length: int, verbose: bool = True
) -> pd.DataFrame:
    """Load all CSVs and return combined DataFrame with monthly returns."""
    all_data: List[pd.DataFrame] = []
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    missing_columns: List[str] = []
    too_short: List[Tuple[str, int]] = []
    load_errors: List[str] = []
    if verbose:
        print(f"Found {len(csv_files)} CSV files in {data_dir}")

    for file_path in csv_files:
        ticker = Path(file_path).stem
        try:
            df = pd.read_csv(file_path)
            if "Date" in df.columns and "Close" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.sort_values("Date")
                df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
                month_end = df.groupby("Month", as_index=False).tail(1)
                month_end = month_end[["Month", "Close"]].copy()
                month_end = month_end.sort_values("Month")
                month_end["Close_Return"] = month_end["Close"].pct_change() * 100
                month_end["Ticker"] = ticker
                month_end = month_end.rename(columns={"Month": "Date"})
                month_end = month_end[["Ticker", "Date", "Close", "Close_Return"]].copy()
                month_end = month_end.dropna().reset_index(drop=True)
                if len(month_end) > sequence_length:
                    all_data.append(month_end)
                else:
                    too_short.append((ticker, len(month_end)))
            else:
                missing_columns.append(ticker)
        except Exception as exc:
            if verbose:
                print(f"Error loading {ticker}: {exc}")
            load_errors.append(ticker)
            continue

    if not all_data:
        raise ValueError("No data loaded.")

    combined_df = pd.concat(all_data, ignore_index=True)
    if verbose:
        print(f"Total records: {len(combined_df)}")
        print(f"Unique tickers: {combined_df['Ticker'].nunique()}")
        print(f"Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
        if missing_columns:
            print(f"Skipped (missing Date/Close): {len(missing_columns)}")
        if too_short:
            print(f"Skipped (<= sequence_length): {len(too_short)}")
            preview = ", ".join(f"{t}:{n}" for t, n in sorted(too_short)[:20])
            print(f"Short tickers (sample): {preview}")
        if load_errors:
            print(f"Skipped (load errors): {len(load_errors)}")
    return combined_df


def load_news_monthly(news_csv: str, verbose: bool = True) -> pd.DataFrame:
    """Load monthly news CSV and aggregate to monthly features."""
    df = pd.read_csv(news_csv)
    if "month_start" not in df.columns:
        raise ValueError("news CSV must include month_start column")
    df["month_start"] = pd.to_datetime(df["month_start"])
    df["title"] = df.get("title", "").fillna("")
    df["text"] = df.get("text", "").fillna("")

    df["title_len"] = df["title"].str.len()
    df["text_len"] = df["text"].str.len()
    df["has_text"] = (df["text"].str.len() > 0).astype(int)

    agg = df.groupby("month_start").agg(
        news_count=("title", "size"),
        avg_title_len=("title_len", "mean"),
        avg_text_len=("text_len", "mean"),
        text_count=("has_text", "sum"),
    )
    agg["news_mask"] = (agg["news_count"] > 0).astype(int)
    if verbose:
        print(f"News months: {len(agg)} (from {agg.index.min()} to {agg.index.max()})")
    return agg


def prepare_multi_stock_sequences(
    df: pd.DataFrame,
    base_stocks: List[str],
    sequence_length: int,
    news_df: pd.DataFrame | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Create sequences for LSTM training."""
    all_stocks = df["Ticker"].unique().tolist()
    pivot_df = df.pivot_table(index="Date", columns="Ticker", values="Close_Return")

    sequences_x: List[np.ndarray] = []
    sequences_y: List[float] = []
    feature_names = base_stocks.copy()
    if "TARGET" not in feature_names:
        feature_names.append("TARGET")
    if news_df is not None:
        feature_names.extend(news_df.columns.tolist())

    stock_iter = all_stocks
    if tqdm is not None:
        stock_iter = tqdm(all_stocks, desc="Build sequences", leave=True)
    for target_stock in stock_iter:
        if target_stock not in pivot_df.columns:
            continue
        base_present = [b for b in base_stocks if b in pivot_df.columns]
        cols = list(dict.fromkeys([target_stock] + base_present))
        pivot_sub = pivot_df[cols].fillna(method="ffill").dropna()
        dates = pivot_sub.index.tolist()
        if len(dates) <= sequence_length + 1:
            continue

        news_values = None
        if news_df is not None:
            news_aligned = news_df.reindex(dates).fillna(0.0)
            news_values = news_aligned.to_numpy(dtype=np.float32)

        for i in range(sequence_length, len(dates) - 1):
            base_stock_returns = []
            for base_stock in base_stocks:
                if base_stock in pivot_df.columns:
                    series = pivot_df[base_stock].reindex(dates).fillna(method="ffill")
                    base_stock_returns.append(
                        series.iloc[i - sequence_length : i].values
                    )
                else:
                    base_stock_returns.append(np.zeros(sequence_length))

            target_series = pivot_df[target_stock].reindex(dates).fillna(method="ffill")
            target_returns = target_series.iloc[i - sequence_length : i].values
            base_matrix = np.column_stack(base_stock_returns + [target_returns])
            if news_values is not None:
                news_window = news_values[i - sequence_length : i]
                feature_matrix = np.concatenate([base_matrix, news_window], axis=1)
            else:
                feature_matrix = base_matrix
            next_day_return = target_series.iloc[i + 1]
            sequences_x.append(feature_matrix)
            sequences_y.append(next_day_return)

    x = np.array(sequences_x, dtype=np.float32)
    y = np.array(sequences_y, dtype=np.float32)
    print(f"Created {len(sequences_x)} sequences")
    print(f"X shape: {x.shape}, y shape: {y.shape}")
    return x, y, feature_names


def train_val_split(
    x: np.ndarray,
    y: np.ndarray,
    val_split: float,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = len(x)
    n_val = int(n_samples * val_split)
    if shuffle:
        indices = np.random.permutation(n_samples)
        x = x[indices]
        y = y[indices]
    x_train, x_val = x[:-n_val], x[-n_val:]
    y_train, y_val = y[:-n_val], y[-n_val:]
    print(f"Train: {len(x_train)} samples, Val: {len(x_val)} samples")
    return x_train, x_val, y_train, y_val


class StockSequenceDataset(Dataset):
    """Dataset for stock sequences."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class LSTMModel(nn.Module):
    """LSTM model for time series forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze(-1)


class Trainer:
    """Trainer with early stopping and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        model_name: str,
        checkpoint_dir: str,
        early_stopping_patience: int,
        use_tqdm: bool = True,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model_name = model_name
        self.early_stopping_patience = early_stopping_patience
        self.use_tqdm = use_tqdm

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_model_state: Dict[str, torch.Tensor] | None = None

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(
            self.checkpoint_dir, f"{self.model_name}_best.pt"
        )

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        loader = self.train_loader
        if tqdm is not None and self.use_tqdm:
            loader = tqdm(self.train_loader, desc="Train", leave=False)
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(1, n_batches)

    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        loader = self.val_loader
        if tqdm is not None and self.use_tqdm:
            loader = tqdm(self.val_loader, desc="Val", leave=False)
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(1, n_batches)

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
        }
        torch.save(
            checkpoint, os.path.join(self.checkpoint_dir, f"{self.model_name}_latest.pt")
        )
        torch.save(checkpoint, self.checkpoint_path)

    def load_checkpoint(self) -> int:
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.train_losses = checkpoint.get("train_losses", [])
            self.val_losses = checkpoint.get("val_losses", [])
            self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            return int(checkpoint["epoch"]) + 1
        return 0

    def train(self, num_epochs: int) -> Dict[str, List[float] | float]:
        start_epoch = self.load_checkpoint()
        print("Starting training...")
        epoch_iter = range(start_epoch, num_epochs)
        pbar = None
        if tqdm is not None and self.use_tqdm:
            pbar = tqdm(epoch_iter, desc="Epochs", leave=True)
            epoch_iter = pbar
        for epoch in epoch_iter:
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            is_best = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
                is_best = True
            else:
                self.patience_counter += 1

            if is_best:
                self.save_checkpoint(epoch, val_loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train: {train_loss:.6f}, Val: {val_loss:.6f}"
                )

            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                if pbar is not None:
                    pbar.close()
                break
        if pbar is not None:
            pbar.close()

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
        }


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def symmetric_mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100


def evaluate_model(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> Dict[str, np.ndarray | float]:
    model.eval()
    all_predictions: List[float] = []
    all_targets: List[float] = []
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            predictions = model(batch_x)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    mape = mean_absolute_percentage_error(targets, predictions)
    smape = symmetric_mean_absolute_percentage_error(targets, predictions)
    pred_sign = np.sign(predictions)
    target_sign = np.sign(targets)
    directional_accuracy = (pred_sign == target_sign).mean() * 100

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "smape": smape,
        "directional_accuracy": directional_accuracy,
        "predictions": predictions,
        "targets": targets,
    }


def save_plots(
    history: Dict[str, List[float] | float],
    train_metrics: Dict[str, np.ndarray | float],
    val_metrics: Dict[str, np.ndarray | float],
    plots_dir: str,
) -> None:
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_losses"], label="Train Loss", linewidth=2)
    plt.plot(history["val_losses"], label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_losses"], label="Train Loss", linewidth=2)
    plt.plot(history["val_losses"], label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE) - Log Scale")
    plt.title("Training and Validation Loss (Log Scale)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "loss_curves.png"), dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].scatter(train_metrics["targets"], train_metrics["predictions"], alpha=0.5, s=10)
    axes[0].plot(
        [train_metrics["targets"].min(), train_metrics["targets"].max()],
        [train_metrics["targets"].min(), train_metrics["targets"].max()],
        "r--",
        linewidth=2,
        label="Perfect Prediction",
    )
    axes[0].set_xlabel("Actual Close Return")
    axes[0].set_ylabel("Predicted Close Return")
    axes[0].set_title(f"Train Set (RMSE: {train_metrics['rmse']:.4f})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(
        val_metrics["targets"],
        val_metrics["predictions"],
        alpha=0.5,
        s=10,
        color="orange",
    )
    axes[1].plot(
        [val_metrics["targets"].min(), val_metrics["targets"].max()],
        [val_metrics["targets"].min(), val_metrics["targets"].max()],
        "r--",
        linewidth=2,
        label="Perfect Prediction",
    )
    axes[1].set_xlabel("Actual Close Return")
    axes[1].set_ylabel("Predicted Close Return")
    axes[1].set_title(f"Validation Set (RMSE: {val_metrics['rmse']:.4f})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "pred_vs_actual.png"), dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    train_errors = train_metrics["predictions"] - train_metrics["targets"]
    val_errors = val_metrics["predictions"] - val_metrics["targets"]

    axes[0].hist(train_errors, bins=50, alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Prediction Error")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Train Set Error (Mean: {train_errors.mean():.4f})")
    axes[0].axvline(0, color="r", linestyle="--", linewidth=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(val_errors, bins=50, alpha=0.7, edgecolor="black", color="orange")
    axes[1].set_xlabel("Prediction Error")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Validation Error (Mean: {val_errors.mean():.4f})")
    axes[1].axvline(0, color="r", linestyle="--", linewidth=2)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "error_distributions.png"), dpi=150)
    plt.close()


def analyze_feature_importance(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    feature_names: List[str],
) -> pd.DataFrame:
    model.eval()
    batch_x, batch_y = next(iter(data_loader))
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    with torch.no_grad():
        baseline_preds = model(batch_x)
        baseline_loss = nn.MSELoss()(baseline_preds, batch_y).item()

    importance_scores: List[float] = []
    for feature_idx in range(len(feature_names)):
        perturbed_x = batch_x.clone()
        perm = torch.randperm(perturbed_x.size(0))
        perturbed_x[:, :, feature_idx] = perturbed_x[:, :, feature_idx][perm]
        with torch.no_grad():
            perturbed_preds = model(perturbed_x)
            perturbed_loss = nn.MSELoss()(perturbed_preds, batch_y).item()
        importance_scores.append(perturbed_loss - baseline_loss)

    scores = np.array(importance_scores)
    scores = scores / np.sum(scores) if np.sum(scores) != 0 else scores
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": scores}
    ).sort_values("Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df)
    plt.title("Feature Importance")
    plt.tight_layout()
    return importance_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an LSTM on multi-stock return sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data/tqbr")
    parser.add_argument("--news-csv", default=None)
    parser.add_argument("--monthly", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--num-base-stocks", type=int, default=10)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--early-stopping", type=int, default=10)
    parser.add_argument("--plots-dir", default="plots")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--model-name", default="lstm_multi_stock_predictor")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-feature-importance", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--metrics-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        data_dir=args.data_dir,
        news_csv=args.news_csv,
        monthly=args.monthly,
        sequence_length=args.sequence_length,
        num_base_stocks=args.num_base_stocks,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        validation_split=args.validation_split,
        early_stopping_patience=args.early_stopping,
        plots_dir=args.plots_dir,
        model_dir=args.model_dir,
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
        use_tqdm=not args.no_progress and not args.metrics_only,
        verbose=not args.metrics_only,
    )

    seed_all(config.seed)
    if config.verbose:
        print(f"Using device: {config.device}")

    if config.monthly:
        df_all = load_all_stock_data_monthly(
            config.data_dir, config.sequence_length, verbose=config.verbose
        )
    else:
        df_all = load_all_stock_data(
            config.data_dir, config.sequence_length, verbose=config.verbose
        )
    df_all["Close_Return"] = np.clip(
        df_all["Close_Return"].to_numpy(), -config.content_clip, config.content_clip
    )

    base_stocks = [
        "SBER",
        "VTBR",
        "GAZP",
        "LKOH",
        "MGNT",
        "X5",
        "YNDX",
        "MTSS",
        "NLMK",
        "PLZL",
    ]
    if config.num_base_stocks > len(base_stocks):
        extras = [
            t for t in df_all["Ticker"].value_counts().index.tolist()
            if t not in base_stocks
        ]
        base_stocks = (base_stocks + extras)[: config.num_base_stocks]
    else:
        base_stocks = base_stocks[: config.num_base_stocks]

    news_df = None
    if config.news_csv:
        news_df = load_news_monthly(config.news_csv, verbose=config.verbose)

    x, y, feature_names = prepare_multi_stock_sequences(
        df_all, base_stocks, config.sequence_length, news_df=news_df
    )
    x_train, x_val, y_train, y_val = train_val_split(
        x, y, config.validation_split, shuffle=True
    )

    train_dataset = StockSequenceDataset(x_train, y_train)
    val_dataset = StockSequenceDataset(x_val, y_val)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = LSTMModel(
        input_size=x_train.shape[2],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
    ).to(config.device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device(config.device),
        model_name=config.model_name,
        checkpoint_dir=config.checkpoint_dir,
        early_stopping_patience=config.early_stopping_patience,
        use_tqdm=config.use_tqdm,
    )

    history = trainer.train(config.num_epochs)
    train_metrics = evaluate_model(model, train_loader, torch.device(config.device))
    val_metrics = evaluate_model(model, val_loader, torch.device(config.device))

    if args.metrics_only:
        print(
            ",".join(
                [
                    str(config.num_base_stocks),
                    str(config.sequence_length),
                    str(config.num_layers),
                    f"{train_metrics['mse']:.6f}",
                    f"{train_metrics['rmse']:.6f}",
                    f"{train_metrics['mae']:.6f}",
                    f"{train_metrics['mape']:.2f}",
                    f"{train_metrics['smape']:.2f}",
                    f"{train_metrics['directional_accuracy']:.2f}",
                    f"{val_metrics['mse']:.6f}",
                    f"{val_metrics['rmse']:.6f}",
                    f"{val_metrics['mae']:.6f}",
                    f"{val_metrics['mape']:.2f}",
                    f"{val_metrics['smape']:.2f}",
                    f"{val_metrics['directional_accuracy']:.2f}",
                ]
            )
        )
    else:
        print("\nTRAIN METRICS")
        print(f"MSE:  {train_metrics['mse']:.6f}")
        print(f"RMSE: {train_metrics['rmse']:.6f}")
        print(f"MAE:  {train_metrics['mae']:.6f}")
        print(f"MAPE: {train_metrics['mape']:.2f}%")
        print(f"SMAPE: {train_metrics['smape']:.2f}%")
        print(f"Directional Accuracy: {train_metrics['directional_accuracy']:.2f}%")

        print("\nVAL METRICS")
        print(f"MSE:  {val_metrics['mse']:.6f}")
        print(f"RMSE: {val_metrics['rmse']:.6f}")
        print(f"MAE:  {val_metrics['mae']:.6f}")
        print(f"MAPE: {val_metrics['mape']:.2f}%")
        print(f"SMAPE: {val_metrics['smape']:.2f}%")
        print(f"Directional Accuracy: {val_metrics['directional_accuracy']:.2f}%")

    if not args.no_plots and not args.metrics_only:
        save_plots(history, train_metrics, val_metrics, config.plots_dir)

    if not args.no_feature_importance and not args.metrics_only:
        importance_df = analyze_feature_importance(
            model, val_loader, torch.device(config.device), feature_names
        )
        importance_path = os.path.join(config.plots_dir, "feature_importance.csv")
        os.makedirs(config.plots_dir, exist_ok=True)
        importance_df.to_csv(importance_path, index=False)

    os.makedirs(config.model_dir, exist_ok=True)
    model_path = os.path.join(config.model_dir, f"{config.model_name}.pth")
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()
