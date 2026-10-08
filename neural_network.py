import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class FlexibleMLP(nn.Module):

  def __init__(
      self, input_dim, output_dim, hidden_layers, neurons, activation="relu"
  ):
    super().__init__()
    layers = []
    in_dim = input_dim

    # Selección de la función de activación
    act_fn = nn.GELU if activation.lower() == "gelu" else nn.ReLU

    for _ in range(hidden_layers):
      layers.append(nn.Linear(in_dim, neurons))
      layers.append(act_fn())
      in_dim = neurons

    layers.append(nn.Linear(in_dim, output_dim))
    self.network = nn.Sequential(*layers)

  def forward(self, x):
    return self.network(x)


def set_seed(seed):
  torch.manual_seed(seed)
  np.random.seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def load_data(data_path):
  if os.path.exists(data_path):
    data = np.load(data_path)
    if "X" in data and "y" in data:
      X, y = data["X"], data["y"]
    elif "inputs" in data and "targets" in data:
      X, y = data["inputs"], data["targets"]
    else:
      keys = list(data.keys())
      X, y = data[keys[0]], data[keys[1]]
  else:
    print(
        f"Notice: '{data_path}' not found. Generating synthetic dataset for"
        " test execution..."
    )
    X = np.random.randn(120000, 14).astype(np.float32)
    y = (
        np.sum(X**2, axis=1, keepdims=True)
        + np.random.randn(120000, 1) * 0.1
    ).astype(np.float32)

  if y.ndim == 1:
    y = y.reshape(-1, 1)
  return X.astype(np.float32), y.astype(np.float32)


def main():
  parser = argparse.ArgumentParser(
      description="Neural Network Training and Evaluation Script"
  )
  parser.add_argument(
      "--quick-test", action="store_true", help="Run a quick sanity check"
  )
  parser.add_argument(
      "--train-size", type=int, default=100000, help="Number of training samples"
  )
  parser.add_argument(
      "--learning-rate", type=float, default=0.001, help="Learning rate"
  )
  parser.add_argument(
      "--hidden-layers", type=int, default=3, help="Number of hidden layers"
  )
  parser.add_argument(
      "--neurons", type=int, default=256, help="Neurons per hidden layer"
  )
  parser.add_argument(
      "--activation", type=str, default="relu", choices=["relu", "gelu"]
  )
  parser.add_argument("--batch-size", type=int, default=10000, help="Batch size")
  parser.add_argument(
      "--weight-decay", type=float, default=0.0, help="L2 regularization"
  )
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  parser.add_argument(
      "--epochs", type=int, default=1000, help="Number of training epochs"
  )
  parser.add_argument(
      "--device", type=str, default="cuda", help="Device (cpu or cuda)"
  )
  parser.add_argument(
      "--data-path",
      type=str,
      default="swept_volume_data.npz",
      help="Path to data file",
  )
  parser.add_argument(
      "--eval-test-best",
      action="store_true",
      help="Evaluate best model checkpoint",
  )
  parser.add_argument(
      "--config-dir",
      type=str,
      default=None,
      help="Directory containing model config and checkpoint",
  )

  args = parser.parse_args()
  set_seed(args.seed)

  # Selección automática de dispositivo si no hay CUDA
  device = (
      args.device
      if torch.cuda.is_available() and args.device == "cuda"
      else "cpu"
  )

  if args.quick_test:
    print("=== Executing Quick-Test Mode ===")
    args.epochs = 5
    args.train_size = min(500, args.train_size)
    args.batch_size = 32
    out_dir = os.path.join("runs", "quick_test")
  else:
    out_dir = os.path.join(
        "runs",
        f"train{args.train_size}_act{args.activation}_h{args.hidden_layers}_n{args.neurons}_b{args.batch_size}_seed{args.seed}",
    )

  os.makedirs(out_dir, exist_ok=True)

  X_full, y_full = load_data(args.data_path)
  total_samples = len(X_full)

  # Split 80% Train/Val, 20% Held-out Test
  test_size = int(0.2 * total_samples)
  train_val_size = total_samples - test_size

  X_train_val, y_train_val = X_full[:train_val_size], y_full[:train_val_size]
  X_test, y_test = X_full[train_val_size:], y_full[train_val_size:]

  # Sub-split para Validación (10% de train_val)
  val_size = int(0.1 * train_val_size)
  X_val, y_val = X_train_val[:val_size], y_train_val[:val_size]
  X_train_base, y_train_base = X_train_val[val_size:], y_train_val[val_size:]

  if args.train_size < len(X_train_base):
    indices = np.random.choice(
        len(X_train_base), args.train_size, replace=False
    )
    X_train, y_train = X_train_base[indices], y_train_base[indices]
  else:
    X_train, y_train = X_train_base, y_train_base

  # Normalización rigurosa calculada solo con X_train
  mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
  X_train_norm = (X_train - mean) / std
  X_val_norm = (X_val - mean) / std
  X_test_norm = (X_test - mean) / std

  if args.eval_test_best:
    eval_dir = args.config_dir if args.config_dir else out_dir
    model_path = os.path.join(eval_dir, "best_model.pt")

    if not os.path.exists(model_path):
      raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model = FlexibleMLP(
        X_full.shape[1],
        y_full.shape[1],
        args.hidden_layers,
        args.neurons,
        args.activation,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
      inputs = torch.tensor(X_test_norm, dtype=torch.float32).to(device)
      targets = torch.tensor(y_test, dtype=torch.float32).to(device)
      preds = model(inputs)
      test_rmse = torch.sqrt(nn.MSELoss()(preds, targets)).item()

    print(f"Held-Out Test RMSE: {test_rmse:.6f}")
    return

  train_dataset = TensorDataset(
      torch.tensor(X_train_norm, dtype=torch.float32),
      torch.tensor(y_train, dtype=torch.float32),
  )
  train_loader = DataLoader(
      train_dataset, batch_size=args.batch_size, shuffle=True
  )

  val_inputs = torch.tensor(X_val_norm, dtype=torch.float32).to(device)
  val_targets = torch.tensor(y_val, dtype=torch.float32).to(device)

  model = FlexibleMLP(
      X_full.shape[1],
      y_full.shape[1],
      args.hidden_layers,
      args.neurons,
      args.activation,
  ).to(device)
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(
      model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
  )

  history = {"epoch": [], "train_rmse": [], "val_rmse": []}
  best_val_loss = float("inf")

  for epoch in range(1, args.epochs + 1):
    model.train()
    total_mse = 0.0
    total_count = 0

    for batch_x, batch_y in train_loader:
      batch_x, batch_y = batch_x.to(device), batch_y.to(device)
      optimizer.zero_grad()
      outputs = model(batch_x)
      loss = criterion(outputs, batch_y)
      loss.backward()
      optimizer.step()

      total_mse += loss.item() * len(batch_x)
      total_count += len(batch_x)

    epoch_train_rmse = np.sqrt(total_mse / total_count)

    # Evaluación en Validación
    model.eval()
    with torch.no_grad():
      val_preds = model(val_inputs)
      epoch_val_rmse = torch.sqrt(criterion(val_preds, val_targets)).item()

    history["epoch"].append(epoch)
    history["train_rmse"].append(epoch_train_rmse)
    history["val_rmse"].append(epoch_val_rmse)

    if epoch_val_rmse < best_val_loss:
      best_val_loss = epoch_val_rmse
      torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))

    if epoch % max(1, args.epochs // 10) == 0 or epoch == 1 or args.quick_test:
      print(
          f"Epoch [{epoch}/{args.epochs}] - Train RMSE: {epoch_train_rmse:.6f}"
          f" | Val RMSE: {epoch_val_rmse:.6f}"
      )

  with open(os.path.join(out_dir, "config.json"), "w") as f:
    json.dump(vars(args), f, indent=4)

  np.savez(
      os.path.join(out_dir, "history.npz"),
      epoch=np.array(history["epoch"]),
      train_rmse=np.array(history["train_rmse"]),
      val_rmse=np.array(history["val_rmse"]),
  )

  print(f"\nExecution finished! Results saved to '{out_dir}'.")


if __name__ == "__main__":
  main()
