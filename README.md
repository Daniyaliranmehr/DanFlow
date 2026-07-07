<p align="center">
  <img src="assets/logo.png" alt="DanFlow Logo" width="220">
</p>
<h1 align="center">DanFlow</h1>

DanFlow is a modular Python library for deep learning, data science, and visualization.

Built on top of PyTorch, it provides reusable components that simplify common workflows while remaining easy to use and extend.

## Features

### Training
- Train PyTorch models with a simple interface
- Support for custom loss functions
- Support for custom optimizers
- Support for custom evaluation metrics
- Automatic checkpoint saving
- Training and validation history tracking
- Real-time training progress with tqdm

### Evaluation

- Evaluate trained models on test datasets
- Compute loss and custom metrics
- Preserve the model's training state during evaluation

### Visualization

- Visualize training and validation loss curves
- Visualize optional training and validation metric curves
- Optionally highlight the best validation loss and/or the best validation metric

> ⏳ More features coming soon...

## Installation

### 🔧 Development version (from source)

If you want to install the latest development version:

```bash
git clone https://github.com/<your-username>/DanFlow.git
cd DanFlow
pip install -e .
```

## Quick Start

This example shows how to train a simple PyTorch model using DanFlow's `Trainer`.

DanFlow provides a clean wrapper around the PyTorch training loop, including training, validation, logging, and history tracking.

---

### 1. Import dependencies

```python
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from danflow.training.trainer import Trainer
```

---

### 2. Create a simple model

```python
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)
```

---

### 3. Prepare dataset

```python
x = torch.randn(200, 10)
y = torch.randint(0, 2, (200,))

dataset = TensorDataset(x, y)
train_loader = DataLoader(dataset, batch_size=16)
valid_loader = DataLoader(dataset, batch_size=16)
```

---

### 4. Define training component

```python
optimizer = Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
```

---

### 5. Train the model

```python
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn
)

history = trainer.fit(
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=5
)
```

---

### 6. View the results

```python
print("Best validation loss:", history["best_valid_loss"])
```

## Contributing

Contributions are welcome!

If you find a bug, please open an issue on GitHub.  
You are also welcome to fix it yourself and submit a pull request.

If you would like to add a new feature or discuss a major change, please open an issue first so we can align on the direction of the project.

You can reach me through:
- GitHub: https://github.com/Daniyaliranmehr/DanFlow/issues
- LinkedIn: https://www.linkedin.com/in/daniyaliranmehr
- Email: daniyaliranmehr@gmail.com

## License

This project is licensed under the MIT License.

## Author

DanFlow was created and is maintained by **Daniyal Iran Mehr**.