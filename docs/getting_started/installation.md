# Installation

DanFlow is a PyTorch-based library for model training, evaluation, data preparation, and visualization.

This guide covers installing DanFlow, preparing its dependencies, and verifying that the installation works correctly.

## Requirements

DanFlow requires:

* Python 3.10 or newer
* PyTorch
* NumPy
* pandas
* Matplotlib
* tqdm
* rich
* PrettyTable

The project metadata currently declares only `torch` and `numpy` as package dependencies, while the source code also imports `pandas`, `matplotlib`, `tqdm`, `rich`, and `prettytable`.

These runtime dependencies should be declared in `pyproject.toml` so that a standard installation installs a complete DanFlow environment automatically.

### Optional Dependencies

Some documentation examples use `torchmetrics` for stateful training metrics.

Install it separately when needed:

```bash
pip install torchmetrics
```

`torchmetrics` is not required by the core DanFlow package itself.

## Install from PyPI

Once DanFlow is published on PyPI, install it with:

```bash
pip install danflow
```

Then verify the installation:

```bash
python -c "import danflow; print('DanFlow installed successfully')"
```

The command should complete without an import error.

## Install from Source

To install the current development version directly from the repository, clone the project and install it in editable mode:

```bash
git clone https://github.com/Daniyaliranmehr/DanFlow.git
cd DanFlow
pip install -e .
```

Editable installation is useful when working on DanFlow itself because changes to the source code are immediately available without reinstalling the package.

Verify the installation:

```bash
python -c "import danflow; print('DanFlow installed successfully')"
```

## Install with Optional Dependencies

When working with examples that use `torchmetrics`, install the package together with the optional dependency:

```bash
pip install torchmetrics
```

For development environments, the project should expose a `dev` optional dependency group through `pyproject.toml`. After that group is defined, a development installation can be performed with:

```bash
pip install -e ".[dev]"
```

The `dev` group is intended for contributors working on the DanFlow source code, tests, or documentation.

## Verify the Environment

After installation, verify that the main package can be imported:

```bash
python -c "import danflow; print('DanFlow installed successfully')"
```

You can also verify the main public namespaces:

```python
import danflow
import danflow.data
import danflow.losses
import danflow.training
import danflow.visualization
```

If these imports complete without an exception, the main DanFlow modules are available in the current Python environment.

## Version Compatibility

The project currently specifies the following Python requirement:

| Component    | Requirement                                     |
| ------------ | ----------------------------------------------- |
| Python       | `>= 3.10`                                       |
| PyTorch      | No minimum version currently declared           |
| NumPy        | No minimum version currently declared           |
| pandas       | No minimum version currently declared           |
| Matplotlib   | No minimum version currently declared           |
| tqdm         | No minimum version currently declared           |
| rich         | No minimum version currently declared           |
| PrettyTable  | No minimum version currently declared           |
| torchmetrics | Optional; no minimum version currently declared |

Exact dependency versions should be added to the project metadata once DanFlow establishes its supported version range. The documentation should then be updated to match those constraints.

## Common Installation Issues

### `ModuleNotFoundError`

An error such as:

```text
ModuleNotFoundError: No module named 'tqdm'
```

usually means that a runtime dependency is missing from the environment.

Install the missing package directly when working with the current source tree:

```bash
pip install tqdm
```

The long-term solution is to declare all runtime dependencies in `pyproject.toml` so they are installed automatically with DanFlow.

### Wrong Python Environment

If DanFlow imports correctly in one terminal but not another, verify that the same Python environment is being used for both installation and execution:

```bash
python --version
python -m pip --version
```

Using `python -m pip` helps ensure that `pip` belongs to the selected Python interpreter.

## Summary

A standard DanFlow environment should provide:

```text
Python >= 3.10
PyTorch
NumPy
pandas
Matplotlib
tqdm
rich
PrettyTable
```

Install `torchmetrics` separately when using examples or workflows that require TorchMetrics-based stateful metrics.

After installation, verify the environment with:

```bash
python -c "import danflow; print('DanFlow installed successfully')"
```