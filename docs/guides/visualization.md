# Visualizing Data and Training

## Introduction

Visualization is useful throughout the machine-learning workflow.

Before training, visualizations can help you understand the structure and distribution of the dataset.

During and after training, training-history plots can help you understand whether the model is learning, whether validation performance is improving, and whether training and validation behavior are beginning to diverge.

DanFlow organizes its visualization utilities into two groups:

```text id="p8xk2r"
Data Visualization
    Correlation
    Distributions
    Spread

Training Visualization
    Loss
    Metrics
    Combined Training History
```

The important question is not simply:

> Which plotting function should I call?

Instead, ask:

> What do I want to learn from this visualization?

This guide focuses on that decision-making process.


## Data Visualization

The data-visualization functions operate on pandas `DataFrame` objects.

They are most useful during exploratory data analysis, before model training and sometimes between preprocessing stages.

The main questions they can help answer are:

```text id="8h6jaf"
How are features distributed?
    Histograms

How much do feature values vary?
    Box plots

How are numerical features related?
    Correlation heatmap

How do several features compare?
    Multiple histograms
    Multiple box plots
```


## Understanding Feature Distributions

Before training a model, it is useful to understand how individual features are distributed.

A feature may be:

* concentrated around a narrow range,
* broadly distributed,
* strongly asymmetric,
* or affected by unusual observations.

A histogram is usually the first visualization to use when the main question is:

> What does the distribution of this variable look like?

### Histograms

Use `plot_histogram()` when you want to inspect one feature in detail.

```python id="v0w9a3"
from danflow.visualization.data import plot_histogram

plot_histogram(
    data,
    column="feature_1",
)
```

A histogram divides the observed values into bins and shows how many observations fall into each interval.

### What to Look For

When interpreting a histogram, examine:

```text id="myqfdd"
Center
    Where most values are concentrated

Spread
    How widely the observations vary

Shape
    The overall distribution pattern

Unusual regions
    Areas with very few observations
```

A histogram can reveal patterns that are not obvious from summary statistics alone.

### Choosing the Number of Bins

The number of bins controls how much detail is visible.

Too few bins can hide important structure.

Too many bins can make the distribution look unnecessarily noisy.

DanFlow uses a default of `50` bins, but the appropriate value depends on the dataset.

For a small dataset, experimenting with a smaller number of bins can make the distribution easier to interpret.

For example:

```python id="gr2h0b"
plot_histogram(
    data,
    column="feature_1",
    bins=10,
)
```

The goal is not to find a universally correct number of bins. The goal is to choose a resolution that makes the distribution understandable.


## Comparing Several Feature Distributions

Sometimes the question is not:

> What is the distribution of one feature?

but:

> How do several features differ in their distributions?

In that situation, use `plot_multi_histograms()`.

```python id="xqy9km"
from danflow.visualization.data import plot_multi_histograms

plot_multi_histograms(
    data,
    columns=[
        "feature_1",
        "feature_2",
        "feature_3",
    ],
)
```

Each column is displayed in its own histogram.

This is useful for comparing:

* relative spread,
* concentration,
* distribution shape,
* and unusual values across several variables.

### Do Not Compare Unrelated Scales Without Context

Suppose one feature contains values between `0` and `1` while another contains values between `0` and `100000`.

Their histograms can still be useful, but the numerical scales must be considered before drawing conclusions about which distribution is "larger" or "more spread out."

Visualization should complement the numerical analysis, not replace it.


## Understanding Feature Spread and Potential Outliers

A histogram is useful for understanding distribution shape, but it is not always the easiest way to identify unusually large or small observations.

For that question, a box plot is often more useful.

### Box Plots

Use `plot_boxplot()` when you want a compact view of the spread of a single numerical feature.

```python id="4j4v1q"
from danflow.visualization.data import plot_boxplot

plot_boxplot(
    data,
    column="feature_2",
)
```

A box plot provides a compact representation of:

```text id="a16b3d"
Central tendency
    Median

Spread
    Interquartile range

Range
    Whiskers

Potential unusual observations
    Points outside the whisker range
```

### What to Look For

Use a box plot when you want to answer questions such as:

* Is the feature tightly concentrated?
* How wide is the central range?
* Is the distribution strongly asymmetric?
* Are there observations that deserve further investigation?

A potential outlier is not automatically an error.

It may represent:

* a real but rare observation,
* a different subgroup,
* measurement variation,
* or a data-quality problem.

The visualization identifies observations that deserve attention; it does not decide how they should be handled.


## Comparing the Spread of Several Features

Use `plot_multi_boxplots()` when you want to compare feature spread side by side.

```python id="j4k6fb"
from danflow.visualization.data import plot_multi_boxplots

plot_multi_boxplots(
    data,
    columns=[
        "feature_1",
        "feature_2",
        "feature_3",
    ],
)
```

This is useful when the main question is:

> How do the central range and variability of these features compare?

Multiple box plots can make differences in median, spread, and potential unusual observations much easier to compare than separate figures.

As with histograms, remember that features with substantially different scales should be interpreted in their numerical context.


## Understanding Relationships Between Features

A distribution tells you about one variable.

Sometimes the more important question is:

> How are the numerical variables related to each other?

For that purpose, use a correlation heatmap.

### Correlation Heatmaps

Use `plot_correlation_heatmap()` to inspect the correlation matrix of numerical columns.

```python id="f6g0im"
from danflow.visualization.data import plot_correlation_heatmap

plot_correlation_heatmap(
    data,
)
```

DanFlow uses numerical columns for this visualization.

Categorical columns are not part of the correlation matrix generated by this function.

### What Correlation Shows

Correlation describes the strength and direction of a numerical relationship.

A correlation near:

```text id="cz0qsl"
+1
```

indicates a strong positive linear relationship.

A correlation near:

```text id="tzl6be"
-1
```

indicates a strong negative linear relationship.

A correlation near:

```text id="q53qjs"
0
```

indicates little or no linear relationship.

### Correlation Is Not Causation

A high correlation does not establish that one feature causes the other to change.

For example, two variables may be correlated because:

* both are influenced by another variable,
* they measure related aspects of the same process,
* or the relationship exists only within the observed dataset.

Use the heatmap to identify relationships worth investigating, not to establish causal conclusions.

### Correlation Is Not the Whole Relationship

Correlation is especially useful for identifying linear relationships.

A low correlation does not necessarily mean that two variables are completely unrelated.

A nonlinear relationship can exist even when linear correlation is small.

Therefore, a heatmap should be treated as one exploratory tool rather than a complete analysis of feature relationships.


## Choosing the Right Data Plot

A simple decision process is useful:

| Question                                         | Recommended plot    |
| ------------------------------------------------ | ------------------- |
| What does one feature's distribution look like?  | Histogram           |
| How do several feature distributions compare?    | Multiple histograms |
| What is the spread of one feature?               | Box plot            |
| How does the spread of several features compare? | Multiple box plots  |
| How are numerical features linearly related?     | Correlation heatmap |

This keeps visualization focused on the question you are trying to answer.


## Combining Data Visualizations

A single visualization rarely provides a complete picture of a dataset.

A practical exploratory workflow can therefore combine several views:

```text id="ydx7lq"
Correlation
    Identify numerical relationships

Histograms
    Understand feature distributions

Box plots
    Inspect spread and unusual observations
```

For example, if a feature appears strongly correlated with another feature, inspect their distributions before deciding whether the relationship is meaningful.

Likewise, if a box plot reveals unusual observations, inspect the corresponding histogram and the original records before deciding whether those observations should be removed or transformed.


# Training Visualization

DanFlow also provides visualization utilities for the history returned by `Trainer.fit()`.

Training plots answer a different set of questions:

```text id="jyr9yv"
Is the model learning?
    Loss history

Is the model performance improving?
    Metric history

Are training and validation behaving differently?
    Combined training history
```

Training-history plots should normally be used after a model has been trained for multiple epochs.


## Understanding Loss History

Use `plot_loss_history()` when the main question is:

> How did training and validation loss change over the course of training?

```python id="w0go8u"
from danflow.visualization import plot_loss_history

plot_loss_history(
    history,
    name="Demo Model",
)
```

The plot compares:

```text id="zzb2lo"
Training loss
Validation loss
```

### A Useful Learning Pattern

A common pattern during effective training is:

```text id="wqsj58"
Training loss decreases
Validation loss decreases
```

This indicates that the model's error is decreasing on both datasets.

### Possible Overfitting Pattern

Another common pattern is:

```text id="qzq3sc"
Training loss continues decreasing
Validation loss stops improving
Validation loss begins increasing
```

This can indicate that the model is fitting the training data more strongly than it generalizes to validation data.

The visualization does not prove the cause by itself, but it gives you a strong signal that the training process should be investigated.

### Loss Should Be Interpreted in Context

The numerical scale of a loss depends on the loss function.

For example, MSE, cross-entropy, and custom losses do not share the same numerical interpretation.

When comparing losses, always know which loss function produced them.


## Highlighting the Best Validation Loss

DanFlow can mark the best validation-loss epoch when the corresponding history information is available.

```python id="3f3t09"
plot_loss_history(
    history,
    name="Demo Model",
    show_best_loss=True,
)
```

The marked point represents the epoch recorded as the best validation-loss epoch.

### Important Checkpointing Detail

Use `show_best_loss=True` only when the history contains a valid `best_loss_epoch`.

In the normal DanFlow workflow, this information is tied to best-model tracking during:

```python id="aokg8o"
trainer.fit(
    ...,
    save_best=True,
)
```

When `save_best=False`, the current `Trainer` implementation does not establish a meaningful best validation-loss epoch in the same way.

Therefore, do not enable the best-loss marker blindly on every history dictionary.


## Understanding Metric History

Use `plot_metric_history()` when the main question is:

> How did the training and validation metric change over the epochs?

```python id="2r2q9f"
from danflow.visualization import plot_metric_history

plot_metric_history(
    history,
    name="Demo Model",
)
```

The plot compares:

```text id="rsxk8g"
Training metric
Validation metric
```

### What to Look For

A useful pattern is:

```text id="g7w3r4"
Training metric improves
Validation metric improves
```

This suggests that the model is improving on both the training and validation data.

Another pattern is:

```text id="c7x4km"
Training metric continues improving
Validation metric stops improving
```

This can indicate that the model is becoming increasingly specialized to the training data.

### Metric Direction Matters

Do not assume that larger values are always better.

For example:

* accuracy is generally interpreted as higher-is-better,
* error-based metrics such as MSE are interpreted as lower-is-better.

The meaning of a metric must therefore be considered before interpreting its curve.

DanFlow's training history records the metric values, but the visualization itself does not determine the semantic direction of the metric.


## Highlighting the Best Validation Metric

When the training history contains `best_metric_epoch`, the metric history can mark that point:

```python id="z5gdk9"
plot_metric_history(
    history,
    name="Demo Model",
    show_best_metric=True,
)
```

The marker is useful when you want to connect the training curve to the epoch selected during model development.

Always interpret the marked epoch according to the metric being used and the training configuration that produced the history.


## Combined Training History

Sometimes inspecting loss and metric separately is unnecessary.

Use `plot_training_history()` when you want a combined view:

```python id="k7g9ha"
from danflow.visualization import plot_training_history

plot_training_history(
    history,
    name="Demo Model",
    show_best_loss=True,
    show_best_metric=True,
)
```

This is particularly useful when you want to inspect multiple signals together:

```text id="76q7tq"
Loss
    Training
    Validation

Metric
    Training
    Validation

Best epochs
    Best validation loss
    Best validation metric
```

### When to Prefer the Combined Plot

Use the combined plot when:

* the training history contains both loss and metric,
* you want a compact overview of the training process,
* and comparing the two signals together is more useful than examining them separately.

Use the individual loss or metric plots when one specific signal deserves closer attention.


## Reading Training Curves as a Whole

Training curves should not be interpreted from a single epoch.

Look at the overall trajectory.

A useful analysis asks:

```text id="n9x4tv"
Did training loss improve?

Did validation loss improve?

Did the metric improve?

Did training and validation remain reasonably close?

Did validation performance stop improving before training ended?

At which epoch did the validation behavior become most useful?
```

This is more informative than simply checking the final epoch.

For example, if validation loss reached its lowest value several epochs before the end of training while training loss continued decreasing, the final epoch may not be the most useful model state.


## Choosing the Right Training Plot

| Question                                     | Recommended visualization                   |
| -------------------------------------------- | ------------------------------------------- |
| How did train and validation loss change?    | `plot_loss_history()`                       |
| How did train and validation metric change?  | `plot_metric_history()`                     |
| How did the overall training process behave? | `plot_training_history()`                   |
| Where was the best validation-loss epoch?    | Loss history with `show_best_loss=True`     |
| Where was the best validation-metric epoch?  | Metric history with `show_best_metric=True` |

---

## Saving Visualization Outputs

DanFlow's data-visualization functions and training-history functions handle `save_path` differently.

### Data Visualization

For data plots, `save_path` can represent either a file path or a directory-like path.

For example:

```python id="j7kb3p"
plot_histogram(
    data,
    column="feature_1",
    save_path="plots/feature_1_histogram.png",
)
```

can explicitly select the output filename.

A directory-like path can also be used:

```python id="9kj1pn"
plot_boxplot(
    data,
    column="feature_1",
    save_path="plots",
)
```

In that case, DanFlow generates the corresponding default filename inside the directory.

The same general behavior applies to the other data-visualization functions, with filenames derived from the feature or plot name.

### Training Visualization

Training-history functions treat `save_path` as the output file path.

Prefer an explicit filename:

```python id="f0nq7v"
plot_training_history(
    history,
    name="Demo Model",
    save_path="plots/training_history.png",
)
```

Do not assume that passing a directory to a training-history function has the same meaning as passing a directory to a data-visualization function.

For reproducible documentation and project outputs, explicit file paths are clearer.


## Organizing Visualization Outputs

For a project containing both exploratory data plots and training plots, keep generated files separate from source code.

For example:

```text id="m1y1d8"
project/
    data/
        raw/
        processed/
    plots/
        correlation_heatmap.png
        feature_1_histogram.png
        feature_spread.png
        loss_history.png
        metric_history.png
        training_history.png
    train.py
```

This keeps generated artifacts easy to inspect without mixing them with implementation files.


## Common Visualization Mistakes

### Using the Wrong Plot for the Question

Do not use a correlation heatmap when you are trying to understand the distribution of one feature.

Use a histogram for distribution.

Use a box plot for compact spread and potential unusual observations.

Use a correlation heatmap for numerical relationships.

### Treating Correlation as Causation

A strong correlation is not evidence that changing one variable causes another variable to change.

Use the heatmap as an exploratory tool.

### Ignoring Numerical Scale

Two features can have very different numerical ranges.

A larger numerical range does not automatically mean greater importance, greater variance, or greater predictive value.

Interpret the plot in the context of the underlying data.

### Treating Every Box-Plot Outlier as an Error

A point outside a whisker may be a legitimate observation.

Investigate the underlying record and domain context before removing it.

### Using Too Many Histogram Bins

Very small bins can make a histogram look noisy and can distract from the underlying distribution.

Very large bins can hide meaningful structure.

Adjust `bins` according to the dataset size and the question being investigated.

### Looking Only at the Final Training Epoch

The final epoch is not necessarily the most informative point in the training history.

Always inspect the validation trajectory across epochs.

### Interpreting a Metric Without Knowing Its Direction

Do not assume that a larger metric is always better.

Know what the metric measures before identifying an apparent improvement.

### Using Best-Epoch Markers Without Valid History Information

`show_best_loss=True` and `show_best_metric=True` depend on the corresponding best-epoch fields being meaningful.

Do not enable these markers without understanding where those fields came from.

### Treating Data and Training `save_path` the Same Way

Data plots can use a directory-like save path with automatic filenames.

Training-history plots should be given an explicit output file path when saving.


## Recommended Visualization Workflow

A practical visualization workflow can be organized into two stages.

### 1. Inspect the Dataset

Start with the questions you need to answer.

```text id="jqt8s1"
Feature distribution
    Histogram

Feature spread
    Box plot

Relationships between numerical features
    Correlation heatmap
```

Use multiple plots when a single plot does not answer the question completely.

### 2. Inspect Training Behavior

After training:

```text id="0cwb2m"
Loss
    Train and validation loss

Metric
    Train and validation metric

Combined history
    Loss + metric
```

Look for:

* consistent improvement,
* divergence between training and validation,
* possible overfitting,
* and the epoch at which validation behavior is most useful.

### 3. Save Important Figures

Save the plots that support decisions or document the experiment.

For example:

```text id="xwm9xb"
plots/
    data/
        correlation_heatmap.png
        feature_distributions.png
        feature_spread.png

    training/
        loss_history.png
        metric_history.png
        training_history.png
```

### 4. Connect Visualization to the Rest of the Workflow

Visualization should not be an isolated step.

A useful workflow is:

```text id="0t6mcz"
Data Preparation
    Inspect dataset

Model Training
    Record history

Visualization
    Inspect data
    Inspect training behavior

Evaluation
    Measure final performance
```

Each visualization should answer a question that helps you understand one of these stages.


## Final Perspective

Good visualization is not about generating as many plots as possible.

It is about selecting a representation that answers a specific question.

For dataset exploration:

* use histograms to understand distributions,
* use box plots to inspect spread and potential unusual observations,
* use a correlation heatmap to explore relationships between numerical features.

For training analysis:

* use loss history to inspect optimization and validation behavior,
* use metric history to inspect model performance,
* use combined training history when both signals need to be considered together.

The most useful plots are the ones that change or clarify your understanding of the data or model.

For complete executable visualization examples, see [Visualization](../examples/visualization.md).