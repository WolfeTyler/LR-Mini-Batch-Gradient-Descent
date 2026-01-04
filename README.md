# Mini-Batch Gradient Descent — with a Loss-Surface “Topographic Map” Visualization

This notebook implements **mini-batch gradient descent** to fit a simple **linear regression** model and uses my favorite visualization—**a loss surface contour plot in (W, b) space**—to make gradient descent feel intuitive.

At the core, we’re learning parameters for a line:

\[
\hat{y} = Wx + b
\]

Where:
- **W** is the slope
- **b** is the intercept

---

## Why this notebook is worth running

Most explanations of gradient descent show only:
- the data + best-fit line, and/or
- a loss curve over time

That’s helpful, but it still feels abstract.

This notebook adds a more “visual brain friendly” view:

### Visual D — Loss surface in (W, b) space + GD path (the star of the show)

Think of the contour plot like a **topographic map**, but instead of showing “height,” it shows **how wrong the model is** (its **MSE loss**).

- Each contour ring is a set of \((W, b)\) values with the **same loss**
- Outer rings = **higher error**
- Inner rings = **lower error**
- The center of the smallest ring is the **best-fitting line**

The **blue path** overlaid on top shows how **mini-batch gradient descent walks downhill**:
- It starts from an initial guess \((W_0, b_0)\)
- Each step updates \(W\) and \(b\) to reduce loss
- The path curves because the loss surface is shaped like a **tilted bowl/valley**
- Near the minimum, mini-batch updates can look a little “wiggly” because each step uses only a subset of data (mini-batch noise)

If you understand this plot, you understand gradient descent.

---

## What’s inside

### 1) A single MSE gradient step (the core math)

We minimize mean squared error:

\[
\text{MSE}(W,b)=\frac{1}{n}\sum_{i=1}^{n}\left(y_i-(X_iW+b)\right)^2
\]

Gradients:

\[
\nabla_W = -\frac{2}{n}X^\top (y-\hat{y}),
\qquad
\nabla_b = -\frac{2}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)
\]

Update rule:

\[
W \leftarrow W - \alpha \nabla_W,
\qquad
b \leftarrow b - \alpha \nabla_b
\]

In code, this becomes a function that:
1. predicts \(\hat{y}\)
2. computes gradients
3. updates \(W\) and \(b\)
4. returns updated params and the batch loss

---

### 2) Mini-batch gradient descent loop

Instead of using all data every step (full-batch GD) or one point at a time (SGD), we use **mini-batches**:

- Shuffle the dataset each epoch
- Split into mini-batches of size `batch_size`
- Apply the gradient step for each batch
- Track loss and parameter history

This gives you the best of both worlds:
- faster than full-batch
- smoother/more stable than pure SGD

---

### 3) Visuals that show learning from multiple angles

The notebook includes several visuals, but the key ones are:

- **Data + predicted line** (what the model is fitting)
- **MSE vs iteration** (training curve)
- **W and b over time** (parameter convergence)
- **Visual D: Contours + GD path** (the “topographic map” view)

---

## How the loss-surface contour plot is created (Visual D)

This is the sequence that generates the visualization:

### Step A — Make a grid of possible parameter values
We choose a range of slopes and intercepts:

- `w_vals = np.linspace(w_min, w_max, ...)`
- `b_vals = np.linspace(b_min, b_max, ...)`

Then build a mesh:

- `W_grid, B_grid = np.meshgrid(w_vals, b_vals)`

### Step B — Compute the loss at every point on the grid
For each \((W, b)\) pair, we compute:

\[
\text{MSE}(W,b)
\]

This produces a 2D “loss landscape” the same shape as the mesh.

### Step C — Plot MSE contours
Using matplotlib’s contour plotting, we draw lines of equal loss.

This is where the “topographic map” comes from.

### Step D — Overlay the mini-batch GD path
During training we store parameter history:

- `W_history`
- `b_history`

Then we plot those points directly on top of the contour map.

That blue line is the algorithm “walking downhill.”

---

## How to run

1. Open the notebook.
2. Run all cells.
3. If you have a dataset (e.g., `data.csv`), place it next to the notebook.
4. Otherwise, the notebook can generate synthetic linear data for demonstration.

You can tweak:
- learning rate (`alpha`)
- batch size (`batch_size`)
- number of epochs / iterations

…and then re-run Visual D to see how the path changes.

---

## What to try next (recommended experiments)

- **Increase learning rate**: the path may overshoot or spiral
- **Decrease learning rate**: smoother, slower convergence
- **Batch size = 1** (SGD): path gets noisy/wiggly
- **Batch size = full dataset**: path becomes smooth and deterministic

Watching how the blue path changes makes these concepts click fast.

---

## Credits / notes

This notebook is designed to be an *intuition builder*:
- The goal isn’t just “train a line”
- It’s to **see** gradient descent moving through parameter space

If you can explain Visual D to someone else, you fully understand what gradient descent is doing under the hood.
