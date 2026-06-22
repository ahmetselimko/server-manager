import numpy as np
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import pickle
import json


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Create sliding window sequences for LSTM input"""
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)


def train_test_split_time(X, y, test_ratio=0.2):
    """Temporal split - never shuffle time series!"""
    split = int(len(X) * (1 - test_ratio))
    return X[:split], X[split:], y[:split], y[split:]


# ─────────────────────────────────────────────
# 1. RANDOM FOREST (NumPy implementation)
# ─────────────────────────────────────────────

class DecisionTreeRegressor:
    """Single decision tree for RF ensemble"""

    def __init__(self, max_depth=8, min_samples_split=5, max_features=None, rng=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = rng or np.random.default_rng()
        self.tree = None

    def _mse(self, y):
        if len(y) == 0:
            return 0.0
        return float(np.var(y))

    def _best_split(self, X, y):
        n_features = X.shape[1]
        n_try = self.max_features or max(1, int(np.sqrt(n_features)))
        feature_idxs = self.rng.choice(n_features, size=n_try, replace=False)

        best_feat, best_thresh, best_gain = None, None, -np.inf
        parent_mse = self._mse(y)

        for feat in feature_idxs:
            thresholds = np.percentile(X[:, feat], [25, 50, 75])
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                    continue
                gain = parent_mse - (
                    left_mask.sum() / len(y) * self._mse(y[left_mask]) +
                    right_mask.sum() / len(y) * self._mse(y[right_mask])
                )
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh

        return best_feat, best_thresh

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.std(y) < 1e-6:
            return {"leaf": True, "value": float(np.mean(y))}
        feat, thresh = self._best_split(X, y)
        if feat is None:
            return {"leaf": True, "value": float(np.mean(y))}
        mask = X[:, feat] <= thresh
        return {
            "leaf": False, "feat": feat, "thresh": thresh,
            "left":  self._build(X[mask],  y[mask],  depth + 1),
            "right": self._build(X[~mask], y[~mask], depth + 1),
        }

    def fit(self, X, y):
        self.tree = self._build(X, y, 0)
        return self

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["value"]
        if x[node["feat"]] <= node["thresh"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])


class RandomForestRegressor:
    """Ensemble of bootstrapped decision trees"""

    def __init__(self, n_estimators=40, max_depth=8, max_features=None,
                 min_samples_split=5, rng_seed=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.rng = np.random.default_rng(rng_seed)
        self.trees: List[DecisionTreeRegressor] = []
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n = len(X)
        self.trees = []
        importance_sum = np.zeros(X.shape[1])

        for i in range(self.n_estimators):
            # Bootstrap sample
            idx = self.rng.integers(0, n, size=n)
            Xb, yb = X[idx], y[idx]
            tree_rng = np.random.default_rng(int(self.rng.integers(0, 2**31)))
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                rng=tree_rng
            )
            tree.fit(Xb, yb)
            self.trees.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.stack([t.predict(X) for t in self.trees], axis=1)
        return preds.mean(axis=1)

    def score(self, X, y):
        yhat = self.predict(X)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-10)


# ─────────────────────────────────────────────
# 2. LSTM (NumPy implementation)
# ─────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

def tanh(x):
    return np.tanh(np.clip(x, -10, 10))


class LSTMCell:
    """Single LSTM cell, forward pass + BPTT"""

    def __init__(self, input_size: int, hidden_size: int, rng):
        k = 1.0 / np.sqrt(hidden_size)
        def W(r, c): return rng.uniform(-k, k, (r, c))
        # Gates: forget, input, output, cell
        self.Wf = W(hidden_size, input_size);  self.Uf = W(hidden_size, hidden_size); self.bf = np.zeros(hidden_size)
        self.Wi = W(hidden_size, input_size);  self.Ui = W(hidden_size, hidden_size); self.bi = np.zeros(hidden_size)
        self.Wo = W(hidden_size, input_size);  self.Uo = W(hidden_size, hidden_size); self.bo = np.zeros(hidden_size)
        self.Wc = W(hidden_size, input_size);  self.Uc = W(hidden_size, hidden_size); self.bc = np.zeros(hidden_size)

    def forward(self, x, h_prev, c_prev):
        f = sigmoid(self.Wf @ x + self.Uf @ h_prev + self.bf)
        i = sigmoid(self.Wi @ x + self.Ui @ h_prev + self.bi)
        o = sigmoid(self.Wo @ x + self.Uo @ h_prev + self.bo)
        g = tanh(self.Wc @ x + self.Uc @ h_prev + self.bc)
        c = f * c_prev + i * g
        h = o * tanh(c)
        cache = (x, h_prev, c_prev, f, i, o, g, c, h)
        return h, c, cache


class LSTMModel:
    """
    Single-layer LSTM → dense output.
    Trained with BPTT + Adam.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1,
                 rng_seed: int = 42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        rng = np.random.default_rng(rng_seed)
        self.lstm = LSTMCell(input_size, hidden_size, rng)
        # Dense output layer
        k = 1.0 / np.sqrt(hidden_size)
        self.Wy = rng.uniform(-k, k, (output_size, hidden_size))
        self.by = np.zeros(output_size)

        # Adam state - collect all params
        self._init_adam()

    def _init_adam(self):
        self._adam_t = 0
        self._adam_lr = 0.001
        self._beta1, self._beta2, self._eps = 0.9, 0.999, 1e-8
        # For simplicity, store params as flat list
        self._params = self._get_params()
        self._m = [np.zeros_like(p) for p in self._params]
        self._v = [np.zeros_like(p) for p in self._params]

    def _get_params(self):
        l = self.lstm
        return [l.Wf,l.Uf,l.bf, l.Wi,l.Ui,l.bi, l.Wo,l.Uo,l.bo, l.Wc,l.Uc,l.bc,
                self.Wy, self.by]

    def _set_params(self, params):
        l = self.lstm
        l.Wf,l.Uf,l.bf, l.Wi,l.Ui,l.bi, l.Wo,l.Uo,l.bo, l.Wc,l.Uc,l.bc = params[:12]
        self.Wy, self.by = params[12], params[13]

    def _forward_with_cache(self, X_seq: np.ndarray):
        """
        Forward pass that saves all intermediate values needed for BPTT.
        X_seq : (seq_len, input_size)
        Returns: (y_pred: float, h_final: ndarray, caches: list)
        """
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        caches = []
        for t in range(X_seq.shape[0]):
            h, c, cache = self.lstm.forward(X_seq[t], h, c)
            caches.append(cache)
        y_pred = (self.Wy @ h + self.by).item()
        return y_pred, h, caches

    def _backward_from_cache(self, caches: list, h_final: np.ndarray,
                              dy_pred: float) -> list:
        """
        Full BPTT from saved forward caches.

        dy_pred : dL/d(y_pred) — scalar, already scaled by batch size
                  e.g.  2 * (y_pred - y_true) / batch_size   for MSE

        Returns : grad list in the same order as _get_params()
                  [dWf,dUf,dbf, dWi,dUi,dbi, dWo,dUo,dbo, dWc,dUc,dbc, dWy,dby]
        """
        lstm = self.lstm

        # ── Output layer ────────────────────────────────────────────
        dWy = dy_pred * h_final[np.newaxis, :]   # (1, hidden)
        dby = np.array([dy_pred])
        dh  = self.Wy[0] * dy_pred               # (hidden,)
        dc  = np.zeros(self.hidden_size)

        # ── Accumulators for LSTM weight gradients ───────────────────
        dWf = np.zeros_like(lstm.Wf); dUf = np.zeros_like(lstm.Uf); dbf = np.zeros_like(lstm.bf)
        dWi = np.zeros_like(lstm.Wi); dUi = np.zeros_like(lstm.Ui); dbi = np.zeros_like(lstm.bi)
        dWo = np.zeros_like(lstm.Wo); dUo = np.zeros_like(lstm.Uo); dbo = np.zeros_like(lstm.bo)
        dWc = np.zeros_like(lstm.Wc); dUc = np.zeros_like(lstm.Uc); dbc = np.zeros_like(lstm.bc)

        # ── BPTT ────────────────────────────────────────────────────
        for t in reversed(range(len(caches))):
            x, h_prev, c_prev, f, ig, o, g, c_t, _ = caches[t]

            tanh_ct = np.tanh(c_t)

            # h = o * tanh(c)
            do   = dh * tanh_ct
            dc  += dh * o * (1.0 - tanh_ct ** 2)   # += because dc can come from future step

            # c = f*c_prev + ig*g
            df      = dc * c_prev
            di      = dc * g
            dg_val  = dc * ig
            dc_prev = dc * f

            # Through gate activations (sigmoid' and tanh')
            df_pre = df    * f  * (1.0 - f)
            di_pre = di    * ig * (1.0 - ig)
            do_pre = do    * o  * (1.0 - o)
            dg_pre = dg_val * (1.0 - g ** 2)

            # Accumulate weight gradients (outer product = rank-1 update)
            dWf += np.outer(df_pre, x);  dUf += np.outer(df_pre, h_prev); dbf += df_pre
            dWi += np.outer(di_pre, x);  dUi += np.outer(di_pre, h_prev); dbi += di_pre
            dWo += np.outer(do_pre, x);  dUo += np.outer(do_pre, h_prev); dbo += do_pre
            dWc += np.outer(dg_pre, x);  dUc += np.outer(dg_pre, h_prev); dbc += dg_pre

            # Flow gradient back to previous timestep through recurrent weights
            dh = (lstm.Uf.T @ df_pre + lstm.Ui.T @ di_pre +
                  lstm.Uo.T @ do_pre + lstm.Uc.T @ dg_pre)
            dc = dc_prev

        return [dWf, dUf, dbf, dWi, dUi, dbi, dWo, dUo, dbo, dWc, dUc, dbc, dWy, dby]

    def forward_sequence(self, X_seq):
        """X_seq: (seq_len, input_size) → scalar output"""
        y_pred, h, caches = self._forward_with_cache(X_seq)
        c_final = caches[-1][7]   # index 7 in cache tuple = c_t
        return y_pred, h, c_final, caches

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (batch, seq_len, features)"""
        preds = []
        for i in range(len(X)):
            y, _, _, _ = self.forward_sequence(X[i])
            preds.append(y)
        return np.array(preds)

    def _adam_update(self, grads):
        self._adam_t += 1
        t = self._adam_t
        params = self._get_params()
        new_params = []
        for idx, (p, g) in enumerate(zip(params, grads)):
            if g.ndim > p.ndim:
                g = np.mean(g, axis=0)
            self._m[idx] = self._beta1 * self._m[idx] + (1 - self._beta1) * g
            self._v[idx] = self._beta2 * self._v[idx] + (1 - self._beta2) * g**2
            m_hat = self._m[idx] / (1 - self._beta1**t)
            v_hat = self._v[idx] / (1 - self._beta2**t)
            new_params.append(p - self._adam_lr * m_hat / (np.sqrt(v_hat) + self._eps))
        self._set_params(new_params)

    def train_epoch(self, X: np.ndarray, y: np.ndarray,
                    batch_size: int = 64) -> float:
        """
        Mini-batch training with exact BPTT gradients + Adam + gradient clipping.
        """
        n          = len(X)
        idxs       = np.random.permutation(n)
        total_loss = 0.0
        n_params   = len(self._get_params())
        clip_val   = 5.0   # global gradient-norm clipping threshold

        for start in range(0, n, batch_size):
            batch_idx = idxs[start:start + batch_size]
            Xb, yb    = X[batch_idx], y[batch_idx]
            bs        = len(Xb)

            grad_sum   = [np.zeros_like(p) for p in self._get_params()]
            batch_loss = 0.0

            for j in range(bs):
                # Single forward pass — saves caches for free
                y_pred_j, h_j, caches_j = self._forward_with_cache(Xb[j])
                err_j       = y_pred_j - yb[j]
                batch_loss += err_j ** 2
                dy_j        = 2.0 * err_j / bs        # batch-averaged MSE gradient
                grads_j     = self._backward_from_cache(caches_j, h_j, dy_j)
                for k in range(n_params):
                    grad_sum[k] += grads_j[k]

            total_loss += batch_loss

            # Global gradient-norm clipping (prevents exploding gradients in BPTT)
            total_norm = float(np.sqrt(sum(np.sum(g ** 2) for g in grad_sum)))
            if total_norm > clip_val:
                scale    = clip_val / (total_norm + 1e-8)
                grad_sum = [g * scale for g in grad_sum]

            self._adam_update(grad_sum)

        return total_loss / n

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 30,
            batch_size: int = 32, verbose: bool = True,
            X_val=None, y_val=None) -> Dict:
        history = {"train_loss": [], "val_loss": []}
        for epoch in range(epochs):
            loss = self.train_epoch(X, y, batch_size)
            history["train_loss"].append(loss)
            val_str = ""
            if X_val is not None:
                val_preds = self.predict(X_val)
                val_loss = float(np.mean((val_preds - y_val)**2))
                history["val_loss"].append(val_loss)
                val_str = f" | val_loss={val_loss:.4f}"
            if verbose:
                print(f"    Epoch {epoch+1:3d}/{epochs} | train_loss={loss:.4f}{val_str}")
        return history

    def score_r2(self, X, y):
        yhat = self.predict(X)
        ss_res = np.sum((y - yhat)**2)
        ss_tot = np.sum((y - y.mean())**2)
        return 1.0 - ss_res / (ss_tot + 1e-10)


# ─────────────────────────────────────────────
# 3. GraphSAGE (NumPy, inductive, mean aggregation)
# ─────────────────────────────────────────────

class GraphSAGELayer:
    """Single GraphSAGE layer with mean aggregation"""

    def __init__(self, in_features: int, out_features: int, rng):
        k = np.sqrt(2.0 / (in_features + out_features))
        self.W_self = rng.normal(0, k, (out_features, in_features))
        self.W_neigh = rng.normal(0, k, (out_features, in_features))
        self.b = np.zeros(out_features)

    def forward(self, node_features: np.ndarray,
                adj_matrix: np.ndarray) -> np.ndarray:
        """
        node_features: (N, in_features)
        adj_matrix: (N, N) - normalized adjacency
        Returns: (N, out_features)
        """
        # Aggregate neighbors (mean)
        neighbor_agg = adj_matrix @ node_features  # (N, in_features)

        h_self  = node_features @ self.W_self.T
        h_neigh = neighbor_agg  @ self.W_neigh.T
        h = h_self + h_neigh + self.b

        # ReLU
        return np.maximum(0, h)

    def parameters(self):
        return [self.W_self, self.W_neigh, self.b]


class GraphSAGEModel:
    """
    2-layer inductive GraphSAGE for node-level CPU prediction.
    Architecture: Input → SAGE(64) → SAGE(32) → Linear(1)
    """

    def __init__(self, in_features: int, hidden_sizes=(64, 32),
                 rng_seed: int = 42):
        rng = np.random.default_rng(rng_seed)
        self.layers = []
        prev = in_features
        for h in hidden_sizes:
            self.layers.append(GraphSAGELayer(prev, h, rng))
            prev = h
        # Output head
        k = np.sqrt(1.0 / prev)
        self.W_out = rng.normal(0, k, (1, prev))
        self.b_out = np.zeros(1)
        self._adam_t = 0
        self._lr = 0.002
        self._beta1, self._beta2, self._eps = 0.9, 0.999, 1e-8
        all_params = self._get_params()
        self._m = [np.zeros_like(p) for p in all_params]
        self._v = [np.zeros_like(p) for p in all_params]

    def _get_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        params += [self.W_out, self.b_out]
        return params

    def _build_adjacency(self, n_nodes: int) -> np.ndarray:
        """
        Build normalized adjacency for a fully-connected topology.
        In production, replace with real service dependency graph.
        """
        # Connect all nodes (microservice mesh) with self-loops
        A = np.ones((n_nodes, n_nodes))
        np.fill_diagonal(A, 2.0)   # self-weight = 2x
        # Row normalize
        row_sums = A.sum(axis=1, keepdims=True)
        return A / (row_sums + 1e-10)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, in_features) - one row per server
        Returns: (N,) predicted CPU for each node
        """
        n = X.shape[0]
        adj = self._build_adjacency(n)
        h = X.copy()
        for layer in self.layers:
            h = layer.forward(h, adj)
        # L2-normalize
        norms = np.linalg.norm(h, axis=1, keepdims=True)
        h = h / (norms + 1e-10)
        out = h @ self.W_out.T + self.b_out  # (N, 1)
        return out.squeeze(1)

    def fit(self, X_snapshots: List[np.ndarray], y_snapshots: List[np.ndarray],
            epochs: int = 50, verbose: bool = True) -> Dict:
        """
        X_snapshots: list of (N, F) arrays (one per timestamp)
        y_snapshots: list of (N,) arrays (CPU at next timestep)
        """
        history = {"loss": []}
        n_samples = len(X_snapshots)
        eps_fd = 1e-4
        params = self._get_params()

        for epoch in range(epochs):
            epoch_loss = 0.0
            idxs = np.random.permutation(n_samples)

            for idx in idxs:
                X = X_snapshots[idx]
                y = y_snapshots[idx]
                preds = self.forward(X)
                loss = float(np.mean((preds - y)**2))
                epoch_loss += loss

                # Approximate gradient for output layer (exact)
                grads = [np.zeros_like(p) for p in params]
                dy = 2.0 * (preds - y) / len(y)
                h = X.copy()
                adj = self._build_adjacency(X.shape[0])
                for layer in self.layers:
                    h = layer.forward(h, adj)
                norms = np.linalg.norm(h, axis=1, keepdims=True)
                h_norm = h / (norms + 1e-10)
                grads[-2] = (dy @ h_norm)[np.newaxis, :]   # (1, hidden) — correct dL/dW_out
                grads[-1] = np.array([dy.sum()])

                # FD for SAGE weights (sampled)
                param_count = len(params)
                for pidx in range(0, param_count - 2, 3):  # W_self only
                    p = params[pidx]
                    g = np.zeros_like(p)
                    flat_idxs = np.random.choice(p.size, size=min(8, p.size), replace=False)
                    for fi in flat_idxs:
                        nd_idx = np.unravel_index(fi, p.shape)
                        p[nd_idx] += eps_fd
                        lp = float(np.mean((self.forward(X) - y)**2))
                        p[nd_idx] -= 2*eps_fd
                        lm = float(np.mean((self.forward(X) - y)**2))
                        p[nd_idx] += eps_fd
                        g[nd_idx] = (lp - lm) / (2 * eps_fd)
                    grads[pidx] = g

                # Adam update → compute new values then copy in-place
                self._adam_t += 1
                t_adam = self._adam_t
                new_params = []
                for i, (p, g) in enumerate(zip(params, grads)):
                    self._m[i] = self._beta1 * self._m[i] + (1-self._beta1) * g
                    self._v[i] = self._beta2 * self._v[i] + (1-self._beta2) * g**2
                    m_hat = self._m[i] / (1 - self._beta1**t_adam)
                    v_hat = self._v[i] / (1 - self._beta2**t_adam)
                    new_params.append(p - self._lr * m_hat / (np.sqrt(v_hat) + self._eps))

                # Copy new values into existing arrays (preserve references)
                param_idx = 0
                for layer in self.layers:
                    layer.W_self[:]  = new_params[param_idx]; param_idx += 1
                    layer.W_neigh[:] = new_params[param_idx]; param_idx += 1
                    layer.b[:]       = new_params[param_idx]; param_idx += 1
                self.W_out[:] = new_params[param_idx]; param_idx += 1
                self.b_out[:] = new_params[param_idx]
                params = self._get_params()

            avg_loss = epoch_loss / n_samples
            history["loss"].append(avg_loss)
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"    Epoch {epoch+1:3d}/{epochs} | loss={avg_loss:.4f}")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

# ─────────────────────────────────────────────
# 4. WORKLOAD CLASSIFIER (OvR RF)
# ─────────────────────────────────────────────

WORKLOAD_CLASSES = [
    "gaming", "streaming", "business", "social",
    "data_batch", "ml_train", "cdn_static", "api_gateway",
]

class WorkloadClassifier:
    """
    One-vs-Rest Random Forest workload sınıflandırıcısı.
    Her iş yükü sınıfı için ayrı bir ikili RF eğitilir.
    predict() → en yüksek skorlu sınıfın string etiketini döner.
    predict_proba() → (N, 5) olasılık matrisini döner.
    """

    def __init__(self, n_estimators: int = 30, max_depth: int = 8,
                 rng_seed: int = 42):
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.rng_seed     = rng_seed
        self.classes_     = WORKLOAD_CLASSES
        self.classifiers_: List[RandomForestRegressor] = []

    def fit(self, X: np.ndarray, y_int: np.ndarray):
        """
        X       : (N, features)
        y_int   : (N,) integer labels  0=gaming,1=streaming,2=business,3=batch,4=social
        """
        self.classifiers_ = []
        for cls_idx in range(len(self.classes_)):
            y_binary = (y_int == cls_idx).astype(np.float32)
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                rng_seed=self.rng_seed + cls_idx,
            )
            rf.fit(X, y_binary)
            self.classifiers_.append(rf)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns (N, n_classes) — raw scores, not calibrated probabilities."""
        scores = np.stack([clf.predict(X) for clf in self.classifiers_], axis=1)
        # Softmax-like normalization so scores sum to 1
        scores = np.clip(scores, 0.0, None)
        row_sum = scores.sum(axis=1, keepdims=True)
        return scores / (row_sum + 1e-8)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns (N,) integer class indices."""
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_label(self, X: np.ndarray) -> List[str]:
        """Returns (N,) list of workload type strings."""
        return [self.classes_[i] for i in self.predict(X)]

    def score_accuracy(self, X: np.ndarray, y_int: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y_int))


# ─────────────────────────────────────────────
# 5. MULTI-OUTPUT LSTM
# ─────────────────────────────────────────────

class MultiOutputLSTM:
    """
    LSTMModel wrapper ile iki çıkış kafası:
      - regression_head : CPU % tahmini (MSE loss)
      - cls_head        : workload type sınıflandırması (cross-entropy loss)

    İki kafa paylaşılan LSTM gövdesini kullanır; eğitim aynı anda yapılır.
    alpha = regression kaybına verilen ağırlık (1-alpha = sınıflandırma)
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 n_classes: int = 8, alpha: float = 0.7, rng_seed: int = 42):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.n_classes   = n_classes
        self.alpha       = alpha          # CPU regression ağırlığı
        self.classes_    = WORKLOAD_CLASSES

        rng = np.random.default_rng(rng_seed)
        self.lstm = LSTMCell(input_size, hidden_size, rng)

        # Regression head: hidden → 1
        k = 1.0 / np.sqrt(hidden_size)
        self.Wy_reg = rng.uniform(-k, k, (1, hidden_size))
        self.by_reg = np.zeros(1)

        # Classification head: hidden → n_classes
        self.Wy_cls = rng.uniform(-k, k, (n_classes, hidden_size))
        self.by_cls = np.zeros(n_classes)

        self._init_adam()

    def _init_adam(self):
        self._t = 0
        self._lr = 0.001
        self._b1, self._b2, self._eps = 0.9, 0.999, 1e-8
        self._params = self._collect_params()
        self._m = [np.zeros_like(p) for p in self._params]
        self._v = [np.zeros_like(p) for p in self._params]

    def _collect_params(self) -> List[np.ndarray]:
        l = self.lstm
        return [l.Wf, l.Uf, l.bf, l.Wi, l.Ui, l.bi,
                l.Wo, l.Uo, l.bo, l.Wc, l.Uc, l.bc,
                self.Wy_reg, self.by_reg,
                self.Wy_cls, self.by_cls]

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / (e.sum() + 1e-10)

    def _forward_seq(self, seq: np.ndarray):
        """seq: (T, input_size) → returns (h_T, reg_out, cls_probs)"""
        T = seq.shape[0]
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        for t in range(T):
            h, c, _ = self.lstm.forward(seq[t], h, c)
        reg_out   = float(self.Wy_reg @ h + self.by_reg)
        cls_logit = self.Wy_cls @ h + self.by_cls
        cls_probs = self._softmax(cls_logit)
        return h, reg_out, cls_probs

    def _adam_step(self, grads: List[np.ndarray]):
        self._t += 1
        new_vals = []
        for i, (p, g) in enumerate(zip(self._params, grads)):
            self._m[i] = self._b1 * self._m[i] + (1 - self._b1) * g
            self._v[i] = self._b2 * self._v[i] + (1 - self._b2) * g**2
            m_hat = self._m[i] / (1 - self._b1 ** self._t)
            v_hat = self._v[i] / (1 - self._b2 ** self._t)
            new_vals.append(p - self._lr * m_hat / (np.sqrt(v_hat) + self._eps))
        # Write back
        l = self.lstm
        (l.Wf[:], l.Uf[:], l.bf[:],
         l.Wi[:], l.Ui[:], l.bi[:],
         l.Wo[:], l.Uo[:], l.bo[:],
         l.Wc[:], l.Uc[:], l.bc[:]) = new_vals[:12]
        self.Wy_reg[:], self.by_reg[:] = new_vals[12], new_vals[13]
        self.Wy_cls[:], self.by_cls[:] = new_vals[14], new_vals[15]
        self._params = self._collect_params()

    def train_epoch(self, X: np.ndarray, y_cpu: np.ndarray,
                    y_wl: np.ndarray, batch_size: int = 256,
                    clip_val: float = 1.0) -> Tuple[float, float]:
        """Returns (avg_reg_loss, avg_cls_loss)"""
        N = len(X)
        idx = np.random.permutation(N)
        total_reg, total_cls = 0.0, 0.0
        n_params = len(self._params)

        for start in range(0, N, batch_size):
            batch_idx = idx[start:start + batch_size]
            grad_sum  = [np.zeros_like(p) for p in self._params]
            batch_reg = 0.0
            batch_cls = 0.0

            for bi in batch_idx:
                seq    = X[bi]           # (T, feat)
                target_cpu = float(y_cpu[bi])
                target_wl  = int(y_wl[bi])

                h, reg_out, cls_probs = self._forward_seq(seq)

                # ── Regression loss & grad ──────────────────────────────
                reg_err  = reg_out - target_cpu
                reg_loss = reg_err ** 2
                batch_reg += reg_loss

                # ── Classification loss (cross-entropy) & grad ──────────
                p_true   = float(np.clip(cls_probs[target_wl], 1e-10, 1.0))
                cls_loss = -np.log(p_true)
                batch_cls += cls_loss

                # ── Backprop through heads (analytical) ────────────────
                # dL/dh from regression head
                dh_reg = self.alpha * 2.0 * reg_err * self.Wy_reg[0]   # (hidden,)

                # dL/dh from cls head
                d_logit        = cls_probs.copy()
                d_logit[target_wl] -= 1.0                  # softmax + CE gradient
                dh_cls = (1.0 - self.alpha) * (self.Wy_cls.T @ d_logit)

                dh = dh_reg + dh_cls   # combined gradient into LSTM hidden

                # BPTT for single step (approximation: only last hidden state)
                eps_fd = 1e-4
                # Analytical head gradients
                dWy_reg = (self.alpha * 2.0 * reg_err * h).reshape(1, -1)
                dby_reg = np.array([self.alpha * 2.0 * reg_err])

                dWy_cls = (1 - self.alpha) * np.outer(d_logit, h)
                dby_cls = (1 - self.alpha) * d_logit

                # FD for LSTM params (sampled — same as original LSTMModel)
                lstm_param_list = self._collect_params()[:12]
                grads_lstm = [np.zeros_like(p) for p in lstm_param_list]
                combined_loss = self.alpha * reg_loss + (1.0 - self.alpha) * cls_loss

                for pidx, p in enumerate(lstm_param_list):
                    flat_idxs = np.random.choice(p.size, size=min(16, p.size), replace=False)
                    for fi in flat_idxs:
                        nd = np.unravel_index(fi, p.shape)
                        p[nd] += eps_fd
                        _, r2, cp2 = self._forward_seq(seq)
                        loss_p = (self.alpha * (r2 - target_cpu)**2
                                  - (1-self.alpha) * np.log(max(cp2[target_wl], 1e-10)))
                        p[nd] -= 2 * eps_fd
                        _, r3, cp3 = self._forward_seq(seq)
                        loss_m = (self.alpha * (r3 - target_cpu)**2
                                  - (1-self.alpha) * np.log(max(cp3[target_wl], 1e-10)))
                        p[nd] += eps_fd
                        grads_lstm[pidx][nd] = (loss_p - loss_m) / (2 * eps_fd)

                all_grads = (grads_lstm +
                             [dWy_reg, dby_reg, dWy_cls, dby_cls])
                for k in range(n_params):
                    grad_sum[k] += all_grads[k]

            total_reg += batch_reg
            total_cls += batch_cls

            # Gradient clipping
            norm = float(np.sqrt(sum(np.sum(g**2) for g in grad_sum)))
            if norm > clip_val:
                grad_sum = [g * (clip_val / (norm + 1e-8)) for g in grad_sum]

            self._adam_step(grad_sum)

        n = N or 1
        return total_reg / n, total_cls / n

    def fit(self, X: np.ndarray, y_cpu: np.ndarray, y_wl: np.ndarray,
            epochs: int = 20, batch_size: int = 256, verbose: bool = True,
            X_val=None, y_cpu_val=None, y_wl_val=None) -> Dict:
        history = {"reg_loss": [], "cls_loss": [], "val_reg": [], "val_cls": []}
        for epoch in range(epochs):
            rl, cl = self.train_epoch(X, y_cpu, y_wl, batch_size)
            history["reg_loss"].append(rl)
            history["cls_loss"].append(cl)
            val_str = ""
            if X_val is not None:
                vr, vc = self._val_loss(X_val, y_cpu_val, y_wl_val)
                history["val_reg"].append(vr)
                history["val_cls"].append(vc)
                val_str = f" | val_reg={vr:.4f}  val_cls={vc:.4f}"
            if verbose:
                print(f"    Epoch {epoch+1:3d}/{epochs}"
                      f" | reg={rl:.4f}  cls={cl:.4f}{val_str}")
        return history

    def _val_loss(self, X, y_cpu, y_wl) -> Tuple[float, float]:
        reg_l, cls_l = 0.0, 0.0
        for i in range(len(X)):
            _, r, cp = self._forward_seq(X[i])
            reg_l += (r - float(y_cpu[i]))**2
            cls_l += -np.log(max(cp[int(y_wl[i])], 1e-10))
        n = len(X) or 1
        return reg_l / n, cls_l / n

    def predict_cpu(self, X: np.ndarray) -> np.ndarray:
        """Returns (N,) CPU % predictions."""
        return np.array([self._forward_seq(X[i])[1] for i in range(len(X))],
                        dtype=np.float32)

    def predict_workload(self, X: np.ndarray) -> np.ndarray:
        """Returns (N,) integer class indices."""
        probs = np.array([self._forward_seq(X[i])[2] for i in range(len(X))])
        return np.argmax(probs, axis=1)

    def predict_workload_label(self, X: np.ndarray) -> List[str]:
        return [self.classes_[i] for i in self.predict_workload(X)]

    # Backward compat — döner cpu tahminini
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_cpu(X)