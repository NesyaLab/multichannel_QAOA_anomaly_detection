# AD_probabilistic.py
"""
ML-guided center selection USING ONLY QAOA-DERIVED FEATURES.
Detection remains hard set-cover.

"""








from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Union, Iterable, Sequence
import numpy as np







#################################################################################################################################################################################################################
######################################################################################  Utilities functions #####################################################################################################
#################################################################################################################################################################################################################



def _ensure_2d_states(top_states: List[List[int]]) -> np.ndarray:
    S = np.asarray(top_states, dtype=int)
    if S.ndim == 1:
        S = S[None, :]
    return S

def _normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    mn, mx = float(np.min(v)), float(np.max(v))
    if mx - mn < eps:
        return np.zeros_like(v)
    return (v - mn) / (mx - mn + eps)

def _euclidean(p, q):
    # p, q are (t, v)
    return float(np.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2))

def _min_dist_to_baseline(points: List[Tuple[int, float]],
                          baseline_idx: List[int],
                          i: int) -> float:
    if not baseline_idx:
        return float("inf")
    pi = points[i]
    return min(_euclidean(pi, points[j]) for j in baseline_idx)

def _is_singleton_for_index(top_states: np.ndarray, i: int) -> bool:
    ones_per_state = np.sum(top_states, axis=1)
    return bool(np.any((ones_per_state == 1) & (top_states[:, i] == 1)))


def topk_votes(top_states: List[List[int]],
               topk_probs: Optional[List[Tuple[str, float]]] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        votes: int array (n,), v_i = # of times index i is 1 across top-k states
        weights: float array (n,), probability-weighted frequency (in [0,1])
    """
    S = _ensure_2d_states(top_states)
    k, n = S.shape
    votes = np.sum(S, axis=0).astype(int)

    if topk_probs and len(topk_probs) == k:
        p = np.asarray([float(p) for _, p in topk_probs], dtype=float)
        p = p / (p.sum() + 1e-12)
        weights = (S * p[:, None]).sum(axis=0)
    else:
        weights = votes / max(1, k)

    return votes, weights




def build_features_from_topk(
    X: List[Tuple[int, float]],
    top_states: List[List[int]],
    topk_probs: Optional[List[Tuple[str, float]]] = None,
    include_cooccurrence: bool = True,
) -> Tuple[np.ndarray, List[int], np.ndarray, np.ndarray]:
    """
    Create QAOA-only features per index i:
      - votes v_i
      - prob-weighted frequency w_i (falls back to v_i/k)
      - distance to baseline (min distance to any top-1 center)
      - (optional) co-occurrence with baseline centers across top-k
    Returns:
      F: (n, d) features
      C0_idx: indices in the top-1 baseline mask
      votes: (n,)
      weights: (n,)
    """
    S = _ensure_2d_states(top_states)
    k, n = S.shape
    votes, weights = topk_votes(top_states, topk_probs)

    C0_idx = [i for i, b in enumerate(S[0]) if b == 1]

    d = np.array([_min_dist_to_baseline(X, C0_idx, i) for i in range(n)], dtype=float)

    if include_cooccurrence and C0_idx:
        co = []
        C0_mask = (S[:, C0_idx].sum(axis=1) > 0).astype(int)  
        for i in range(n):
            co.append(int(np.sum(S[:, i] * C0_mask)))
        co = np.asarray(co, dtype=float) / float(k)
    else:
        co = np.zeros(n, dtype=float)

    F = np.column_stack([
        _normalize(votes),
        _normalize(weights),
        1.0 - _normalize(d),     
        _normalize(co),
    ])
    return F, C0_idx, votes, weights



def _state_to_bits(s, n: int) -> str:
    """
    Converte s in una bitstring di lunghezza n.
    s può essere: str, list/tuple/np.ndarray di 0/1, oppure int (bitmask).
    Assunzione: il bit più a sinistra (MSB) mappa l'indice 0.
    """
    try:
        import numpy as np
    except Exception:
        np = None

    if isinstance(s, (list, tuple)) and len(s) > 0 and not isinstance(s[0], (int, str)):
        s = s[0]

    if isinstance(s, str):
        if len(s) < n:
            s = s.rjust(n, '0')
        elif len(s) > n:
            s = s[-n:]
        return s

    if (np is not None and isinstance(s, np.ndarray)) or isinstance(s, (list, tuple)):
        return ''.join('1' if int(x) else '0' for x in (s.tolist() if (np is not None and hasattr(s, 'tolist')) else s))

    if isinstance(s, (int,)) or (np is not None and isinstance(s, np.integer)):
        b = format(int(s), f'0{n}b')
        return b[-n:]  

    s = str(s)
    if len(s) < n:
        s = s.rjust(n, '0')
    elif len(s) > n:
        s = s[-n:]
    return s



#################################################################################################################################################################################################################
#################################################################################################################################################################################################################








# -----------------------------
# Pseudo-labels (weak supervision)
# -----------------------------


def make_pseudolabels(
    votes: np.ndarray,
    weights: np.ndarray,
    C0_idx: List[int],
    v_strong: int = 2,         # >=2 votes when k=5
    w_strong: float = 0.35,    # >=0.35 weighted frequency when k=5
    use_weights: bool = True,
) -> np.ndarray:
    """
    Returns y in {0,1,-1} (1=positive center, 0=negative, -1=unlabeled)
    """
    n = len(votes)
    y = np.full(n, -1, dtype=int)

    # positives
    y[C0_idx] = 1
    strong_in = (votes >= int(v_strong))
    if use_weights:
        strong_in = strong_in | (weights >= float(w_strong))
    y[strong_in] = 1

    # negatives
    y[votes == 0] = 0

    return y













# -----------------------------
# Regressor (logistic)
# -----------------------------





def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_fit(
    X: np.ndarray, y: np.ndarray,
    l2: float = 1e-3, lr: float = 0.1, epochs: int = 200
) -> Tuple[np.ndarray, float]:
    """
    Simple logistic regression (binary) with L2. y ∈ {0,1}.
    Returns weights w and bias b.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0

    for _ in range(int(epochs)):
        z = X @ w + b
        p = _sigmoid(z)
        # gradients
        grad_w = X.T @ (p - y) / n + l2 * w
        grad_b = float(np.mean(p - y))
        # update
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b

def logistic_predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return _sigmoid(X @ w + b)



def select_centers_mask_from_topk(
    X: List[Tuple[int, float]],
    top_states: List[List[int]],
    topk_probs: Optional[List[Tuple[str, float]]] = None,
    tau: float = 0.75,        # decision threshold on learned probability
    v_strong: int = 2,
    w_strong: float = 0.35,
    kappa_r: float = 1.2,     # singleton guard scale vs batch radius (baseline estimation)
    ad_qaoa_obj=None,         # optional: to estimate rB from baseline via its radius_adj
) -> List[int]:
    """
    Build final center mask 'c' (list of 0/1) for the batch.
    - Superset guarantee: all top-1 centers are kept.
    - Add more centers via ML selector on QAOA-only features.
    - Singleton safeguard: if an index appears only in a one-hot top-k state, require proximity to C0.
    """
    S = _ensure_2d_states(top_states)
    k, n = S.shape

    # Features
    F, C0_idx, votes, weights = build_features_from_topk(X, top_states, topk_probs)
    # Pseudo-labels (weak supervision)
    y = make_pseudolabels(votes, weights, C0_idx, v_strong=v_strong, w_strong=w_strong, use_weights=True)

    # Train on labeled only
    labeled = y >= 0
    X_tr, y_tr = F[labeled], y[labeled]
    if X_tr.size == 0 or np.unique(y_tr).size < 2:
        # fallback: only baseline
        return [1 if i in C0_idx else 0 for i in range(n)]

    w, b = logistic_fit(X_tr, y_tr.astype(float), l2=1e-3, lr=0.1, epochs=200)
    P = logistic_predict_proba(F, w, b)

    # Estimate batch radius from baseline C0 (optional, if ad_qaoa provided)
    rB = None
    if ad_qaoa_obj is not None and len(C0_idx) > 0:
        # make a baseline mask to reuse existing API associate_centers_with_radius
        base_mask = [1 if i in C0_idx else 0 for i in range(n)]
        try:
            centers_r = ad_qaoa_obj.associate_centers_with_radius(state=base_mask)
            if centers_r:
                rB = float(centers_r[0][1])
        except Exception:
            rB = None

    # Final decision
    c = np.zeros(n, dtype=int)
    c[C0_idx] = 1  # superset guarantee

    singleton_flags = np.array([_is_singleton_for_index(S, i) for i in range(n)], dtype=bool)
    dist_to_C0 = np.array([_min_dist_to_baseline(X, C0_idx, i) for i in range(n)], dtype=float)

    for i in range(n):
        if c[i] == 1:
            continue
        if float(P[i]) >= float(tau):
            if singleton_flags[i] and rB is not None:
                if dist_to_C0[i] > float(kappa_r) * float(rB):
                    continue  
            c[i] = 1

    return c.tolist()



















# -----------------------------
# Batch-level wrapper: end-to-end
# -----------------------------

def _fmt_bits(arr) -> str:
    return "".join(str(int(x)) for x in arr)

def _print_topk_and_mask(batch_idx: int, top_states, mask):
    print(f"[ProbCover-ML] Top-{len(top_states)} states (batch {batch_idx}):")
    for s in top_states:
        print("   ", _fmt_bits(s))
    print(f"[ProbCover-ML] Combined mask (batch {batch_idx}):")
    print("   ", _fmt_bits(mask))


def execute_qaoa_probcover_on_batches_ml(
    batches: List[List[Tuple[int, float]]],
    alpha_mean: float,
    beta_mean: float,
    model_name: str = "cubic",
    model_params: Optional[Dict] = None,
    top_k: int = 5,
    tau: float = 0.75,
    v_strong: int = 2,
    w_strong: float = 0.35,
    kappa_r: float = 1.2,
    verbose: bool = True,
    show_states: bool = True,
    top_sel: Optional[Union[int, List[int], Tuple[int, ...], str]] = None,
    anomaly_ts: Optional[Iterable[int]] = None,
):
    """
    Probabilistic covering con selettore ML e **L globale fisso**.
    NEW: se anomaly_ts è fornito, la mask viene costruita direttamente dalle label:
         - un indice (timestamp) è 'buono' se NON è in anomaly_ts e
           (occ >= v_strong) oppure (freq >= tau) sulle bitstring selezionate.
    """
    import types
    import numpy as np
    from AD_QAOA_ext import AD_QAOA
    from functions.AD_utilities import build_global_model_cache

    if model_params is None:
        model_params = {}

    anomaly_set = set(anomaly_ts) if anomaly_ts is not None else None
    all_centers_with_radii = []

    # Utility: parsing top_sel (solo per sicurezza, per evitare errori di tipologia o valore)
  
    def _parse_top_sel_indices(n_states: int,
                               top_k_default: int,
                               sel: Optional[Union[int, List[int], Tuple[int, ...], str]]) -> List[int]:
        if n_states <= 0:
            return []
        if sel is None or isinstance(sel, int):
            k = int(sel) if isinstance(sel, int) else int(top_k_default)
            k = max(0, k)
            return list(range(min(k, n_states)))
        if isinstance(sel, str):
            chosen_1b = []
            parts = [p.strip() for p in sel.split(',')] if sel else []
            for p in parts:
                if not p:
                    continue
                if ':' in p:
                    a, b = p.split(':', 1)
                    a, b = int(a), int(b)
                    if a <= b:
                        chosen_1b.extend(range(a, b + 1))
                    else:
                        chosen_1b.extend(range(b, a + 1))
                else:
                    chosen_1b.append(int(p))
            chosen_0b = sorted({i - 1 for i in chosen_1b if 1 <= i <= n_states})
            return chosen_0b
        if isinstance(sel, (list, tuple)):
            chosen_0b = sorted({int(i) for i in sel if 0 <= int(i) < n_states})
            return chosen_0b
        return list(range(min(int(top_k_default), n_states)))

 
    X_full = [pt for batch in batches for pt in batch]
    try:
        L_full, L_by_ts, _ = build_global_model_cache(X_full, model_name, model_params)
        if verbose:
            print(f"[GLOBAL MODEL] built once: model={model_name}, params={model_params}, len(L_full)={len(L_full)}")
    except Exception as e:
        L_full, L_by_ts = None, None
        if verbose:
            print(f"[GLOBAL MODEL] WARNING: failed to build global cache ({e}). Falling back to per-batch M).")

    for i, batch in enumerate(batches, 1):
        if verbose:
            print(f"\n[ProbCover-ML] Batch {i}/{len(batches)}")

        ad = AD_QAOA(
            X=batch,
            alpha=alpha_mean, beta=beta_mean,
            model_name=model_name, model_params=(model_params or {}),
            radius_adjustment=True,
            top_n_samples=top_k,
        )

        # Override M (per l'aggiustamento di L globale)
        M_override = None
        if L_by_ts is not None:
            try:
                ts_b = [t for (t, _) in batch]
                L_b = np.array([L_by_ts[t] for t in ts_b], dtype=float)
                Q_b = ad.off_diag_M(batch)
                M_override = np.diag(L_b) + Q_b
            except Exception as e:
                if verbose:
                    print(f"[GLOBAL MODEL] WARNING: cannot assemble M_override for batch {i} ({e}). Using default M.")
                M_override = None

        if M_override is not None:
            def _matrix_M_override(self):
                return M_override
            ad.matrix_M = types.MethodType(_matrix_M_override, ad)

        try:
            top_states, _vars, topk_probs = ad.solve_qubo_extended()
        except AttributeError:
            top_states = []
            st, _ = ad.solve_qubo()
            if st is not None:
                top_states.append(st)
            topk_probs = None

        sel_indices = _parse_top_sel_indices(len(top_states), top_k, top_sel)
        if not sel_indices:
            if verbose:
                print("[ProbCover-ML] WARNING: no selected states; skipping batch.")
            continue

        top_states_sel = [top_states[j] for j in sel_indices]
        topk_probs_sel = None
        if topk_probs is not None:
            try:
                topk_probs_sel = [topk_probs[j] for j in sel_indices]
            except Exception:
                topk_probs_sel = None

        if anomaly_set is None:
            mask = select_centers_mask_from_topk(
                X=batch,
                top_states=top_states_sel,
                topk_probs=topk_probs_sel,
                tau=tau,
                v_strong=v_strong,
                w_strong=w_strong,
                kappa_r=kappa_r,
                ad_qaoa_obj=ad,
            )
        else:
            # Caso supervisionato
            B = len(top_states_sel)
            occ = np.zeros(len(batch), dtype=int)
            for s in top_states_sel:
                state_bits = _state_to_bits(s, n=len(batch))
                for j, ch in enumerate(state_bits):
                    if ch == '1':
                        occ[j] += 1
            freq = occ / max(1, B)
            mask = np.zeros(len(batch), dtype=float)
            for j, (ts, _val) in enumerate(batch):
                if ts in anomaly_set:
                    mask[j] = 0.0
                elif (occ[j] >= int(v_strong)) or (freq[j] >= float(tau)):
                    mask[j] = 1.0
            if show_states:
                print(f"[Supervised] Batch {i}: normals selected = {int(mask.sum())} / {len(mask)}")

        if len(mask) != len(batch):
            raise ValueError(f"[ProbCover-ML] Mask length {len(mask)} != batch length {len(batch)}")

        if show_states:
            _print_topk_and_mask(i, top_states_sel, mask)

        selected_idx = [j for j, v in enumerate(mask) if v > 0.5]
        centers = [batch[j] for j in selected_idx]
        if not centers:
            C0_idx = [j for j, b in enumerate(_state_to_bits(top_states_sel[0], n=len(batch))) if b == '1']
            centers = [batch[j] for j in C0_idx]
            if verbose:
                print(f"[ProbCover-ML] Empty mask -> fallback to first selected state (k={len(C0_idx)})")

        radius = ad.radius_adj(centers)
        centers_with_r = [(c, radius) for c in centers]

        if verbose:
            print(f"[ProbCover-ML] mask len={len(mask)} | batch len={len(batch)} | #centers={len(centers_with_r)}")
            print(f"[ProbCover-ML] Selected centers: {centers_with_r}")

        all_centers_with_radii.extend(centers_with_r)

    unique = []
    seen_ts = set()
    for (center, radius) in all_centers_with_radii:
        ts = center[0]
        if ts not in seen_ts:
            unique.append((center, radius))
            seen_ts.add(ts)

    if verbose:
        print("\n[ProbCover-ML] Merged centers across batches:")
        for j, (c, r) in enumerate(unique):
            print(f"  {j}: center={c}, radius={r}")

    return unique























#################################################################################################################################################################################################################################################################################################
##################################################################################### Plotting utilities for the new article (scale adjustmens and so on) #######################################################################################################################################
#################################################################################################################################################################################################################################################################################################




def to_ellipses(
    centers_with_radii: Sequence[Tuple[Tuple[float, float], float]],
    eta: float = 0.75,          # scaling verticale (ry = eta * r se r > soglia)
    xi: float = 1.0,            # scaling orizzontale (rx = xi * r, SEMPRE applicato)
    only_if_r_gt: float = 1.0,  # soglia per applicare schiacciamento verticale
) -> List[Tuple[Tuple[float, float], float, float]]:
    """
    Converte [(center=(t,v), r)] -> [(center=(t,v), rx, ry)] con scaling anisotropo.
    - rx = xi * r         (sempre applicato, allarga/accorcia sull'asse tempo)
    - ry = eta * r se r > only_if_r_gt, altrimenti ry = r (cerchio in verticale)
    """
    out: List[Tuple[Tuple[float, float], float, float]] = []
    xi = float(xi)
    eta = float(eta)
    thr = None if only_if_r_gt is None else float(only_if_r_gt)

    for (center, r) in centers_with_radii:
        r = float(r)
        rx = xi * r  # sempre applicato
        if thr is not None and r <= thr:
            ry = r
        else:
            ry = eta * r
        out.append((center, rx, ry))
    return out






def apply_ellipses_to_new_dataset(
    data: Sequence[Tuple[float, float]],
    ellipses: Sequence[Tuple[Tuple[float, float], float, float]],
    tol: float = 0.0
) -> List[bool]:
    """
    Restituisce una lista di boolean (True = coperto, quindi 'normale'; False = anomalo)
    usando ellissi: ((t-cx)/rx)^2 + ((v-cy)/ry)^2 <= (1 + tol).
    
    Args:
        data: lista di punti (timestamp, valore).
        ellipses: lista di ellissi ((cx, cy), rx, ry).
        tol: tolleranza moltiplicativa (>0 allarga le ellissi).
             Esempio: tol=0.1 permette un 10% in più di raggio.
    """
    covered = []
    for (t, v) in data:
        inside = False
        for ((cx, cy), rx, ry) in ellipses:
            if rx <= 0 or ry <= 0:
                continue
            dt = (t - cx) / rx
            dv = (v - cy) / ry
            if (dt*dt + dv*dv) <= (1.0 + tol):
                inside = True
                break
        covered.append(inside)

    anomalies = sum(1 for c in covered if not c)
    print(f"Anomalies detected: {anomalies} / {len(data)}")
    return covered





def plot_elliptical_covering(ax, ellipses, facecolor=(0.2, 1.0, 0.2, 0.25), edgecolor=(0.2, 0.7, 0.2, 0.6)):
    """
    Disegna ellissi su un axes Matplotlib (ax).
    """
    from matplotlib.patches import Ellipse
    for ((cx, cy), rx, ry) in ellipses:
        e = Ellipse((cx, cy), width=2*rx, height=2*ry, facecolor=facecolor, edgecolor=edgecolor, lw=1)
        ax.add_patch(e)















########################################## MULTI ###########################################################################################################################################################################################################################################################################################################################
from typing import Dict, List, Tuple, Sequence

def to_ellipses_mv_by_channel(
    centers_with_radii_by_channel: Dict[int, Sequence[Tuple[Tuple[float, float], float]]],
    eta: float = 0.75,
    xi: float = 1.0,
    only_if_r_gt: float = 1.0,
) -> Dict[int, List[Tuple[Tuple[float, float], float, float]]]:
    """
    MV (per-canale): converte
      dict[c] -> [((t,y), r), ...]
    in
      dict[c] -> [((t,y), rx, ry), ...]

    Stessa logica della monovariata:
      rx = xi * r (sempre)
      ry = r se r <= thr, altrimenti ry = eta * r
    """
    out: Dict[int, List[Tuple[Tuple[float, float], float, float]]] = {}
    xi = float(xi)
    eta = float(eta)
    thr = None if only_if_r_gt is None else float(only_if_r_gt)

    for c, lst in centers_with_radii_by_channel.items():
        ell_c: List[Tuple[Tuple[float, float], float, float]] = []
        for (center, r) in lst:
            r = float(r)
            rx = xi * r
            if thr is not None and r <= thr:
                ry = r
            else:
                ry = eta * r
            ell_c.append((center, rx, ry))
        out[int(c)] = ell_c

    return out
