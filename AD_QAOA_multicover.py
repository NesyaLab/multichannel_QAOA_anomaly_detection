# AD_QAOA_multicover.py
# ============================================================
# Multichannel covering utilities (geometric, ellipse-based)
# V0: do not modify centers; just detect using global union
# ============================================================

from __future__ import annotations
from typing import Dict, List, Tuple, Sequence, Any
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from functions.AD_utilities import detection_stats_from_timestamps


Ellipse2D = Tuple[Tuple[float, float], float, float]   # ((cx, cy), rx, ry)


# -----------------------------
# Geometry primitives
# -----------------------------
def inside_ellipse_2d(
    t: float,
    y: float,
    ellipse: Ellipse2D,
    tol: float = 0.0,
) -> bool:
    """
    Geometric inside test for an axis-aligned ellipse:
      ((t-cx)/rx)^2 + ((y-cy)/ry)^2 <= 1 + tol
    """
    (cx, cy), rx, ry = ellipse
    rx = float(rx)
    ry = float(ry)
    if rx <= 0.0 or ry <= 0.0:
        return False
    dt = (float(t) - float(cx)) / rx
    dv = (float(y) - float(cy)) / ry
    return (dt * dt + dv * dv) <= (1.0 + float(tol))


def flatten_ellipses(ellipses_by_channel: Dict[int, Sequence[Ellipse2D]]) -> List[Ellipse2D]:
    """
    Flatten dict-of-lists into a single list of ellipses.
    """
    out: List[Ellipse2D] = []
    for _, ell_list in ellipses_by_channel.items():
        out.extend(list(ell_list))
    return out


def contiguous_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Given a boolean mask over indices, return contiguous (start,end) index segments
    where mask == True. Indices are inclusive.
    """
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    segs: List[Tuple[int, int]] = []
    s = int(idx[0])
    p = int(idx[0])
    for x in idx[1:]:
        x = int(x)
        if x == p + 1:
            p = x
        else:
            segs.append((s, p))
            s = x
            p = x
    segs.append((s, p))
    return segs


# -----------------------------
# Detection V0 (GLOBAL union)
# -----------------------------
def mv_detection_global_all_ellipses(
    X_mv: Sequence[Tuple[int, np.ndarray]],
    ellipses_by_channel: Dict[int, Sequence[Ellipse2D]],
    tol: float = 0.0,
    any_uncovered_makes_timestamp_anomaly: bool = True,
    make_plots: bool = True,
    title: str = "MV global anomalies (test each (t,y_c) vs ALL ellipses)",
    figsize_total: Tuple[float, float] = (30, 6),
    band_alpha: float = 0.10,
) -> Dict[str, Any]:
    """
    V0 multichannel detection (your current definition):
      - build a GLOBAL set of ellipses = union over channels
      - for each timestamp i and channel c:
            point p_ic = (t_i, y_c(t_i))
            covered_ic = exists ellipse E in GLOBAL set s.t. inside(p_ic, E)
      - timestamp anomaly:
            if any_uncovered_makes_timestamp_anomaly:
                anomaly(t_i) = any_c (not covered_ic)
            else:
                anomaly(t_i) = all_c (not covered_ic)

    Returns:
      dict with keys:
        t, Y, uncovered_mask, anom_ts_mask, anom_ts, anom_points
    """

    # unpack
    t = np.array([int(tt) for tt, _ in X_mv], dtype=int)
    Y = np.stack([np.asarray(v, dtype=float) for _, v in X_mv], axis=0)  # (n,C)
    n, C = Y.shape

    # global ellipses
    all_ellipses = flatten_ellipses(ellipses_by_channel)
    tol = float(tol)

    # uncovered_mask[i,c] = True if point is outside ALL ellipses
    uncovered_mask = np.zeros((n, C), dtype=bool)

    for i in range(n):
        tt = float(t[i])
        for c in range(C):
            yy = float(Y[i, c])

            ok = False
            for E in all_ellipses:
                if inside_ellipse_2d(tt, yy, E, tol=tol):
                    ok = True
                    break
            uncovered_mask[i, c] = (not ok)

    if any_uncovered_makes_timestamp_anomaly:
        anom_ts_mask = np.any(uncovered_mask, axis=1)
    else:
        anom_ts_mask = np.all(uncovered_mask, axis=1)

    anom_ts = [int(t[i]) for i in np.where(anom_ts_mask)[0]]

    # point-level anomalies (diagnostic)
    anom_points: List[Tuple[int, int, float]] = []
    bad = np.where(uncovered_mask)
    for i, c in zip(bad[0].tolist(), bad[1].tolist()):
        anom_points.append((int(t[i]), int(c), float(Y[i, c])))

    # prints
    print(f"Total points tested: {n*C} (n*C = {n}*{C})")
    print(f"Total ellipses used: {len(all_ellipses)} (sum over channels)")
    print(f"Timestamp anomalies: {int(np.sum(anom_ts_mask))} / {n}")
    for c in range(C):
        print(f"  Channel {c}: uncovered {int(np.sum(uncovered_mask[:, c]))} / {n}")

    # plot total (big)
    if make_plots:
        segs = contiguous_segments(anom_ts_mask)

        fig, ax = plt.subplots(figsize=figsize_total)

        for (s, e) in segs:
            ax.axvspan(t[s] - 0.5, t[e] + 0.5, color="red", alpha=band_alpha)

        for c in range(C):
            ax.plot(t, Y[:, c], marker="o", markersize=3, linewidth=1.8, alpha=0.9, label=f"channel {c}")

            bad_c = np.where(uncovered_mask[:, c])[0]
            if bad_c.size > 0:
                ax.scatter(t[bad_c], Y[bad_c, c], s=90, marker="x", color="red", linewidths=2)

        ax.set_title(title)
        ax.set_xlabel("t")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper left", ncols=2)
        plt.tight_layout()
        plt.show()

    return {
        "t": t,
        "Y": Y,
        "uncovered_mask": uncovered_mask,
        "anom_ts_mask": anom_ts_mask,
        "anom_ts": anom_ts,
        "anom_points": anom_points,
    }










# -----------------------------
# Detection V1 (GLOBAL minimized overlap)
# -----------------------------
def reduce_overlap_same_timestamp(
    ellipses_by_channel: Dict[int, Sequence[Ellipse2D]],
    gap: float = 0.0,
    eps: float = 1e-12,
) -> Tuple[Dict[int, List[Ellipse2D]], Dict[str, Any]]:
    """
    Strategy #1 (overlap reduction at same timestamp cx):
      - Consider ALL ellipses across ALL channels.
      - Group ellipses by same cx (timestamp).
      - For each group: keep upper ellipses fixed and push lower ones DOWN
        until consecutive ellipses just touch on the border (no overlap).

    We only move cy. rx, ry remain unchanged.

    Touch/no-overlap condition for same cx:
        cy_above - cy_below >= ry_above + ry_below + gap

    Args:
      gap: extra vertical separation (0.0 => just touching).
      eps: tiny tolerance to avoid numerical issues.

    Returns:
      new_ellipses_by_channel, report
    """

    gap = float(gap)

    # ---- Flatten with provenance so we can rebuild dict ----
    flat = []  # entries: (channel, idx_in_channel, ((cx,cy), rx, ry))
    for c, ell_list in ellipses_by_channel.items():
        for k, E in enumerate(list(ell_list)):
            flat.append((int(c), int(k), E))

    # ---- Group by cx ----
    groups: Dict[int, List[Tuple[int,int,Ellipse2D]]] = {}
    for c, k, E in flat:
        (cx, cy), rx, ry = E
        cx_i = int(round(float(cx)))  # timestamps are ints in your pipeline
        groups.setdefault(cx_i, []).append((c, k, E))

    # We'll build mutable buffers per channel
    new_by_channel: Dict[int, List[Ellipse2D]] = {
        int(c): list(map(tuple, ellipses_by_channel[c]))  # shallow copy
        for c in ellipses_by_channel.keys()
    }

    shift_log = []  # (cx, channel, old_cy, new_cy, ry)

    # ---- Process each timestamp group independently ----
    for cx, items in groups.items():
        if len(items) <= 1:
            continue

        # sort by cy descending (top first). We'll keep top fixed and push others down.
        items_sorted = sorted(items, key=lambda x: float(x[2][0][1]), reverse=True)

        # start from the top ellipse (unchanged)
        (_, _, E_top) = items_sorted[0]
        (cx0, cy_top), rx_top, ry_top = E_top
        prev_cy = float(cy_top)
        prev_ry = float(ry_top)

        # for each lower ellipse, enforce non-overlap with the one immediately above
        for (c, k, E) in items_sorted[1:]:
            (cx_i, cy_i), rx_i, ry_i = E
            cy_i = float(cy_i)
            ry_i = float(ry_i)

            # target: place this ellipse just below the previous one
            target_cy = prev_cy - (prev_ry + ry_i + gap)

            if cy_i > target_cy + eps:
                # overlap -> push down
                new_cy = target_cy
            else:
                # already far enough -> keep as is
                new_cy = cy_i

            # write back
            new_E = ((float(cx_i), float(new_cy)), float(rx_i), float(ry_i))
            new_by_channel[int(c)][int(k)] = new_E
            shift_log.append((int(cx), int(c), float(cy_i), float(new_cy), float(ry_i)))

            # update "previous" to this ellipse (now fixed for the next below)
            prev_cy = float(new_cy)
            prev_ry = float(ry_i)

    report = {
        "n_shifted": len([s for s in shift_log if abs(s[2]-s[3]) > 0]),
        "shift_log": shift_log,  # list of (cx, channel, old_cy, new_cy, ry)
    }
    return new_by_channel, report












# -----------------------------
# Detection V2 (GLOBAL shifting cx only to improve coverage)
# -----------------------------
def recenter_ellipses_x_mv(
    X_mv_train,
    ellipses_by_channel,
    labels_by_ts,
    step=0.05,
    delta=3.0,
    K_ts_candidates=(6, 10, 14, 20, 30),
    lambda_anom=3.0,
    verbose=True,
):
    """
    Recenter delle ellissi SOLO lungo asse x.
    - cy, rx, ry restano invariati
    - accetta shift solo se migliora la copertura dei normali
      (penalizzando gli anomali)
    - usa TUTTA la serie multivariata (tutti i canali)

    Returns:
        new_ellipses_by_channel
        debug_report (list of dict)
    """

    # ---------- helpers ----------
    def inside_ellipse(t, y, cx, cy, rx, ry):
        if rx <= 0 or ry <= 0:
            return False
        return ((t - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0

    def score_for_center(cx, cy, rx, ry, ts_subset):
        n_norm, n_anom = 0, 0

        for t, vec in ts_subset:
            for y in vec:
                if inside_ellipse(t, y, cx, cy, rx, ry):
                    if labels_by_ts[t]:
                        n_anom += 1
                    else:
                        n_norm += 1

        score = n_norm - lambda_anom * n_anom
        return score, n_norm, n_anom

    # ---------- prepare ----------
    X_by_ts = {t: vec for t, vec in X_mv_train}
    timestamps = np.array(sorted(X_by_ts.keys()), dtype=float)

    new_ellipses_by_channel = deepcopy(ellipses_by_channel)
    debug_report = []

    total_shifted = 0

    # ---------- main loop ----------
    for c, ell_list in ellipses_by_channel.items():
        if verbose:
            print(f"\n[Channel {c}] processing {len(ell_list)} ellipses")

        for idx, ((cx0, cy), rx, ry) in enumerate(ell_list):
            best_score = None
            best_cx = cx0
            best_K = None

            # faccio grid search su K
            for K in K_ts_candidates:
                # seleziona finestra temporale
                mask = np.abs(timestamps - cx0) <= K
                ts_subset = [(int(t), X_by_ts[int(t)]) for t in timestamps[mask]]

                if len(ts_subset) == 0:
                    continue

                # score iniziale
                s0, n0, a0 = score_for_center(cx0, cy, rx, ry, ts_subset)

                # facci prova su ogni posizione nella griglia disponibile
                for dx in np.arange(-delta, delta + step, step):
                    cx_new = cx0 + dx
                    s, n, a = score_for_center(cx_new, cy, rx, ry, ts_subset)

                    if best_score is None or s > best_score:
                        best_score = s
                        best_cx = cx_new
                        best_K = K

            # accetta solo se migliora rispetto all'origine
            s_orig, _, _ = score_for_center(cx0, cy, rx, ry, ts_subset)

            if best_score is not None and best_score > s_orig:
                new_ellipses_by_channel[c][idx] = ((best_cx, cy), rx, ry)
                total_shifted += 1

                debug_report.append({
                    "channel": c,
                    "index": idx,
                    "cx_old": cx0,
                    "cx_new": best_cx,
                    "delta": best_cx - cx0,
                    "best_score": best_score,
                    "orig_score": s_orig,
                    "best_K": best_K,
                })

                if verbose:
                    print(
                        f"  ✔ ellipse {idx}: cx {cx0:.2f} → {best_cx:.2f} "
                        f"(Δ={best_cx - cx0:+.2f}, K={best_K}, score {s_orig}→{best_score})"
                    )
            else:
                if verbose:
                    print(f"  · ellipse {idx}: no beneficial shift")

    if verbose:
        print("\n========== RECENTER SUMMARY ==========")
        print(f"Total ellipses shifted: {total_shifted}")
        print("=====================================\n")

    return new_ellipses_by_channel, debug_report




# -----------------------------
# Detection V3 (GLOBAL simplex cover in (t, y1..yC) )
#   - vertices picked by parallel timestamps (full multivariate vector)
#   - STOP when next candidate timestamp is anomalous (if labels_by_ts provided)
#   - ENFORCE barycenter == QAOA center point in R^(C+1):
#         center_vec = [t0, y0(t0), ..., yC-1(t0)]
#     by translating vertices (shape can be unbalanced, bary is forced)
#   - tolerance: isotropic expansion around barycenter (scale 1+tol_expand)
# -----------------------------

from typing import Optional, Sequence, Tuple, Dict, Any, List, Union
import numpy as np

SimplexCover = Dict[str, Any]  # {"center_t":int, "bary":np.ndarray, "verts":np.ndarray, "dim":int, ...}

def _as_Xmv_dict(X_mv: Sequence[Tuple[int, np.ndarray]]) -> Dict[int, np.ndarray]:
    return {int(t): np.asarray(vec, dtype=float).ravel() for t, vec in X_mv}

def _build_point_ty(t: int, vec: np.ndarray) -> np.ndarray:
    # point in R^(C+1): [t, y1, ..., yC]
    v = np.asarray(vec, dtype=float).ravel()
    return np.concatenate(([float(t)], v))

def _expand_vertices_isotropic(verts: np.ndarray, bary: np.ndarray, tol_expand: float) -> np.ndarray:
    """
    Isotropic expansion around barycenter:
      v' = bary + (1+tol_expand)*(v - bary)
    tol_expand=0.5 => +50% in every direction from bary.
    """
    s = 1.0 + float(tol_expand)
    return bary[None, :] + s * (verts - bary[None, :])

def _extract_centers_ts(
    centers: Sequence[Union[int, Tuple, List, np.integer, np.floating]]
) -> List[int]:
    """
    Accepts either:
      - centers_ts: [t0, t1, ...]
      - unique_centers like [(t,c,cy), ...]  -> take first element as t
    Returns sorted unique timestamps.
    """
    ts = []
    for x in centers:
        if isinstance(x, (tuple, list, np.ndarray)) and len(x) >= 1:
            ts.append(int(x[0]))
        else:
            ts.append(int(x))
    return sorted(set(ts))

def build_simplices_from_centers(
    X_mv_train: Sequence[Tuple[int, np.ndarray]],
    centers: Sequence[Union[int, Tuple, List, np.integer, np.floating]],
    labels_by_ts: Optional[Dict[int, bool]] = None,
    k_max: int = 30,
    K_ts_candidates: Sequence[int] = (6, 10, 14, 20, 30),
    stop_when_hit_anomaly: bool = True,
    tol_expand: float = 0.5,
    verbose: bool = True,
) -> Tuple[List[SimplexCover], Dict[str, Any]]:
    """
    Per ogni centro (timestamp) costruisce un cover convesso (tipo simplesso) in spazio (t, y1..yC)
    usando i k vicini temporali (timestamp "paralleli", quindi vettore multivariato completo per ogni t),
    con k che cresce fino a:
      - incontrare anomalia (se labels_by_ts e stop_when_hit_anomaly), oppure
      - k_max, oppure
      - non ci sono abbastanza punti.

    Vincolo richiesto:
      - il baricentro del politope deve coincidere col "centro QAOA" multivariato:
            center_vec = [t0, y(t0)]
        anche se la figura risulta sbilanciata.
        -> si ottiene traslando tutti i vertici per imporre bary == center_vec.

    Il cover è salvato come:
      - verts: array (m, D) con D=C+1 (già espanso con tolleranza)
      - bary: baricentro imposto (=center_vec)
    """

    from scipy.spatial import Delaunay  # local import

    X_by_ts = _as_Xmv_dict(X_mv_train)
    ts_all = np.array(sorted(X_by_ts.keys()), dtype=int)
    if ts_all.size == 0:
        raise ValueError("X_mv_train is empty")

    C = int(np.asarray(next(iter(X_by_ts.values()))).size)
    D = C + 1  # (t + C channels)

    labels_by_ts = labels_by_ts or {}

    centers_ts = _extract_centers_ts(centers)

    simplices: List[SimplexCover] = []
    build_log: List[Dict[str, Any]] = []

    n_skipped_rank = 0
    n_skipped_too_few = 0
    n_skipped_missing_center_vec = 0

    min_needed = D + 1  # to have a full-dimensional hull in R^D

    for t0 in centers_ts:
        t0 = int(t0)

        if t0 not in X_by_ts:
            # non posso costruire center_vec=[t0, y(t0)] se manca y(t0)
            n_skipped_missing_center_vec += 1
            if verbose:
                print(f"[V3] center {t0}: missing X_mv_train vector at t0 -> skipped")
            continue

        # centro QAOA multivariato (bary vincolato)
        center_vec = _build_point_ty(t0, X_by_ts[t0])  # shape (D,)

        best_cover = None
        best_info = None

        for K in K_ts_candidates:
            K = int(K)

            mask = np.abs(ts_all - t0) <= K
            ts_win = ts_all[mask]
            if ts_win.size < min_needed:
                continue

            # ordina per distanza temporale: t0, t0±1, ...
            order = np.argsort(np.abs(ts_win - t0))
            ts_sorted = ts_win[order].tolist()

            chosen_ts: List[int] = []
            hit_anomaly = False

            for t in ts_sorted:
                t = int(t)
                if stop_when_hit_anomaly and bool(labels_by_ts.get(t, False)):
                    hit_anomaly = True
                    break
                chosen_ts.append(t)
                if len(chosen_ts) >= int(k_max):
                    break

            if len(chosen_ts) < min_needed:
                n_skipped_too_few += 1
                continue

            # Vertici in R^(C+1): timestamp paralleli -> vettore multivariato completo per ogni t
            verts = np.vstack([_build_point_ty(t, X_by_ts[t]) for t in chosen_ts])  # (m, D)

            # 1) bary reale dei vertici
            bary_real = np.mean(verts, axis=0)

            # 2) traslazione per imporre bary == center_vec (vincolo richiesto)
            shift = center_vec - bary_real
            verts_shifted = verts + shift[None, :]

            # bary ora è esattamente center_vec
            bary_forced = center_vec.copy()

            # 3) tolleranza: espansione isotropa rispetto al baricentro imposto
            verts_exp = _expand_vertices_isotropic(verts_shifted, bary_forced, tol_expand=tol_expand)

            # controlla rango: hull pieno dimensionalmente
            if np.linalg.matrix_rank(verts_exp - verts_exp[0:1, :]) < D:
                n_skipped_rank += 1
                continue

            # prova Delaunay (inside-test)
            try:
                _ = Delaunay(verts_exp)
            except Exception:
                n_skipped_rank += 1
                continue

            # seleziona il cover migliore: più vertici (k maggiore)
            if (best_cover is None) or (verts_exp.shape[0] > best_cover["verts"].shape[0]):
                best_cover = {
                    "center_t": t0,
                    "bary": bary_forced,
                    "verts": verts_exp,
                    "dim": D,
                    "k_used": int(verts_exp.shape[0]),
                    "K_used": int(K),
                }
                best_info = {
                    "t0": t0,
                    "K": int(K),
                    "k_used": int(verts_exp.shape[0]),
                    "hit_anomaly": bool(hit_anomaly),
                }

        if best_cover is not None:
            simplices.append(best_cover)
            build_log.append(best_info)
            if verbose:
                print(f"[V3] center {t0}: K={best_info['K']}, k={best_info['k_used']}, hit_anom={best_info['hit_anomaly']}")
        else:
            if verbose:
                print(f"[V3] center {t0}: no valid simplex built (degenerate/too few points)")

    report = {
        "n_centers_input": len(list(centers)),
        "n_centers_unique_ts": len(centers_ts),
        "n_simplices": len(simplices),
        "n_skipped_rank": n_skipped_rank,
        "n_skipped_too_few": n_skipped_too_few,
        "n_skipped_missing_center_vec": n_skipped_missing_center_vec,
        "build_log": build_log,
        "dim": D,
        "tol_expand": float(tol_expand),
        "k_max": int(k_max),
        "K_ts_candidates": list(map(int, K_ts_candidates)),
        "stop_when_hit_anomaly": bool(stop_when_hit_anomaly),
    }
    return simplices, report


def mv_detection_global_all_simplices(
    X_mv_test: Sequence[Tuple[int, np.ndarray]],
    simplices: Sequence[SimplexCover],
    any_uncovered_makes_timestamp_anomaly: bool = True,
    make_plots: bool = True,
    title: str = "MV anomalies via GLOBAL union of simplex-covers",
    figsize_total: Tuple[float, float] = (30, 6),
    band_alpha: float = 0.10,
) -> Dict[str, Any]:
    """
    Detection usando unione globale di covers convessi (tipo simplesso) in spazio (t, y1..yC).
    Un timestamp è coperto se il punto (t, y_vec) cade in ALMENO un simplex.
    """

    from scipy.spatial import Delaunay
    import matplotlib.pyplot as plt

    X_list = [(int(t), np.asarray(vec, dtype=float).ravel()) for t, vec in X_mv_test]
    t_arr = np.array([t for t, _ in X_list], dtype=int)
    Y = np.vstack([vec for _, vec in X_list])  # (n, C)
    n, C = Y.shape

    if len(simplices) == 0:
        raise ValueError("No simplices provided to detection")

    # prebuild Delaunay triangulations
    delaunays: List[Delaunay] = []
    for S in simplices:
        verts = np.asarray(S["verts"], dtype=float)
        delaunays.append(Delaunay(verts))

    # uncovered_mask (timestamp-level replicated per channel for compatibility)
    uncovered_mask = np.zeros((n, C), dtype=bool)

    for i in range(n):
        x = _build_point_ty(int(t_arr[i]), Y[i, :])  # (D,)
        covered = False
        for tri in delaunays:
            if tri.find_simplex(x) >= 0:
                covered = True
                break
        if not covered:
            uncovered_mask[i, :] = True

    if any_uncovered_makes_timestamp_anomaly:
        anom_ts_mask = np.any(uncovered_mask, axis=1)
    else:
        anom_ts_mask = np.all(uncovered_mask, axis=1)

    anom_ts = [int(t_arr[i]) for i in np.where(anom_ts_mask)[0]]

    print(f"[V3] Total timestamps tested: {n}")
    print(f"[V3] Total simplices used: {len(simplices)}")
    print(f"[V3] Timestamp anomalies: {int(np.sum(anom_ts_mask))} / {n}")

    if make_plots:
        fig, ax = plt.subplots(1, 1, figsize=figsize_total)
        for c in range(C):
            ax.plot(t_arr, Y[:, c], lw=0.8, label=f"ch {c}")
        for tt in anom_ts:
            ax.axvspan(tt - 0.5, tt + 0.5, alpha=band_alpha)
        ax.set_title(title)
        ax.legend(ncol=min(C, 6))
        plt.show()

    return {
        "anom_ts": anom_ts,
        "anom_ts_mask": anom_ts_mask,
        "uncovered_mask": uncovered_mask,
        "n_simplices": len(simplices),
    }



def mv_detection_v3_with_stats_and_plot(
    X_mv_test,
    simplices,
    labels_by_ts,
    any_uncovered_makes_timestamp_anomaly=True,
    title="Multivariate V3 detection (simplex cover)",
    figsize=(30, 6),
    band_alpha=0.12,
    verbose=True,
):
    """
    Wrapper completo per Detection V3:
      - detection con union globale dei simplessi
      - plotting multivariato con bande anomale
      - statistiche timestamp-level (TP, FP, FN, TN, ecc.)

    Args:
        X_mv_test : list[(t, y_vec)]
        simplices : output di build_simplices_from_centers
        labels_by_ts : dict[int -> bool]
        any_uncovered_makes_timestamp_anomaly : bool
        title : str
        figsize : tuple
        band_alpha : float
        verbose : bool

    Returns:
        out : dict con detection + stats
    """

    # -----------------------
    # 1) Detection V3 pura
    # -----------------------
    out_det = mv_detection_global_all_simplices(
        X_mv_test=X_mv_test,
        simplices=simplices,
        any_uncovered_makes_timestamp_anomaly=any_uncovered_makes_timestamp_anomaly,
        make_plots=False,   # plottiamo noi sotto
    )

    anom_ts = out_det["anom_ts"]
    anom_ts_mask = out_det["anom_ts_mask"]
    uncovered_mask = out_det["uncovered_mask"]

    # -----------------------
    # 2) Plot multivariato
    # -----------------------
    import matplotlib.pyplot as plt
    import numpy as np

    X_list = [(int(t), np.asarray(vec, dtype=float)) for t, vec in X_mv_test]
    t_arr = np.array([t for t, _ in X_list])
    Y = np.vstack([vec for _, vec in X_list])
    n, C = Y.shape

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for c in range(C):
        ax.plot(t_arr, Y[:, c], lw=1.0, label=f"Channel {c}")

    # bande verticali sugli anomaly timestamps
    for t in anom_ts:
        ax.axvspan(t - 0.5, t + 0.5, color="red", alpha=band_alpha)

    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Value")
    ax.legend(ncol=min(C, 6))
    ax.grid(alpha=0.3)
    plt.show()

    # -----------------------
    # 3) Coverage per canale
    # -----------------------
    if verbose:
        print(f"\nTimestamp anomalies: {int(np.sum(anom_ts_mask))} / {n}")
        for c in range(C):
            n_unc = int(np.sum(uncovered_mask[:, c]))
            print(f"Channel {c}: uncovered {n_unc} / {n}")

    # -----------------------
    # 4) Statistiche timestamp-level
    # -----------------------
    stats = detection_stats_from_timestamps(
        labels_by_ts=labels_by_ts,
        detected_anom_ts=anom_ts,
        verbose=verbose,
    )

    return {
        "anom_ts": anom_ts,
        "anom_ts_mask": anom_ts_mask,
        "uncovered_mask": uncovered_mask,
        "stats": stats,
        "n_simplices": out_det["n_simplices"],
    }






from typing import Optional, Sequence, Tuple, Dict, Any, List, Union
import numpy as np

SimplexCover = Dict[str, Any]

def _as_Xmv_dict(X_mv: Sequence[Tuple[int, np.ndarray]]) -> Dict[int, np.ndarray]:
    return {int(t): np.asarray(vec, dtype=float).ravel() for t, vec in X_mv}

def _build_point_ty(t: int, vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=float).ravel()
    return np.concatenate(([float(t)], v))

def _expand_vertices_isotropic(verts: np.ndarray, bary: np.ndarray, tol_expand: float) -> np.ndarray:
    s = 1.0 + float(tol_expand)
    return bary[None, :] + s * (verts - bary[None, :])

def _extract_centers_ts(centers) -> List[int]:
    ts = []
    for x in centers:
        if isinstance(x, (tuple, list, np.ndarray)) and len(x) >= 1:
            ts.append(int(x[0]))
        else:
            ts.append(int(x))
    return sorted(set(ts))

def build_simplices_v3A_skip_anomalies(
    X_mv_train: Sequence[Tuple[int, np.ndarray]],
    centers,
    labels_by_ts: Optional[Dict[int, bool]] = None,
    k_max: int = 30,
    K_ts_candidates: Sequence[int] = (6, 10, 14, 20, 30),
    tol_expand: float = 0.5,
    min_rank_tol: float = 1e-10,
    verbose: bool = True,
) -> Tuple[List[SimplexCover], Dict[str, Any]]:
    """
    V3A:
      - spazio: p(t) = [t, y0(t), ..., yC-1(t)] in R^(C+1)
      - ordine candidati: crescente per |t - t0|
      - se un candidato è anomalo: lo SALTA (non stoppa)
      - continua finché:
          * ha almeno D+1 vertici (D=C+1), e rank pieno (>=D),
          * oppure finisce candidati / k_max
      - impone bary == center_vec = [t0, y(t0)] tramite traslazione
      - poi espande isotropicamente con tol_expand
    """

    from scipy.spatial import Delaunay

    X_by_ts = _as_Xmv_dict(X_mv_train)
    ts_all = np.array(sorted(X_by_ts.keys()), dtype=int)
    if ts_all.size == 0:
        raise ValueError("X_mv_train is empty")

    C = int(np.asarray(next(iter(X_by_ts.values()))).size)
    D = C + 1
    min_needed = D + 1

    labels_by_ts = labels_by_ts or {}
    centers_ts = _extract_centers_ts(centers)

    simplices = []
    build_log = []
    skipped_missing_center = 0
    skipped_no_solution = 0

    for t0 in centers_ts:
        t0 = int(t0)
        if t0 not in X_by_ts:
            skipped_missing_center += 1
            if verbose:
                print(f"[V3A] center {t0}: missing center vector -> skipped")
            continue

        center_vec = _build_point_ty(t0, X_by_ts[t0])

        best_cover = None
        best_info = None

        for K in K_ts_candidates:
            K = int(K)
            ts_win = ts_all[np.abs(ts_all - t0) <= K]
            if ts_win.size == 0:
                continue

            order = np.argsort(np.abs(ts_win - t0))
            ts_sorted = ts_win[order].tolist()

            chosen_ts = []
            skipped_ts = []

            # prova ad accumulare finché riesci a costruire hull full-rank
            for t in ts_sorted:
                t = int(t)

                if bool(labels_by_ts.get(t, False)):
                    skipped_ts.append(t)
                    continue  # <-- differenza: salta e continua

                chosen_ts.append(t)
                if len(chosen_ts) >= int(k_max):
                    break

                # appena hai abbastanza punti, prova a costruire
                if len(chosen_ts) >= min_needed:
                    verts = np.vstack([_build_point_ty(tt, X_by_ts[tt]) for tt in chosen_ts])
                    bary_real = np.mean(verts, axis=0)
                    shift = center_vec - bary_real
                    verts_shifted = verts + shift[None, :]
                    bary_forced = center_vec.copy()
                    verts_exp = _expand_vertices_isotropic(verts_shifted, bary_forced, tol_expand=tol_expand)

                    # rank check pieno in R^D
                    rank = np.linalg.matrix_rank(verts_exp - verts_exp[0:1, :], tol=min_rank_tol)
                    if rank < D:
                        # ancora degenere: continua a raccogliere altri timestamp
                        continue

                    try:
                        _ = Delaunay(verts_exp)
                    except Exception:
                        continue

                    # costruzione riuscita con il minimo necessario (o oltre)
                    candidate_cover = {
                        "center_t": t0,
                        "bary": bary_forced,
                        "verts": verts_exp,
                        "dim": D,
                        "K_used": K,
                        "k_used": int(verts_exp.shape[0]),
                        "skipped_ts": skipped_ts,
                    }
                    best_cover = candidate_cover
                    best_info = {
                        "t0": t0,
                        "K": K,
                        "k_used": int(verts_exp.shape[0]),
                        "n_skipped": int(len(skipped_ts)),
                        "rank": int(rank),
                    }
                    # per V3A: appena trovi una soluzione valida per questo K,
                    # puoi break oppure continuare per cercare una con più vertici.
                    # Io continuo per trovare la più grande (più stabile).
                    # (se vuoi "minima", metti break qui)
                    # break

            # se abbiamo trovato una cover valida, possiamo anche scegliere
            # la più grande tra diversi K
            if best_cover is not None:
                # tieni quella con più vertici
                if (best_cover is None) or (best_cover["k_used"] > best_cover["k_used"]):
                    pass

        if best_cover is not None:
            simplices.append(best_cover)
            build_log.append(best_info)
            if verbose:
                print(f"[V3A] center {t0}: K={best_info['K']}, k={best_info['k_used']}, skipped={best_info['n_skipped']}, rank={best_info['rank']}")
        else:
            skipped_no_solution += 1
            if verbose:
                print(f"[V3A] center {t0}: no valid simplex built")

    report = {
        "n_centers": len(centers_ts),
        "n_simplices": len(simplices),
        "skipped_missing_center": skipped_missing_center,
        "skipped_no_solution": skipped_no_solution,
        "dim": D,
        "tol_expand": float(tol_expand),
        "k_max": int(k_max),
        "K_ts_candidates": list(map(int, K_ts_candidates)),
        "build_log": build_log,
    }
    return simplices, report


def mv_detection_global_all_simplices(
    X_mv_test: Sequence[Tuple[int, np.ndarray]],
    simplices: Sequence[SimplexCover],
    any_uncovered_makes_timestamp_anomaly: bool = True,
    make_plots: bool = False,
) -> Dict[str, Any]:
    """
    Detection per simplessi multivariati in R^(C+1).
    (uguale a prima, lasciata qui per completezza: se già ce l’hai, non duplicarla)
    """
    from scipy.spatial import Delaunay
    import numpy as np

    X_list = [(int(t), np.asarray(vec, dtype=float).ravel()) for t, vec in X_mv_test]
    t_arr = np.array([t for t, _ in X_list], dtype=int)
    Y = np.vstack([vec for _, vec in X_list])  # (n, C)
    n, C = Y.shape

    if len(simplices) == 0:
        raise ValueError("No simplices provided to detection")

    delaunays = [Delaunay(np.asarray(S["verts"], dtype=float)) for S in simplices]
    uncovered_mask = np.zeros((n, C), dtype=bool)

    for i in range(n):
        x = _build_point_ty(int(t_arr[i]), Y[i, :])
        covered = False
        for tri in delaunays:
            if tri.find_simplex(x) >= 0:
                covered = True
                break
        if not covered:
            uncovered_mask[i, :] = True

    if any_uncovered_makes_timestamp_anomaly:
        anom_ts_mask = np.any(uncovered_mask, axis=1)
    else:
        anom_ts_mask = np.all(uncovered_mask, axis=1)

    anom_ts = [int(t_arr[i]) for i in np.where(anom_ts_mask)[0]]
    return {
        "anom_ts": anom_ts,
        "anom_ts_mask": anom_ts_mask,
        "uncovered_mask": uncovered_mask,
        "n_simplices": len(simplices),
    }







from typing import Optional, Sequence, Tuple, Dict, Any, List, Union
import numpy as np

ChannelHull = Dict[str, Any]  # {"center_t":int, "bary_t":float, "verts2d":np.ndarray, ...}

def _as_Xmv_dict(X_mv: Sequence[Tuple[int, np.ndarray]]) -> Dict[int, np.ndarray]:
    return {int(t): np.asarray(vec, dtype=float).ravel() for t, vec in X_mv}

def _extract_centers_ts(centers) -> List[int]:
    ts = []
    for x in centers:
        if isinstance(x, (tuple, list, np.ndarray)) and len(x) >= 1:
            ts.append(int(x[0]))
        else:
            ts.append(int(x))
    return sorted(set(ts))

def _expand_2d_isotropic(verts2d: np.ndarray, bary2d: np.ndarray, tol_expand: float) -> np.ndarray:
    s = 1.0 + float(tol_expand)
    return bary2d[None, :] + s * (verts2d - bary2d[None, :])

def build_hulls_v3B_channel_as_point(
    X_mv_train: Sequence[Tuple[int, np.ndarray]],
    centers,
    labels_by_ts: Optional[Dict[int, bool]] = None,
    k_max: int = 30,
    K_ts_candidates: Sequence[int] = (6, 10, 14, 20, 30),
    stop_when_hit_anomaly: bool = True,   # qui: se il timestamp candidato è anomalo, stop (come nella tua regola base)
    tol_expand: float = 0.5,
    verbose: bool = True,
) -> Tuple[List[ChannelHull], Dict[str, Any]]:
    """
    V3B:
      - spazio 2D: punti per canale p_c(t) = [t, y_c(t)] in R^2
      - per un centro t0, prendi timestamp vicini (k cresce) e accumuli C punti per timestamp
      - stop se tocchi un timestamp anomalo (opzione)
      - costruisci convex hull / triangolazione 2D per la nuvola di punti
      - baricentro imposto: (t0, mean_c y_c(t0)) oppure (t0, y_c(t0))? -> qui uso media sui canali
        e traslo la nuvola per imporre bary == center2d
      - tol_expand: espansione isotropa 2D
    """
    from scipy.spatial import Delaunay

    X_by_ts = _as_Xmv_dict(X_mv_train)
    ts_all = np.array(sorted(X_by_ts.keys()), dtype=int)
    if ts_all.size == 0:
        raise ValueError("X_mv_train is empty")

    labels_by_ts = labels_by_ts or {}
    centers_ts = _extract_centers_ts(centers)

    hulls = []
    build_log = []

    for t0 in centers_ts:
        t0 = int(t0)
        if t0 not in X_by_ts:
            if verbose:
                print(f"[V3B] center {t0}: missing -> skipped")
            continue

        y0 = X_by_ts[t0]
        # centro 2D: (t0, media canali a t0)  (scelta naturale per bary in 2D)
        center2d = np.array([float(t0), float(np.mean(y0))], dtype=float)

        best = None
        best_info = None

        for K in K_ts_candidates:
            K = int(K)
            ts_win = ts_all[np.abs(ts_all - t0) <= K]
            if ts_win.size == 0:
                continue

            order = np.argsort(np.abs(ts_win - t0))
            ts_sorted = ts_win[order].tolist()

            pts = []
            hit_anom = False

            for t in ts_sorted:
                t = int(t)

                if stop_when_hit_anomaly and bool(labels_by_ts.get(t, False)):
                    hit_anom = True
                    break

                vec = X_by_ts[t]
                # C punti per timestamp: (t, y_c(t))
                for yc in vec:
                    pts.append([float(t), float(yc)])

                # k_max in termini di timestamp, non punti
                if len(set(int(p[0]) for p in pts)) >= int(k_max):
                    break

            pts = np.asarray(pts, dtype=float)
            if pts.shape[0] < 3:
                continue  # in 2D servono almeno 3 punti non collineari

            # imponi baricentro == center2d con traslazione
            bary_real = np.mean(pts, axis=0)
            shift = center2d - bary_real
            pts_shift = pts + shift[None, :]

            # espandi tolleranza
            pts_exp = _expand_2d_isotropic(pts_shift, center2d, tol_expand=tol_expand)

            # prova Delaunay 2D
            try:
                tri = Delaunay(pts_exp)
            except Exception:
                continue

            # scegli migliore: più punti
            if (best is None) or (pts_exp.shape[0] > best["verts2d"].shape[0]):
                best = {
                    "center_t": t0,
                    "bary2d": center2d,
                    "verts2d": pts_exp,
                    "K_used": K,
                    "n_points": int(pts_exp.shape[0]),
                }
                best_info = {
                    "t0": t0,
                    "K": K,
                    "n_points": int(pts_exp.shape[0]),
                    "hit_anom": bool(hit_anom),
                }

        if best is not None:
            hulls.append(best)
            build_log.append(best_info)
            if verbose:
                print(f"[V3B] center {t0}: K={best_info['K']}, points={best_info['n_points']}, hit_anom={best_info['hit_anom']}")
        else:
            if verbose:
                print(f"[V3B] center {t0}: no valid 2D hull built")

    report = {"n_centers": len(centers_ts), "n_hulls": len(hulls), "build_log": build_log}
    return hulls, report


def mv_detection_v3B_channel_points(
    X_mv_test: Sequence[Tuple[int, np.ndarray]],
    hulls: Sequence[ChannelHull],
    mode: str = "all_channels",  # "all_channels" or "any_channel"
) -> Dict[str, Any]:
    """
    Detection V3B:
      - per ogni timestamp t nel test, genera C punti (t, y_c(t))
      - un timestamp è "covered" se:
          mode="all_channels": tutti i C punti sono dentro almeno un hull (stesso hull o hull diversi)
          mode="any_channel": basta che almeno 1 canale sia coperto
      - uncovered_mask per canale viene calcolato davvero in 2D.

    Returns:
      dict compatibile con stats+plot wrapper (anom_ts, uncovered_mask)
    """
    from scipy.spatial import Delaunay

    X_list = [(int(t), np.asarray(vec, dtype=float).ravel()) for t, vec in X_mv_test]
    t_arr = np.array([t for t, _ in X_list], dtype=int)
    Y = np.vstack([vec for _, vec in X_list])
    n, C = Y.shape

    if len(hulls) == 0:
        raise ValueError("No hulls provided")

    delaunays = []
    metas = []
    for H in hulls:
        verts2d = np.asarray(H["verts2d"], dtype=float)
        delaunays.append(Delaunay(verts2d))
        metas.append(H)

    uncovered_mask = np.ones((n, C), dtype=bool)  # True=uncovered

    for i in range(n):
        t = float(t_arr[i])
        for c in range(C):
            pt = np.array([t, float(Y[i, c])], dtype=float)
            covered_c = False
            for tri in delaunays:
                if tri.find_simplex(pt) >= 0:
                    covered_c = True
                    break
            uncovered_mask[i, c] = (not covered_c)

    if mode == "all_channels":
        anom_ts_mask = np.any(uncovered_mask, axis=1)  # se almeno un canale scoperto => anomalia
    elif mode == "any_channel":
        anom_ts_mask = np.all(uncovered_mask, axis=1)  # anomalia solo se tutti scoperti
    else:
        raise ValueError("mode must be 'all_channels' or 'any_channel'")

    anom_ts = [int(t_arr[i]) for i in np.where(anom_ts_mask)[0]]

    return {
        "anom_ts": anom_ts,
        "anom_ts_mask": anom_ts_mask,
        "uncovered_mask": uncovered_mask,
        "n_simplices": len(hulls),  # per compatibilità nome
    }


def mv_detection_v3B_with_stats_and_plot(
    X_mv_test,
    hulls,
    labels_by_ts,  # <-- passa qui labels_by_ts_anomaly
    mode="all_channels",
    title="V3B – Channel-as-point 2D hull cover",
    figsize=(30, 6),
    band_alpha=0.12,
    verbose=True,
):
    """
    V3B full report:
      - detection con mv_detection_v3B_channel_points
      - plot nello stile della vecchia mv_detection_global_all_ellipses:
          * bande rosse su segmenti contigui di anomalia (timestamp-level)
          * X rosse sui punti uncovered per canale
      - stats timestamp-level via detection_stats_from_timestamps (la tua)
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # ---- helper: segmenti contigui True ----
    def contiguous_segments(mask: np.ndarray):
        mask = np.asarray(mask, dtype=bool)
        n = mask.size
        segs = []
        i = 0
        while i < n:
            if not mask[i]:
                i += 1
                continue
            s = i
            while i + 1 < n and mask[i + 1]:
                i += 1
            e = i
            segs.append((s, e))
            i += 1
        return segs

    # -----------------------
    # 1) Detection V3B
    # -----------------------
    out_det = mv_detection_v3B_channel_points(X_mv_test, hulls=hulls, mode=mode)

    anom_ts = out_det["anom_ts"]
    anom_ts_mask = np.asarray(out_det["anom_ts_mask"], dtype=bool)
    uncovered_mask = np.asarray(out_det["uncovered_mask"], dtype=bool)

    # unpack serie test
    t = np.array([int(tt) for tt, _ in X_mv_test], dtype=int)
    Y = np.stack([np.asarray(v, dtype=float) for _, v in X_mv_test], axis=0)  # (n,C)
    n, C = Y.shape

    # -----------------------
    # 2) Print diagnostica (come ellissi)
    # -----------------------
    if verbose:
        print(f"\nTimestamp anomalies: {int(np.sum(anom_ts_mask))} / {n}")
        for c in range(C):
            print(f"Channel {c}: uncovered {int(np.sum(uncovered_mask[:, c]))} / {n}")

    # -----------------------
    # 3) Plot stile ellissi
    # -----------------------
    segs = contiguous_segments(anom_ts_mask)

    fig, ax = plt.subplots(figsize=figsize)

    # bande rosse sui segmenti
    for (s, e) in segs:
        ax.axvspan(t[s] - 0.5, t[e] + 0.5, color="red", alpha=band_alpha)

    # linee + marker e X rosse sui punti uncovered
    for c in range(C):
        ax.plot(t, Y[:, c], marker="o", markersize=3, linewidth=1.8, alpha=0.9, label=f"channel {c}")

        bad_c = np.where(uncovered_mask[:, c])[0]
        if bad_c.size > 0:
            ax.scatter(t[bad_c], Y[bad_c, c], s=90, marker="x", color="red", linewidths=2)

    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", ncols=2)
    plt.tight_layout()
    plt.show()

    # -----------------------
    # 4) Statistiche (usa la tua funzione)
    # -----------------------
    stats = detection_stats_from_timestamps(labels_by_ts, anom_ts, verbose=verbose)

    return {**out_det, "stats": stats}




