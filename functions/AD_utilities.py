"Anomaly detection Utilites functions"




import numpy as np
import pandas as pd
import math
from typing import Tuple, List, Dict
from matplotlib import pyplot as plt
from statistics import mean
import random
import gdown
import time
import itertools
from AD_QAOA_ext import AD_QAOA 
import functions.AD_preprocessing as preprocessing
import functions.AD_detection as detection
import functions.AD_training as training



def plot_training_time_series(dataset_train):
    """
    Plots the training time series.

    Args:
        dataset_train (list of tuples): The training dataset represented as a list of (timestamp, value) pairs.

    """
    times = np.array([t for t, _ in dataset_train])
    values = np.array([v for _, v in dataset_train])

    plt.figure(figsize=(10, 6))
    plt.plot(times, values, marker='o', color='goldenrod', linestyle='-', label='Time Series')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Training Time Series')
    plt.show()




def plot_test_time_series(dataset_test):
    """
    Plots the test time series.
    Commentato i titoli e i nomi degli assi.
    Commentato salvatore pdf.

    Args:
        dataset_test (list of tuples): The test dataset represented as a list of (timestamp, value) pairs.

    """
    times = np.array([t for t, _ in dataset_test])
    values = np.array([v for _, v in dataset_test])

    plt.figure(figsize=(10, 6))
    plt.plot(times, values, marker='o', color='blue', linestyle='-', label='Time Series')
    #plt.xlabel('Time')
    #plt.ylabel('Value')
    #plt.title('Test Time Series')
    plt.grid(True)

    #output_pdf = 'test_set.pdf'
    #plt.tight_layout()
    #plt.savefig(output_pdf, format='pdf', dpi=300)
    #print(f"Grafico salvato come: {output_pdf}")
    
    plt.show()




def plot_training_time_series_batches(dataset_train, overlap=2, batch_sizes=[7, 8, 9]):
    """
    Plots the training time series divided into batches, with vertical lines separating each batch.

    Args:
        dataset_train (list of tuples): The training dataset represented as a list of (timestamp, value) pairs.
        overlap (int, optional): The number of overlapping samples between consecutive batches. Default is 2.
        batch_sizes (list of int, optional): List of possible batch sizes to test for splitting the dataset. 
                                             Default is [7, 8, 9].
    """
    # print("Inside plot_training_time_series_batches:") # Debug
    # print("split_dataset_with_best_batch_size:", split_dataset_with_best_batch_size)  # Debug

    batches, best_batch_size = preprocessing.split_dataset_with_best_batch_size(dataset_train, overlap, batch_sizes)

    plt.figure(figsize=(10, 6))

    for batch in batches:
        batch_times = [t for t, _ in batch]
        batch_values = [v for _, v in batch]
        plt.plot(batch_times, batch_values, marker='o', linestyle='-', color='goldenrod')

    for i in range(len(batches) - 1):
        last_time_in_batch = batches[i][-1][0]
        plt.axvline(x=last_time_in_batch, color='darkslategray', linestyle='--', alpha=0.8)

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Training Time Series Divided into Batches')
    plt.show()





def plot_benchmark_results(data, anomalies, title="Classical Model Anomaly Detection"):
    """
    Plots the dataset with anomalies highlighted for benchmarking a classical model.

    Args:
        times (list): A list of timestamps for the dataset.
        values (list): A list of values corresponding to each timestamp in the dataset.
        anomalies (list): A list of timestamps identified as anomalies.
        title (str, optional): The title of the plot. Default is "Classical Model Anomaly Detection".

    """
    times = np.array([t for t, _ in data])
    values = np.array([v for _, v in data])

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(times, values, 'bo-', label='Dataset')

    ax.plot(anomalies, values[np.isin(times, anomalies)], 'ro', label='Anomalies')

    ax.set_title(title)
    ax.set_xlabel("Timestamps")
    ax.set_ylabel("Values")
    ax.legend()
    plt.grid(True)
    plt.show()




def execute_batch_processing(
    batches,
    alpha_range=np.linspace(-1, 0, 10),
    selected_position=1,
    model_name=None,
    model_params=None,
    verbose=False,
):
    """
    Esegue una grid su alpha per ciascun batch (con beta=1+alpha), raccoglie i risultati,
    calcola le medie alpha/beta alla posizione 'selected_position' e ritorna anche i dati
    per le curve/diagnostiche.

    >>> NOVITÀ: il modello (model_name/model_params) viene FITTATO UNA SOLA VOLTA
    sull'intera serie (cache globale di L). Nei batch NON si rifitta:
    rank_grid_search riceve L_override (sottovettore di L_full sui timestamp del batch).
    Le distanze restano batch-locali (off_diag_M).

    Returns:
        alpha_mean, beta_mean, alpha_values, beta_values, normalized_rank_values
    """
    all_batch_results = []
    start_total_time = time.time()

    # ------------------------------
    # 1) COSTRUISCI L GLOBALE (una sola volta)
    # ------------------------------
    # Ricostruisci la serie completa nell'ordine naturale dei batch
    X_full = [pt for batch in batches for pt in batch]

    # Usa la funzione di cache globale che hai già inserito in utilities
    # build_global_model_cache(X_full, model_name, model_params)
    try:
        L_full, L_by_ts, _ = build_global_model_cache(X_full, model_name, model_params)
        if verbose:
            print(f"[GLOBAL MODEL] built once: model={model_name}, params={model_params}, len(L_full)={len(L_full)}")
    except Exception as e:
        # Fallback: se non riesce, prosegui senza override (comportamento precedente)
        L_full, L_by_ts = None, None
        if verbose:
            print(f"[GLOBAL MODEL] WARNING: failed to build global cache ({e}). Falling back to per-batch fitting.")

    # ------------------------------
    # 2) LOOP sui batch (NESSUN rifit del modello)
    # ------------------------------
    for i, batch in enumerate(batches, 1):
        if verbose:
            print(f"\nProcessing batch {i}/{len(batches)}...")

        start_batch_time = time.time()

        # Estrai L del batch dai timestamp (se disponibile)
        L_b = None
        if L_by_ts is not None:
            ts_b = [t for (t, _) in batch]
            try:
                L_b = np.array([L_by_ts[t] for t in ts_b], dtype=float)
            except KeyError:
                # Se i timestamp non corrispondono, disattiva l'override per questo batch
                L_b = None
                if verbose:
                    print("[GLOBAL MODEL] WARNING: timestamps mismatch on this batch; skipping L_override.")

        # >>> passiamo il modello (se presente) + L_override (se disponibile) <<<
        try:
            batch_results = training.rank_grid_search(
                batch,
                alpha_range=alpha_range,
                model_name=model_name,
                model_params=model_params,
                L_override=L_b,          # <<< QUI il trucco: L fisso per il batch
            )
        except TypeError:
            # compat vecchie versioni (senza args opzionali)
            batch_results = training.rank_grid_search(batch, alpha_range=alpha_range)

        if not batch_results:
            if verbose:
                print(f"Warning: rank_grid_search did not produce results for batch {i}. Skipping this batch.")
            all_batch_results.append([])
            continue

        # Stampa compatta del BEST del batch
        if verbose:
            best = batch_results[0]
            a  = best.get("alpha")
            b  = best.get("beta")
            ar = best.get("approx_ratio")
            sr = best.get("string_rank", None)
            si = best.get("selected_index", None)
            rd = best.get("selected_rank_in_diag", None)
            msg = f"Batch {i} best: alpha={a:.3f}, beta={b:.3f}, approx_ratio={ar:.6g}"
            if sr is not None: msg += f", string_rank={sr}"
            if si is not None: msg += f", selected_index={si}"
            if rd is not None: msg += f", rank_in_diag={rd}"
            print(msg)

        all_batch_results.append(batch_results)

        end_batch_time = time.time()
        if verbose:
            print(f"Batch {i} completed in {end_batch_time - start_batch_time:.2f} seconds.")

    end_total_time = time.time()
    total_time = end_total_time - start_total_time

    # Filtra solo i batch con risultati
    valid_results = [result for result in all_batch_results if result]

    if valid_results:
        alpha_mean, beta_mean = training.calculate_mean_alpha_beta(valid_results, selected_position)
        if verbose:
            print(f"\nMean Alpha: {alpha_mean}, Mean Beta: {beta_mean}")
    else:
        if verbose:
            print("No valid results were found across all batches.")
        return None, None, [], [], []

    if verbose:
        print(f"Process completed in {total_time:.2f} seconds.")

    alpha_values, beta_values, normalized_rank_values = training.collect_normalized_rank_data(valid_results)
    return alpha_mean, beta_mean, alpha_values, beta_values, normalized_rank_values





def execute_qaoa_on_batches(
    batches,
    model_name='cubic',
    model_params=None,
    alpha_mean=None,
    beta_mean=None,
    verbose: bool = True,
):
    """
    Final run:
      - USA un'unica L globale (fittata UNA volta sull'intera serie) per costruire M in ogni batch.
      - Per ogni batch: M_override = diag(L_batch) + off_diag_M(batch).
      - Risolve il QUBO con solve_qubo() dell'istanza AD_QAOA, ma monkey-patchiamo matrix_M()
        per restituire M_override (così non tocchiamo AD_QAOA_ext.py).
      - Ritorna i centri con raggio come prima.

    Returns:
        unique_centers_with_radii: [ ((ts,val), radius), ... ]
    """
    import types
    import numpy as np
    import pandas as pd
    from AD_QAOA_ext import AD_QAOA

    print("\nInitializing the final run...")

    if model_params is None:
        model_params = {}

    # ------------------------------------------------------------------
    # 1) COSTRUISCI L GLOBALE una volta (sull'intera serie concatenata)
    # ------------------------------------------------------------------
    X_full = [pt for batch in batches for pt in batch]
    try:
        L_full, L_by_ts, _ = build_global_model_cache(X_full, model_name, model_params)
        if verbose:
            print(f"[GLOBAL MODEL] built once: model={model_name}, params={model_params}, len(L_full)={len(L_full)}")
    except Exception as e:
        L_full, L_by_ts = None, None
        print(f"[GLOBAL MODEL] WARNING: failed to build global cache ({e}). Falling back to per-batch M.")

    all_centers_with_radii = []

    for i, batch in enumerate(batches, 1):
        print(f"\nExecuting QAOA model on batch {i}/{len(batches)}...")

        ad_qaoa = AD_QAOA(
            X=batch,
            model_name=model_name,
            model_params=model_params,
            radius_adjustment=True,
            alpha=alpha_mean,
            beta=beta_mean,
        )

        # --------------------------------------------------------------
        # 2) Costruisci M_override = diag(L_batch) + off_diag_M(batch)
        # --------------------------------------------------------------
        M_override = None
        if L_by_ts is not None:
            try:
                ts_b = [t for (t, _) in batch]
                L_b = np.array([L_by_ts[t] for t in ts_b], dtype=float)
                Q_b = ad_qaoa.off_diag_M(batch)
                M_override = np.diag(L_b) + Q_b
            except Exception as e:
                if verbose:
                    print(f"[GLOBAL MODEL] WARNING: cannot assemble M_override for batch {i} ({e}). Using default M.")
                M_override = None

        # --------------------------------------------------------------
        # 3) Monkey-patch: forza ad usare M_override dentro solve_qubo()
        # --------------------------------------------------------------
        if M_override is not None:
            def _matrix_M_override(self):
                return M_override
            # bind come metodo dell'istanza
            ad_qaoa.matrix_M = types.MethodType(_matrix_M_override, ad_qaoa)

        # Ora tutte le chiamate che usano matrix_M() (solve_qubo, cost_function, ecc.)
        # vedranno M_override (se presente).

        # ----------------- solve & diagnostica -----------------
        top_n_states, qaoa_solution = ad_qaoa.solve_qubo()
        print(f"QAOA solution for batch {i}: {top_n_states}")

        # usa la stessa M della risoluzione per i costi/ratio
        if M_override is None:
            M_used = ad_qaoa.matrix_M()
        else:
            M_used = M_override

        df = pd.DataFrame(M_used).round(3)

        qaoa_cost = ad_qaoa.cost_function(M_used, np.array(top_n_states[0]))
        best_state, classical_sol = ad_qaoa.find_min_cost(M_used)
        approximation_ratio = qaoa_cost / classical_sol
        print(f"Batch {i} - QAOA cost: {qaoa_cost}, Classical optimal solution: {classical_sol}, Approximation ratio: {approximation_ratio}")

        # ----------------- centri + raggio -----------------
        batch_centers_with_radii = ad_qaoa.associate_centers_with_radius()
        all_centers_with_radii.extend(batch_centers_with_radii)
        print(f"Batch {i} completed. Saved centers with radii: {batch_centers_with_radii}")

    # ----------------- merge univoco per timestamp -----------------
    unique_centers_with_radii = []
    seen_timestamps = set()
    for center_with_radius in all_centers_with_radii:
        timestamp = center_with_radius[0][0]
        if timestamp not in seen_timestamps:
            unique_centers_with_radii.append(center_with_radius)
            seen_timestamps.add(timestamp)

    print("\nResulting Set Coverage:")
    for j, centers_radii in enumerate(unique_centers_with_radii):
        print(f"Center {j}: {centers_radii}")

    return unique_centers_with_radii







def build_global_model_cache(X_full, model_name: str, model_params: dict | None = None):
    """
    Fit UNA VOLTA il modello globale sull'intera serie X_full e calcola L_full
    con la stessa trasformazione usata in AD_QAOA.diag_M. Restituisce:
      - L_full: np.ndarray in ordine di X_full
      - L_by_ts: dict {timestamp -> L_value} per recuperare L nei batch
      - meta: info del modello
    """
    from AD_QAOA_ext import AD_QAOA
    ad_global = AD_QAOA(
        X=X_full, alpha=0.0, beta=0.0,          # i parametri non importano qui
        model_name=model_name, model_params=(model_params or {}),
        radius_adjustment=False
    )
    L_full = ad_global.diag_M(X_full)           # <<— computa L con il modello globale
    ts_full = [t for (t, _) in X_full]
    L_by_ts = {t: float(L_full[i]) for i, t in enumerate(ts_full)}
    meta = {"model_name": model_name, "model_params": (model_params or {})}
    return L_full, L_by_ts, meta


















##################################################### da qui in giù multivariato #####################################################



def build_global_model_cache_mv(
    X_full_mv,
    model_name: str,
    model_params: dict | None = None,
    transform: str = "rational",
    rho: float = 1.0,
    lambda_global: float = 0.0,
):
    """
    Fit UNA VOLTA il modello globale sull'intera serie multivariata X_full_mv
    e calcola L_flat_full con diag_M_mv.

    Returns:
      - L_flat_full: np.ndarray shape (n_full*C,)
      - L_by_ts_vec: dict {timestamp -> np.ndarray shape (C,)}  (L per-canale)
      - meta: info varie (C, transform, rho, lambda_global, modello)
    """
    from AD_QAOA_ext import AD_QAOA

    if model_params is None:
        model_params = {}

    n_full = len(X_full_mv)
    if n_full == 0:
        return np.array([]), {}, {
            "model_name": model_name,
            "model_params": model_params,
            "C": 0,
            "transform": transform,
            "rho": rho,
            "lambda_global": lambda_global,
        }

    v0 = np.asarray(X_full_mv[0][1])
    C = int(v0.shape[0]) if v0.ndim > 0 else 1

    ad_global = AD_QAOA(
        X=X_full_mv,
        alpha=0.0, beta=0.0,  # non importa qui
        model_name=model_name,
        model_params=model_params,
        radius_adjustment=False,
    )

    L_flat_full = ad_global.diag_M_mv(
        X_full_mv, transform=transform, rho=rho, lambda_global=lambda_global
    )
    L_flat_full = np.asarray(L_flat_full, dtype=float).reshape(-1)

    expected = n_full * C
    if L_flat_full.shape[0] != expected:
        raise RuntimeError(
            f"diag_M_mv returned length {L_flat_full.shape[0]} != n_full*C={expected} "
            f"(n_full={n_full}, C={C})"
        )

    ts_full = [int(t) for (t, _) in X_full_mv]
    L_by_ts_vec = {}
    for i, t in enumerate(ts_full):
        L_by_ts_vec[t] = L_flat_full[i * C:(i + 1) * C].copy()

    meta = {
        "model_name": model_name,
        "model_params": model_params,
        "C": C,
        "transform": transform,
        "rho": rho,
        "lambda_global": lambda_global,
    }
    return L_flat_full, L_by_ts_vec, meta


def execute_batch_processing_mv(
    batches_mv,
    alpha_range=np.linspace(-1, 0, 10),
    selected_position: int = 0,
    model_name: str | None = None,
    model_params: dict | None = None,
    transform: str = "rational",
    rho: float = 1.0,
    lambda_global: float = 0.0,
    verbose: bool = False,
):
    """
    MV version:
    - Costruisce L globale UNA VOLTA su X_full_mv (diag_M_mv)
    - Per ogni batch: passa L_override_flat (n_batch*C) a training.rank_grid_search_mv
      (così NO refit nei batch)
    - Ritorna media alpha/beta e diagnostiche come prima.

    Returns:
      alpha_mean, beta_mean, alpha_values, beta_values, normalized_rank_values
    """
    from functions import AD_training as training
  # usa il tuo modulo training

    all_batch_results = []
    start_total_time = time.time()

    # 1) Serie completa (nell'ordine naturale)
    X_full_mv = [pt for batch in batches_mv for pt in batch]

    # 2) Cache globale L
    try:
        _, L_by_ts_vec, meta = build_global_model_cache_mv(
            X_full_mv,
            model_name=model_name,
            model_params=(model_params or {}),
            transform=transform,
            rho=rho,
            lambda_global=lambda_global,
        )
        C = int(meta["C"])
        if verbose:
            print(f"[GLOBAL MV MODEL] built once: model={model_name}, params={model_params}, C={C}, n_full={len(X_full_mv)}")
    except Exception as e:
        L_by_ts_vec = None
        C = None
        if verbose:
            print(f"[GLOBAL MV MODEL] WARNING: failed to build global cache ({e}). Falling back to per-batch diag_M_mv.")

    # 3) Loop batch
    for i, batch_mv in enumerate(batches_mv, 1):
        if verbose:
            print(f"\nProcessing MV batch {i}/{len(batches_mv)}...")

        start_batch_time = time.time()

        # Estrai L_flat del batch dalla cache (se disponibile)
        L_override_flat = None
        if L_by_ts_vec is not None:
            ts_b = [int(t) for (t, _) in batch_mv]
            try:
                # flatten coerente con k = i*C + c
                L_override_flat = np.concatenate([L_by_ts_vec[t] for t in ts_b], axis=0).astype(float)
            except KeyError:
                L_override_flat = None
                if verbose:
                    print("[GLOBAL MV MODEL] WARNING: timestamps mismatch in this batch; skipping L_override_flat.")

        # rank grid search MV
        batch_results = training.rank_grid_search_mv(
            batch_mv,
            alpha_range=alpha_range,
            model_name=model_name,
            model_params=model_params,
            L_override_flat=L_override_flat,
            transform=transform,
            rho=rho,
            lambda_global=lambda_global,
        )

        if not batch_results:
            if verbose:
                print(f"Warning: rank_grid_search_mv did not produce results for batch {i}. Skipping.")
            all_batch_results.append([])
            continue

        if verbose:
            best = batch_results[0]
            print(
                f"Batch {i} best (MAX ones): alpha={best['alpha']:.6g}, beta={best['beta']:.6g}, "
                f"n_ones={best.get('n_ones', None)}"
            )

        all_batch_results.append(batch_results)

        if verbose:
            print(f"Batch {i} completed in {time.time() - start_batch_time:.2f} seconds.")

    # 4) Media alpha/beta (come prima)
    valid_results = [r for r in all_batch_results if r]
    if not valid_results:
        if verbose:
            print("No valid results were found across all batches.")
        return None, None, [], [], []

    alpha_mean, beta_mean = training.calculate_mean_alpha_beta(valid_results, selected_position)

    if verbose:
        print(f"\nMean Alpha: {alpha_mean}, Mean Beta: {beta_mean}")
        print(f"Process completed in {time.time() - start_total_time:.2f} seconds.")

    alpha_values, beta_values, normalized_rank_values = training.collect_normalized_rank_data(valid_results)
    return alpha_mean, beta_mean, alpha_values, beta_values, normalized_rank_values
















# def execute_qaoa_on_batches_mv(
#     batches_mv,
#     model_name: str = "cubic",
#     model_params: dict | None = None,
#     alpha_mean: float | None = None,
#     beta_mean: float | None = None,
#     transform: str = "rational",
#     rho: float = 1.0,
#     lambda_global: float = 0.0,
#     verbose: bool = True,
# ):
#     """
#     MV final run:
#       - Costruisce L globale UNA volta (diag_M_mv su serie completa)
#       - Per ogni batch:
#           M_override = diag(L_override_flat) + off_diag_M_mv(batch)
#           monkey-patch matrix_M_mv -> M_override
#           solve_qubo_mv
#           centers_storage_mv -> centers (t, c, y_c)
#       - Deduplica finale per (t,c)

#     Returns:
#       unique_centers: list of (t, c, y_c)
#     """
#     import types
#     import numpy as np
#     from AD_QAOA_ext import AD_QAOA

#     if model_params is None:
#         model_params = {}

#     print("\nInitializing the MV final run...")

#     # 1) Cache globale L (una volta)
#     X_full_mv = [pt for batch in batches_mv for pt in batch]
#     try:
#         _, L_by_ts_vec, meta = build_global_model_cache_mv(
#             X_full_mv,
#             model_name=model_name,
#             model_params=model_params,
#             transform=transform,
#             rho=rho,
#             lambda_global=lambda_global,
#         )
#         C = int(meta["C"])
#         if verbose:
#             print(f"[GLOBAL MV MODEL] built once: model={model_name}, params={model_params}, C={C}, n_full={len(X_full_mv)}")
#     except Exception as e:
#         L_by_ts_vec = None
#         C = None
#         print(f"[GLOBAL MV MODEL] WARNING: failed to build global cache ({e}). Falling back to per-batch M.")

#     all_centers = []

#     for i, batch_mv in enumerate(batches_mv, 1):
#         print(f"\nExecuting MV QAOA on batch {i}/{len(batches_mv)}...")

#         ad_qaoa = AD_QAOA(
#             X=batch_mv,
#             model_name=model_name,
#             model_params=model_params,
#             radius_adjustment=True,   # se ti serve, ma qui i centri MV non hanno raggio
#             alpha=alpha_mean,
#             beta=beta_mean,
#         )

#         M_override = None
#         if L_by_ts_vec is not None:
#             try:
#                 ts_b = [int(t) for (t, _) in batch_mv]
#                 L_override_flat = np.concatenate([L_by_ts_vec[t] for t in ts_b], axis=0).astype(float)
#                 Q_mv = ad_qaoa.off_diag_M_mv(batch_mv)
#                 M_override = np.diag(L_override_flat) + Q_mv
#             except Exception as e:
#                 if verbose:
#                     print(f"[GLOBAL MV MODEL] WARNING: cannot assemble M_override for batch {i} ({e}). Using default MV M.")
#                 M_override = None

#         # 2) Monkey-patch MV matrix (se override disponibile)
#         if M_override is not None:
#             def _matrix_M_mv_override(self, *args, **kwargs):
#                 return M_override
#             ad_qaoa.matrix_M_mv = types.MethodType(_matrix_M_mv_override, ad_qaoa)

#         # 3) Solve MV
#         top_states, _ = ad_qaoa.solve_qubo_mv(
#             batch_mv, transform=transform, rho=rho, lambda_global=lambda_global
#         )
#         best_state = np.array(top_states[0], dtype=int)
#         print(f"MV QAOA top state for batch {i}: ones={int(best_state.sum())}")

#         # 4) Centri MV (t, c, y_c)
#         batch_centers = ad_qaoa.centers_storage_mv(
#             batch_mv,
#             state=best_state.tolist(),
#             threshold=0.5,
#             transform=transform,
#             rho=rho,
#             lambda_global=lambda_global,
#         )
#         all_centers.extend(batch_centers)

#         if verbose:
#             print(f"Batch {i} centers saved: {batch_centers}")

#     # 5) Deduplica per (t,c)
#     unique = []
#     seen = set()
#     for (t, c, y) in all_centers:
#         key = (int(t), int(c))
#         if key not in seen:
#             unique.append((int(t), int(c), float(y)))
#             seen.add(key)

#     if verbose:
#         print("\nResulting MV selected centers (unique by (t,c)):")
#         for j, ctr in enumerate(unique):
#             print(f"Center {j}: (t={ctr[0]}, c={ctr[1]}, y={ctr[2]:.6g})")

#     return unique




























def execute_qaoa_on_batches_mv(
    batches_mv,
    model_name: str = "cubic",
    model_params: dict | None = None,
    alpha_mean: float | None = None,
    beta_mean: float | None = None,
    transform: str = "rational",
    rho: float = 1.0,
    lambda_global: float = 0.0,
    verbose: bool = True,
):
    """
    Keeps the current working behavior:
      - Builds GLOBAL L once (diag_M_mv on full series) -> used as diagonal override per batch.
      - For each batch: M_override = diag(L_override_flat) + off_diag_M_mv(batch)
      - solve_qubo_mv -> best_state
      - centers_storage_mv -> batch centers (t, c, y_c)

    NEW:
      - For each batch, builds per-channel monovariate series Xc_batch and uses the SAME
        monovariate radius logic (associate_centers_with_radius) to get a radius PER CENTER.
      - Accumulates and deduplicates per (t,c), preserving first occurrence order (batch order).
      - Returns:
          unique_centers: list[(t, c, y_c)]
          centers_with_radii_by_channel: dict[c] -> list[ ((t, y_c), radius) ]
          unique_centers_with_radii_mv: list[ ((t, c, y_c), radius) ]


    """
    import types
    import numpy as np
    from collections import defaultdict
    from AD_QAOA_ext import AD_QAOA

    if model_params is None:
        model_params = {}

    print("\nInitializing the MV final run...")



    X_full_mv = [pt for batch in batches_mv for pt in batch]
    try:
        _, L_by_ts_vec, meta = build_global_model_cache_mv(
            X_full_mv,
            model_name=model_name,
            model_params=model_params,
            transform=transform,
            rho=rho,
            lambda_global=lambda_global,
        )
        C_global = int(meta["C"])
        if verbose:
            print(
                f"[GLOBAL MV MODEL] built once: model={model_name}, params={model_params}, "
                f"C={C_global}, n_full={len(X_full_mv)}"
            )
    except Exception as e:
        L_by_ts_vec = None
        C_global = None
        print(f"[GLOBAL MV MODEL] WARNING: failed to build global cache ({e}). Falling back to per-batch M.")

    all_centers = []


    all_centers_with_radii_mv = []

    for i, batch_mv in enumerate(batches_mv, 1):
        print(f"\nExecuting MV QAOA on batch {i}/{len(batches_mv)}...")

        ad_qaoa = AD_QAOA(
            X=batch_mv,
            model_name=model_name,
            model_params=model_params,
            radius_adjustment=True,
            alpha=alpha_mean,
            beta=beta_mean,
        )


        M_override = None
        if L_by_ts_vec is not None:
            try:
                ts_b = [int(t) for (t, _) in batch_mv]
                # L_override_flat in ordine coerente con k=i*C + c
                L_override_flat = np.concatenate([L_by_ts_vec[t] for t in ts_b], axis=0).astype(float)
                Q_mv = ad_qaoa.off_diag_M_mv(batch_mv)
                M_override = np.diag(L_override_flat) + Q_mv
            except Exception as e:
                if verbose:
                    print(
                        f"[GLOBAL MV MODEL] WARNING: cannot assemble M_override for batch {i} ({e}). "
                        f"Using default MV M."
                    )
                M_override = None

        if M_override is not None:
            def _matrix_M_mv_override(self, *args, **kwargs):
                return M_override
            ad_qaoa.matrix_M_mv = types.MethodType(_matrix_M_mv_override, ad_qaoa)



         
        # Solve qubo
        top_states, _ = ad_qaoa.solve_qubo_mv(
            batch_mv, transform=transform, rho=rho, lambda_global=lambda_global
        )
        best_state = np.array(top_states[0], dtype=int)
        print(f"MV QAOA top state for batch {i}: ones={int(best_state.sum())}")





        batch_centers = ad_qaoa.centers_storage_mv(
            batch_mv,
            state=best_state.tolist(),
            threshold=0.5,
            transform=transform,
            rho=rho,
            lambda_global=lambda_global,
        )
        all_centers.extend(batch_centers)

        if verbose:
            print(f"Batch {i} centers saved: {batch_centers}")





        if len(batch_centers) == 0:
            continue

        v0 = np.asarray(batch_mv[0][1])
        C = int(v0.shape[0]) if v0.ndim > 0 else 1


        Xc_batch_map = {c: [] for c in range(C)}
        for t0, vec in batch_mv:
            vec = np.asarray(vec, dtype=float)
            for c in range(C):
                Xc_batch_map[c].append((int(t0), float(vec[c])))


        centers_c_map = defaultdict(list)
        for (t0, c0, y0) in batch_centers:
            centers_c_map[int(c0)].append((int(t0), float(y0)))


        for c in range(C):
            centers_c_batch = centers_c_map.get(c, [])
            if not centers_c_batch:
                continue

            Xc_batch = Xc_batch_map[c]

            ad_c = AD_QAOA(
                X=Xc_batch,
                model_name=model_name,
                model_params=model_params,
                radius_adjustment=True,

                alpha=0.0,
                beta=0.0,
            )


            def _centers_storage_override(self, state=None, threshold=0.5):
                return centers_c_batch

            ad_c.centers_storage = types.MethodType(_centers_storage_override, ad_c)

            # This is the SAME monovariate routine you used before:
            # returns [((t,y), radius), ...]
            centers_wr_c = ad_c.associate_centers_with_radius()


            for ((t1, y1), r1) in centers_wr_c:
                all_centers_with_radii_mv.append(((int(t1), int(c), float(y1)), float(r1)))


    unique = []
    seen_tc = set()
    for (t, c, y) in all_centers:
        key = (int(t), int(c))
        if key not in seen_tc:
            unique.append((int(t), int(c), float(y)))
            seen_tc.add(key)

    if verbose:
        print("\nResulting MV selected centers (unique by (t,c)):")
        for j, ctr in enumerate(unique):
            print(f"Center {j}: (t={ctr[0]}, c={ctr[1]}, y={ctr[2]:.6g})")


    # Deduplica centers_with_radii per (t,c), keep first (batch-order)
    
    
    unique_centers_with_radii_mv = []
    seen_tc2 = set()
    for (tcY, r) in all_centers_with_radii_mv:
        t, c, y = tcY
        key = (int(t), int(c))
        if key not in seen_tc2:
            unique_centers_with_radii_mv.append(((int(t), int(c), float(y)), float(r)))
            seen_tc2.add(key)




    #  Build per-channel lists like monovariate: dict[c] -> [((t,y), r), ...]
    centers_with_radii_by_channel = defaultdict(list)
    for ((t, c, y), r) in unique_centers_with_radii_mv:
        centers_with_radii_by_channel[int(c)].append(((int(t), float(y)), float(r)))



    for c in list(centers_with_radii_by_channel.keys()):
        centers_with_radii_by_channel[c].sort(key=lambda item: item[0][0])

    if verbose:
        print("\nResulting MV centers WITH radii (unique by (t,c)):")
        for j, (tcY, r) in enumerate(unique_centers_with_radii_mv):
            t, c, y = tcY
            print(f"Center {j}: (t={t}, c={c}, y={y:.6g}), r={r:.6g}")


    return unique, dict(centers_with_radii_by_channel), unique_centers_with_radii_mv







































def detection_stats_from_timestamps(
    labels_by_ts,
    detected_anom_ts,
    verbose=True,
):
    """
    Calcola metriche di anomaly detection a livello timestamp.

    Args:
        labels_by_ts: dict[int -> bool]
            True = anomalia reale
            False = normale
        detected_anom_ts: iterable[int]
            timestamp rilevati come anomali dal detector
        verbose: bool

    Returns:
        stats: dict
    """

    detected_set = set(int(t) for t in detected_anom_ts)

    TP = FP = TN = FN = 0

    for t, is_anom in labels_by_ts.items():
        detected = t in detected_set

        if is_anom and detected:
            TP += 1
        elif not is_anom and detected:
            FP += 1
        elif is_anom and not detected:
            FN += 1
        elif not is_anom and not detected:
            TN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy  = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    stats = {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "specificity": specificity,
    }

    if verbose:
        print("\n=== Detection statistics (timestamp-level) ===")
        print(f"TP: {TP} | FP: {FP} | TN: {TN} | FN: {FN}")
        print(f"Precision   : {precision:.4f}")
        print(f"Recall      : {recall:.4f}")
        print(f"F1-score    : {f1:.4f}")
        print(f"Accuracy    : {accuracy:.4f}")
        print(f"Specificity : {specificity:.4f}")
        print("=============================================\n")

    return stats


































































#pattern recognition


def plot_series_with_suspicious_windows(
    series,
    batches,
    res_train,
    quantile=0.65,
    title="Training series with anomalous windows"
):
    """
    series   : 1D array di valori OPPURE 2D (N,2) tipo [time, value].
               In caso 2D usa l'ultima colonna come valore.
    batches  : lista di batch [(t, value), ...]
    res_train: lista di dict con chiavi "energy" e "batch_id"
    quantile : quantile sulle energie per definire la soglia
    """

    arr = np.asarray(series)
    if arr.ndim == 1:
        y = arr
    elif arr.ndim == 2:
        # usiamo l'ultima colonna come valore (tipicamente quella scalata)
        y = arr[:, -1]
    else:
        raise ValueError("series deve essere 1D o 2D")

    x = np.arange(len(y))

    energies = np.array([r["energy"] for r in res_train])
    T = np.quantile(energies, quantile)
    suspicious_batches = [r["batch_id"] for r in res_train if r["energy"] >= T]  #cambiato da minore uguale

    print(f"Energy threshold (q={quantile}): {T:.4f}")
    print("Suspicious batch IDs:", suspicious_batches)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, marker="o", linestyle="-", color="tab:orange", label="Train (scaled)")

    # sfondo rosa per i batch sospetti
    for r, batch in zip(res_train, batches):
        if r["energy"] >= T: #cambiato d aminore uguale
            start_idx = batch[0][0]      # primo indice tempo del batch
            end_idx   = batch[-1][0] + 1 # +1 per includere l'ultimo punto
            ax.axvspan(start_idx, end_idx, color="pink", alpha=0.25)

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    plt.show()













def solve_qubo_single_batch_v2(
    batch,
    model_name="linear",
    model_params=None,
    alpha=-0.2,
    beta=0.8,
):
    """
    batch: lista di (t, value), es:
           [(0, 1.22), (1, 1.27), ..., (10, 1.98)]

    Ritorna:
        bits  -> bitstring ottimale (il migliore fra i top_n)
        E     -> energia minima fval
    """

    ad = AD_QAOA(
        batch,                 # <-- primo argomento: i dati
        model_name=model_name,
        model_params=model_params,
        alpha=alpha,
        beta=beta,
    )

    top_states_v2, E_v2 = ad.solve_qubo_with_energy_v2()
    best_bits_v2 = top_states_v2[0]   # il primo è quello con energia minima
    

    return best_bits_v2, E_v2


def solve_qubo_on_batches_v2(
    batches,
    model_name="linear",
    model_params=None,
    alpha=-0.2,
    beta=0.8,
):
    """
    batches: lista di batch, ciascuno nel formato [(t, value), ...]
    """

    results_v2 = []
    for i, batch in enumerate(batches):
        bits, E = solve_qubo_single_batch_v2(
            batch,
            model_name=model_name,
            model_params=model_params,
            alpha=alpha,
            beta=beta,
        )
        results_v2.append({
            "batch_id": i,
            "size": len(batch),
            "bits": bits,
            "energy": E,
        })
        print(f"[Batch {i}] size={len(batch)} → Energy={E:.4f}, bits={bits}")

    return results_v2






















import numpy as np
import matplotlib.pyplot as plt

def vote_heatmap_from_runs(all_batches, all_res, q_energy=0.7, vote_mode="binary"):
    """
    all_batches: list of runs -> run_batches = [batch0, batch1, ...]
    all_res:     list of runs -> run_res     = [res0, res1, ...] con res_i dict {energy,bits,...}
    q_energy: quantile per-run per definire batch sospetti (energia >= soglia)
    vote_mode: "binary" (1 voto) oppure "weighted" (peso = max(0, E - T))

    Returns:
      heatmap: np.array length N con voti/pesi
      run_thresholds: lista soglie per run
      suspicious_by_run: lista di liste di batch_id sospetti
    """
    # N = max timestamp +1 (assumiamo 0..N-1)
    N = 1 + max(batch[-1][0] for run in all_batches for batch in run if len(batch) > 0)
    heatmap = np.zeros(N, dtype=float)

    run_thresholds = []
    suspicious_by_run = []

    for run_batches, run_res in zip(all_batches, all_res):
        energies = np.array([r["energy"] for r in run_res], dtype=float)
        T = float(np.quantile(energies, q_energy))
        run_thresholds.append(T)

        suspicious_ids = []
        for batch, r in zip(run_batches, run_res):
            E = float(r["energy"])
            if E >= T:  # energia più alta (meno negativa) => più sospetto
                suspicious_ids.append(r["batch_id"])

                w = 1.0
                if vote_mode == "weighted":
                    w = max(0.0, T-E)
                    if w == 0.0:
                        w = 1.0

                for (t, _) in batch:
                    heatmap[t] += w

        suspicious_by_run.append(suspicious_ids)

    return heatmap, run_thresholds, suspicious_by_run


def segments_from_heatmap(heatmap, min_votes=2, min_len=1, gap_merge=0):
    """
    heatmap -> boolean mask -> segmenti.
    min_votes: soglia sui voti (es. 2 su 4 run)
    min_len: elimina segmenti troppo corti
    gap_merge: se >0, unisce segmenti separati da gap <= gap_merge

    Returns:
      segments: list[(start,end)] end esclusivo
      mask: boolean array
    """
    mask = heatmap >= min_votes

    # estrai segmenti grezzi
    segs = []
    in_seg = False
    s = 0
    for i, m in enumerate(mask):
        if m and not in_seg:
            in_seg = True
            s = i
        elif (not m) and in_seg:
            in_seg = False
            segs.append((s, i))
    if in_seg:
        segs.append((s, len(mask)))

    # filtra per lunghezza
    segs = [(a,b) for (a,b) in segs if (b-a) >= min_len]

    # merge gap piccoli
    if gap_merge > 0 and len(segs) > 1:
        merged = [segs[0]]
        for (a,b) in segs[1:]:
            a0,b0 = merged[-1]
            if a - b0 <= gap_merge:
                merged[-1] = (a0, b)
            else:
                merged.append((a,b))
        segs = merged

    return segs, mask


def plot_series_with_refined_segments(series, segments, heatmap=None, title="Refined anomalous segments (multi-offset voting)"):
    y = np.array([v for _, v in series], dtype=float)
    x = np.arange(len(y))

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(x, y, marker="o", linewidth=1.5, color="tab:orange", label="Train (scaled)")

    for (s,e) in segments:
        ax.axvspan(s, e, color="pink", alpha=0.25)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True)
    ax.legend()
    plt.show()

    if heatmap is not None:
        plt.figure(figsize=(10,2.2))
        plt.plot(x, heatmap[:len(x)])
        plt.title("Heatmap votes / score per timestamp")
        plt.xlabel("Time")
        plt.ylabel("Votes/Score")
        plt.grid(True)
        plt.show()























