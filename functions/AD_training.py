"Training for alpha and beta functions"



import re
import numpy as np
import math
import pandas as pd
from typing import Tuple, List, Dict
from statistics import mean
import random
import itertools
from AD_QAOA_ext import AD_QAOA 
from matplotlib import pyplot as plt

try:
    from scipy.signal import savgol_filter
except Exception:
    savgol_filter = None

try:
    from scipy.interpolate import UnivariateSpline
except Exception:
    UnivariateSpline = None

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
except Exception:
    _lowess = None


def grid_search_alpha_beta(X, alpha_range=np.linspace(-1, 0, 10)):
    """
    First grid-search approach for single batch (or small dataset) application and testing. 
    Performs a grid search over a range of alpha values to find the optimal (alpha, beta) 
    pair for minimizing the QAOA cost function on a given dataset.

    Args:
        X (list of tuples): The dataset represented as a list of (timestamp, value) pairs.
        alpha_range (np.ndarray, optional): A range of alpha values to search. Default is a linearly spaced 
                                            range from -1 to 0 with 10 values.

    Returns:
        best_alpha (float): The alpha value that results in the minimum cost.
        best_beta (float): The corresponding beta value (1 + alpha) for the optimal alpha.
        best_cost (float): The minimum cost value achieved.
        best_state (np.ndarray): The QAOA state that achieves the best cost.
    """
    best_alpha = None
    best_beta = None
    best_cost = float('inf')
    best_state = None

    print(f"Dataset:\n{X}\n")

    for alpha in alpha_range:
        beta = 1 + alpha

        ad_qaoa = AD_QAOA(X, alpha=alpha, beta=beta)

        states, _ = ad_qaoa.solve_qubo()
        qaoa_state = np.array(states[0])

        M = ad_qaoa.matrix_M()

        df = pd.DataFrame(M)
        df_rounded = df.round(3) 
        print(f"Matrix M for Alpha = {alpha} and Beta = {beta}:\n{df_rounded}\n")

        qaoa_cost = ad_qaoa.cost_function(M, qaoa_state)

        best_classical_state, best_classical_cost = ad_qaoa.find_min_cost(M)

        print(f"Alpha: {alpha}, Beta: {beta}")
        print(f"QAOA State: {qaoa_state}, QAOA Cost: {qaoa_cost}")
        num_ones = np.sum(qaoa_state)
        total_length = len(qaoa_state)
        percentage_ones = (num_ones / total_length)
        print(f"Percentage of '1' in QAOA state: {percentage_ones}")
        print(f"Approximation ratio: {qaoa_cost / best_classical_cost}")
        print(f"Classical Best State: {best_classical_state}, Classical Max Cost: {best_classical_cost}\n")

        if qaoa_cost < best_cost:
            best_cost = qaoa_cost
            best_alpha = alpha
            best_beta = beta
            best_state = qaoa_state

    return best_alpha, best_beta, best_cost, best_state




def rank_grid_search(
    X,
    alpha_range=np.linspace(-1, 0, 10),
    model_name: str | None = None,
    model_params: dict | None = None,
    L_override: np.ndarray | None = None,   # <<< NEW: se presente, evita rifit per batch
):
    """
    Per un batch X, scansiona alpha (beta=1+alpha), costruisce il QUBO e valuta QAOA.
    Se L_override è fornito, USA quello per la diagonale di M (modello globale fisso)
    e calcola solo la parte off-diagonale batch-locale. Per evitare rifit interni,
    monkey-patchiamo matrix_M() dell'istanza AD_QAOA a restituire proprio M_override.
    """
    from AD_QAOA_ext import AD_QAOA  # local import to evitare cicli
    import types
    import numpy as np

    model_name, model_params = normalize_model_spec(model_name, model_params)
    results = []

    for alpha in alpha_range:
        beta = 1 + alpha
        kwargs = {}
        if model_name is not None:
            kwargs["model_name"] = model_name
            kwargs["model_params"] = model_params or {}

        ad_qaoa = AD_QAOA(X, alpha=alpha, beta=beta, **kwargs)

        # --- Costruisci M (con o senza override) ---
        if L_override is None:
            # Vecchio comportamento: ricava diag e off-diag internamente
            M = ad_qaoa.matrix_M()
        else:
            # Nuovo: usa L_override per la diagonale (modello globale fisso)
            L_override = np.asarray(L_override, dtype=float)
            if L_override.shape[0] != len(X):
                raise ValueError(f"L_override length {L_override.shape[0]} != batch length {len(X)}")
            Q = ad_qaoa.off_diag_M(X)            # distanze batch-locali
            M = np.diag(L_override) + Q

            # >>> Monkey-patch: forza TUTTO (solve_qubo, ecc.) a usare M qui costruita
            def _matrix_M_override(self):
                return M
            ad_qaoa.matrix_M = types.MethodType(_matrix_M_override, ad_qaoa)

        # --- QAOA sul M ottenuto ---
        states, _ = ad_qaoa.solve_qubo()
        qaoa_state = np.array(states[0], dtype=int)

        # usa lo stesso M per i costi/diagnostica
        qaoa_cost = float(ad_qaoa.cost_function(M, qaoa_state))
        best_state, classical_cost = ad_qaoa.find_min_cost(M)
        classical_cost = float(classical_cost)
        approx_ratio = float(qaoa_cost / classical_cost) if classical_cost != 0 else np.nan

        # Rank diagnostic = posizione dell'indice selezionato in diag(M) (desc)
        diag = np.diag(M).astype(float)
        sel = np.where(qaoa_state == 1)[0]
        idx = int(sel[0]) if sel.size > 0 else int(np.argmax(qaoa_state))
        order = np.argsort(-diag)
        rank_map = {int(i): r for r, i in enumerate(order)}
        rank_in_diag = int(rank_map.get(idx, len(diag)-1))

        results.append({
            "alpha": float(alpha),
            "beta": float(beta),
            "qaoa_state": qaoa_state,
            "qaoa_cost": qaoa_cost,
            "classical_cost": classical_cost,
            "approx_ratio": approx_ratio,
            "selected_index": idx,
            "selected_rank_in_diag": rank_in_diag,
        })

    # Ordina best-first (come prima)
    results.sort(key=lambda r: (r["approx_ratio"], r["qaoa_cost"]))
    return results





def collect_normalized_rank_data(all_epoch_results):
    """
    Raccoglie alpha, beta e un 'normalized_rank' robusto:
      - preferisce 'string_rank' se presente (compat vecchia)
      - altrimenti 'selected_rank_in_diag'
      - altrimenti 'selected_index'
      - altrimenti argmax(qaoa_state)
    """
    import numpy as np

    alpha_values, beta_values, normalized_rank_values = [], [], []

    for epoch_results in all_epoch_results:
        for result in epoch_results:
            q = np.array(result.get("qaoa_state", []))
            n = int(len(q)) if len(q) > 0 else 1

            if "string_rank" in result:
                r = int(result["string_rank"])
            elif "selected_rank_in_diag" in result:
                r = int(result["selected_rank_in_diag"])
            elif "selected_index" in result:
                r = int(result["selected_index"])
            else:
                r = int(np.argmax(q)) if q.size else 0

            alpha_values.append(float(result.get("alpha", 0.0)))
            beta_values.append(float(result.get("beta", 0.0)))
            normalized_rank_values.append(float(r) / float(n))

    return alpha_values, beta_values, normalized_rank_values




def calculate_mean_alpha_beta(all_epoch_results, selected_position):
    """
    Calculates the mean alpha and beta values based on a specific position in the ranking.

    Args:
        all_epoch_results (list of lists): A list where each element is a list of dictionaries representing 
                                           results for an epoch. Each dictionary should contain 'alpha' and 'beta'.
        selected_position (int): The ranking position to select for averaging alpha and beta values.

    Returns:
        alpha_mean (float): The mean alpha value for the specified ranking position across epochs.
        beta_mean (float): The mean beta value for the specified ranking position across epochs.
        If no results are found at the specified position, returns (None, None).
    """
    selected_alpha_beta = []

    for epoch_results in all_epoch_results:
        if selected_position - 1 < len(epoch_results):
            result = epoch_results[selected_position - 1]
            selected_alpha_beta.append((result["alpha"], result["beta"]))

    if selected_alpha_beta:
        alpha_mean = np.mean([alpha for alpha, _ in selected_alpha_beta])
        beta_mean = np.mean([beta for _, beta in selected_alpha_beta])
        return alpha_mean, beta_mean
    else:
        print(f"No configurations found at position {selected_position}.")
        return None, None












def _fit_predict_model(times, values, model_name: str, model_params: dict | None = None):
    """
    Fit su (times, values) e predizione sullo stesso asse temporale.
    Ritorna: yhat, k_params (per AIC), details_dict.

    Supporta:
      - "linear" / "quadratic" / "cubic"
      - "moving_average" (param: window)
      - "savgol"         (param: window, polyorder)
      - "spline"         (param: s  OR  num_knots)  -> vedi note
      - "loess"          (param: frac)

    NOTE spline:
      - s  : smoothing spline (UnivariateSpline), più grande = più liscia.
             Accetta anche s="auto" con opzionale "lambda" (default 2.0).
      - num_knots : LSQUnivariateSpline con N nodi interni equispaziati (meno nodi = più liscia).
    """
    import numpy as np

    model_name, model_params = normalize_model_spec(model_name, model_params)
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)

    # --- modelli polinomiali -------------------------------------------------
    if model_name == "linear":
        coefs = np.polyfit(t, y, 1)
        yhat = np.polyval(coefs, t)
        return yhat, 2, {"coefs": coefs}

    if model_name == "quadratic":
        coefs = np.polyfit(t, y, 2)
        yhat = np.polyval(coefs, t)
        return yhat, 3, {"coefs": coefs}

    if model_name == "cubic":
        coefs = np.polyfit(t, y, 3)
        yhat = np.polyval(coefs, t)
        return yhat, 4, {"coefs": coefs}

    # --- moving average ------------------------------------------------------
    if model_name == "moving_average":
        mp = dict(model_params or {})
        w = max(1, int(mp.get("window", 5)))
        pad = w // 2
        ypad = np.pad(y, (pad, pad), mode="edge")
        kernel = np.ones(w) / w
        yhat = np.convolve(ypad, kernel, mode="valid")
        return yhat, 1, {"window": w}

    # --- Savitzky–Golay ------------------------------------------------------
    if model_name == "savgol":
        if savgol_filter is None:
            raise ImportError("scipy.signal.savgol_filter non disponibile.")
        mp = dict(model_params or {})
        w = int(mp.get("window", 11))
        p = int(mp.get("polyorder", 3))
        # finestra dispari e > polyorder
        if w % 2 == 0:
            w += 1
        if w <= p:
            w = p + 2 + (p % 2 == 0)  # garantisce disparità
        yhat = savgol_filter(y, window_length=w, polyorder=p, mode="interp")
        return yhat, p + 1, {"window": w, "polyorder": p}

    # --- Spline (smoothing o con numero di nodi) ------------------------------
    if model_name == "spline":
        try:
            from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
        except Exception as e:
            raise RuntimeError("SciPy è richiesto per i modelli 'spline'.") from e

        mp = dict(model_params or {})
        k = 3  # cubica

        # assicurati che l'asse temporale sia crescente (richiesto da SciPy)
        order = np.argsort(t)
        t_s, y_s = t[order], y[order]
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))

        s = mp.get("s", None)
        num_knots = mp.get("num_knots", None)

        # s="auto": scala con rumore robusto
        if isinstance(s, str) and s.lower() == "auto":
            try:
                sigma_base = _robust_scale_baseline(list(zip(t_s, y_s)))
            except Exception:
                sigma_base = np.median(np.abs(y_s - np.median(y_s))) * 1.4826 + 1e-12
            N = len(y_s)
            lam = float(mp.get("lambda", 2.0))
            s = lam * N * (sigma_base ** 2)

        if num_knots is not None and int(num_knots) >= 2:
            num_knots = int(num_knots)
            tmin, tmax = float(t_s.min()), float(t_s.max())
            knots = np.linspace(tmin, tmax, num_knots + 2)[1:-1]  # interni
            spl = LSQUnivariateSpline(t_s, y_s, t=knots, k=k)
            yhat_s = spl(t_s)
            yhat = yhat_s[inv]
            k_params = len(knots) + k + 1
            return yhat, k_params, {"num_knots": num_knots}

        # smoothing spline (più grande s = più liscia)
        s_val = None if s is None else float(s)
        spl = UnivariateSpline(t_s, y_s, s=s_val, k=k)
        yhat_s = spl(t_s)
        yhat = yhat_s[inv]
        k_params = (len(spl.get_knots()) + k + 1) if hasattr(spl, "get_knots") else (k + 1)
        return yhat, k_params, {"s": s_val if s is not None else None}

    # --- LOESS ----------------------------------------------------------------
    if model_name == "loess":
        if _lowess is None:
            raise ImportError("statsmodels non installato (serve per LOESS).")
        mp = dict(model_params or {})
        frac = float(mp.get("frac", 0.2))
        yhat = _lowess(y, t, frac=frac, return_sorted=False)
        return yhat, 2, {"frac": frac}

    # --------------------------------------------------------------------------
    raise ValueError(f"Unknown model_name: {model_name}")


def _aic_from_residuals(residuals: np.ndarray, k_params: int) -> float:
    n = residuals.size
    rss = float(np.sum(residuals**2)) / max(n, 1)
    return n * np.log(rss + 1e-12) + 2 * k_params






def normalize_model_spec(name, params=None):
    """
    Converte stringhe compatte in (model canonico, params).
    Esempi:
      'ma5' -> ('moving_average', {'window': 5})
      'savgol_w11_p3' -> ('savgol', {'window':11,'polyorder':3})
      'loess_f0.3' -> ('loess', {'frac':0.3})
      'spline_s50.0' -> ('spline', {'s': 50.0})
      'spline_k6' -> ('spline', {'num_knots': 6})
    """
    s = (name or "").lower().strip()
    p = dict(params or {})

    m = re.fullmatch(r"ma(\d+)", s)
    if m:
        p.setdefault("window", int(m.group(1)))
        return "moving_average", p

    m = re.fullmatch(r"savgol_w(\d+)_p(\d+)", s)
    if m:
        p.setdefault("window", int(m.group(1)))
        p.setdefault("polyorder", int(m.group(2)))
        return "savgol", p

    m = re.fullmatch(r"loess_f([0-9]*\.?[0-9]+)", s)
    if m:
        p.setdefault("frac", float(m.group(1)))
        return "loess", p

    # smoothing via 's'
    m = re.fullmatch(r"spline(?:_s([0-9]*\.?[0-9]+))?", s)
    if m:
        if m.group(1) is not None:
            p["s"] = float(m.group(1))
        return "spline", p

    # numero di nodi interni (LSQUnivariateSpline)
    m = re.fullmatch(r"spline_k(\d+)", s)
    if m:
        p["num_knots"] = int(m.group(1))
        return "spline", p

    if s in ("linear","quadratic","cubic"):
        return s, p

    return s, p









def evaluate_models_on_training(dataset_train,
                                candidates=("linear", "quadratic", "cubic", "ma3", "ma5", "ma7",
                                            "savgol_w11_p3", "spline_s5.0", "loess_f0.2")) -> pd.DataFrame:
    """
    Calcola MAE, RMSE, AIC per ogni candidato sul training.
    """
    times  = np.array([t for t, _ in dataset_train], dtype=float)
    values = np.array([v for _, v in dataset_train], dtype=float)

    rows = []
    for cand in candidates:
        try:
            name, params = normalize_model_spec(cand, {})
            yhat, k, details = _fit_predict_model(times, values, name, params)
            resid = values - yhat
            mae  = float(np.mean(np.abs(resid)))
            rmse = float(np.sqrt(np.mean(resid**2)))
            aic  = float(_aic_from_residuals(resid, k))
            label = cand  # label “umano”
            if name == "moving_average":
                label = f"ma{details.get('window', params.get('window', 5))}"
            elif name == "savgol":
                label = f"savgol(w={details['window']},p={details['polyorder']})"
            elif name == "spline":
                label = f"spline(s={details['s']})"
            elif name == "loess":
                label = f"loess(frac={details['frac']})"
            rows.append({"model": name, "label": label, "params": details, "MAE": mae, "RMSE": rmse, "AIC": aic})
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["MAE", "RMSE", "AIC"], ascending=[True, True, True]).reset_index(drop=True)
    return df


def plot_model_fits_on_training(
    dataset_train,
    models_to_plot=("linear","quadratic","cubic","ma3","ma5","ma7","savgol_w11_p3","spline_s50.0","loess_f0.3"),
    max_plots=10,
    figsize=(12, 3),
    show_residuals=False,
):
    """
    Plotta i fit dei modelli richiesti sulla serie di training.
    Accetta sia nomi canonici (e.g. 'moving_average', 'spline') con params=dict,
    sia shorthand tipo 'ma5', 'savgol_w11_p3', 'spline_s50.0', 'loess_f0.2'.

    models_to_plot può contenere:
      - stringhe shorthand ('ma7', 'spline_k6', 'loess_f0.3', ...)
      - tuple ('moving_average', {'window':7})
      - dict {'model': 'moving_average', 'params': {...}}
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import math

    times  = np.array([t for t,_ in dataset_train], float)
    values = np.array([v for _,v in dataset_train], float)

    # prepara lista (name, params, label) già normalizzata
    normalized = []
    for spec in list(models_to_plot)[:max_plots]:
        try:
            if isinstance(spec, str):
                name, params = normalize_model_spec(spec, {})
                label = spec
            elif isinstance(spec, tuple) and len(spec) == 2:
                name, params = spec[0], dict(spec[1] or {})
                # fai passare comunque dal normalizzatore per uniformità
                name, params = normalize_model_spec(name, params)
                label = f"{name}"
            elif isinstance(spec, dict):
                name = spec.get("model")
                params = dict(spec.get("params", {}) or {})
                name, params = normalize_model_spec(name, params)
                label = f"{name}"
            else:
                # sconosciuto
                continue
            normalized.append((name, params, label))
        except Exception:
            continue

    n = len(normalized)
    if n == 0:
        print("[plot_model_fits_on_training] Nessun modello valido da plottare.")
        return

    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)
    plt.figure(figsize=(figsize[0]*cols, figsize[1]*rows))

    # baseline robusta per diagnostica
    try:
        sigma_base = _robust_scale_baseline(dataset_train)
    except Exception:
        sigma_base = np.median(np.abs(values - np.median(values))) * 1.4826 + 1e-12

    for idx, (name, params, label) in enumerate(normalized, start=1):
        ax = plt.subplot(rows, cols, idx)
        try:
            yhat, k_params, details = _fit_predict_model(times, values, name, params)
            resid = values - yhat
            mae  = float(np.mean(np.abs(resid)))
            # rugosità normalizzata
            dt = np.diff(times); dt[dt==0] = 1.0
            rough = (np.median(np.abs(np.diff(yhat))/dt) /
                     (np.median(np.abs(np.diff(values))/dt) + 1e-12))
            # copertura residui > 2*sigma_base (diagnostica)
            cover = float(np.mean((np.abs(resid) / (sigma_base + 1e-12)) > 2.0))

            ax.plot(times, values, lw=1.2, alpha=0.8, label="series")
            ax.plot(times, yhat, lw=1.6, label=f"{label}")
            ax.set_title(f"{label}  |  MAE={mae:.3f}, ROUGH={rough:.2f}, cover≈{cover:.2f}")
            ax.grid(True, alpha=0.2)
            if show_residuals:
                ax2 = ax.twinx()
                ax2.plot(times, resid, lw=0.8, alpha=0.4)
                ax2.set_ylabel("residuals", fontsize=8)

            ax.legend(loc="best", fontsize=8)
        except Exception as e:
            ax.plot(times, values, lw=1.2, alpha=0.8, label="series")
            ax.set_title(f"{label}  (errore: {e})")
            ax.grid(True, alpha=0.2)
            ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.show()






def select_global_model_regularized(dataset_train,
                                    candidates=("linear","quadratic","cubic",
                                                "ma5","ma7","ma9","savgol_w11_p3","spline_s5.0","loess_f0.2"),
                                    trim_ratio=0.1,            # MAE robusta
                                    one_se_rule=0.05,          # entro +5% dal best
                                    min_ma_window=7,           # MA troppo piccola = overfit
                                    min_loess_frac=0.2,        # LOESS troppo locale = overfit
                                    min_savgol_window=11):     # SavGol: finestra minima
    """
    Ritorna (model_name, params) scegliendo un modello che non 'copia' gli spike:
    - scarta candidati troppo espressivi
    - usa MAE robusta
    - applica one-standard-error rule e preferisce il più semplice tra i 'quasi migliori'
    """
    # 1) valutazioni
    df = evaluate_models_on_training(dataset_train, candidates=candidates)
    if df.empty:
        return "cubic", {}

    # 2) filtra candidati troppo 'aggressivi'
    def _is_ok(row):
        m = row["model"]
        p = row["params"] or {}
        if m == "moving_average":
            if int(p.get("window", 5)) < min_ma_window:
                return False
        if m == "loess":
            if float(p.get("frac", 0.2)) < min_loess_frac:
                return False
        if m == "savgol":
            if int(p.get("window", 11)) < min_savgol_window:
                return False
        return True
    dff = df[df.apply(_is_ok, axis=1)].copy()
    if dff.empty:
        dff = df.copy()

    # 3) MAE robusta per soglia 'one se'
    best_mae = dff["MAE"].min()
    threshold = best_mae * (1.0 + float(one_se_rule))
    near_best = dff[dff["MAE"] <= threshold].copy()
    if near_best.empty:
        near_best = dff.nsmallest(3, "MAE").copy()

    # 4) ordine di 'semplicità' (più a sinistra = più semplice)
    def _complexity_key(row):
        m = row["model"]; p = row["params"] or {}
        if m == "linear": return (0, 0)
        if m == "quadratic": return (1, 0)
        if m == "cubic": return (2, 0)
        if m == "moving_average":
            # finestra grande = più smoothing = più 'semplice'
            return (3, -int(p.get("window", 5)))
        if m == "savgol":
            # finestra grande e polyorder piccolo = più semplice
            return (4, -int(p.get("window", 11)), int(p.get("polyorder", 3)))
        if m == "spline":
            # smoothing alto = più semplice (se s non c'è, consideralo medio)
            s = p.get("s", None)
            return (5, 0 if s is None else -float(s))
        if m == "loess":
            # frac grande = più smoothing = più 'semplice'
            return (6, -float(p.get("frac", 0.2)))
        return (9, 0)

    best_row = sorted(near_best.to_dict("records"), key=_complexity_key)[0]
    return best_row["model"], best_row["params"]




def _robust_scale_baseline(dataset_train, big_window=None):
    """Scala robusta del rumore: usa una MA 'coarse' e calcola MAD dei residui."""
    import numpy as np
    times  = np.array([t for t,_ in dataset_train], float)
    values = np.array([v for _,v in dataset_train], float)
    n = len(values)
    if n < 5:
        return 1.0
    if big_window is None:
        big_window = max(11, (n//10)*2+1)  # dispari ~ N/10
    # baseline: moving average ampia
    pad = big_window//2
    ypad = np.pad(values, (pad,pad), mode="edge")
    kernel = np.ones(big_window)/big_window
    yhat = np.convolve(ypad, kernel, mode="valid")
    resid = values - yhat
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-12
    return 1.4826 * mad  # ≈ sigma robusta


def evaluate_models_on_training_extended(dataset_train, candidates):
    """Valuta MAE/RMSE/AIC + ROUGH e MAD residui per ogni candidato."""
    import numpy as np, pandas as pd
    times  = np.array([t for t,_ in dataset_train], float)
    values = np.array([v for _,v in dataset_train], float)
    rows = []
    for cand in candidates:
        try:
            name, params = normalize_model_spec(cand, {})
            yhat, k, details = _fit_predict_model(times, values, name, params)
            resid = values - yhat
            mae  = float(np.mean(np.abs(resid)))
            rmse = float(np.sqrt(np.mean(resid**2)))
            aic  = float(_aic_from_residuals(resid, k))
            # roughness
            dt = np.diff(times); dt[dt==0]=1.0
            rough = float(
                np.median(np.abs(np.diff(yhat))/dt) /
                (np.median(np.abs(np.diff(values))/dt) + 1e-12)
            )
            # MAD residui
            mad = np.median(np.abs(resid - np.median(resid))) + 1e-12
            sigma_res = 1.4826 * mad
            rows.append({"model":name, "params":details, "label":cand,
                         "MAE":mae, "RMSE":rmse, "AIC":aic,
                         "ROUGH":rough, "SIGMA_RES":sigma_res})
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["MAE","RMSE","AIC"], ascending=[True,True,True]).reset_index(drop=True)
    return df


def select_global_model_for_ad(
    dataset_train,
    candidates=("linear","quadratic","cubic","ma3","ma5","ma7",
                "savgol_w11_p3","spline_s5.0","loess_f0.3"),
    # pesi (bilanciati e neutrali)
    w_acc=0.40, w_smooth=0.30, w_floor=0.15, w_cover=0.15,
    # iperparametri dei termini
    rho=1.05,          # richiedi sigma_res >= 1.05 * sigma_base (evita residui troppo bassi)
    gamma=2.0,         # soglia z-residuo per "residuo grande" (≈ 2σ)
    target_cov=0.10,   # target frazione residui grandi (10%)
    # guard-rail anti overfit
    min_ma_window=5, min_loess_frac=0.25, min_savgol_window=11,
):
    """
    Seleziona il modello per AD minimizzando:
      J = w_acc * Z(MAE) + w_smooth * Z(RD) + w_floor * FLOOR + w_cover * Z(|COVER - target_cov|)
    dove:
      - RD = |log(ROUGH)| (penalità simmetrica: troppo liscio o troppo nervoso),
      - FLOOR = max(0, rho - SIGMA_RES / sigma_base),
      - COVER = fraction( |resid| / sigma_base > gamma ).
    Tutto senza preferenze di famiglia/modello.
    Ritorna (model_name_canonico, params).
    """
    import numpy as np
    import pandas as pd

    df = evaluate_models_on_training_extended(dataset_train, candidates)
    if df.empty:
        return "moving_average", {"window": 5}

    # guard-rail semplici (evita modelli troppo “aggressivi”)
    def _ok(row):
        m,p = row["model"], row["params"] or {}
        if m=="moving_average" and int(p.get("window",5)) < min_ma_window: return False
        if m=="loess"           and float(p.get("frac",0.2)) < min_loess_frac: return False
        if m=="savgol"          and int(p.get("window",11)) < min_savgol_window: return False
        return True
    dff = df[df.apply(_ok, axis=1)].copy()
    if dff.empty:
        dff = df.copy()

    # baseline robusta per scaling residui
    sigma_base = _robust_scale_baseline(dataset_train)

    # calcola COVER per ogni candidato (ri-fit veloce per estrarre i residui)
    times  = np.array([t for t,_ in dataset_train], float)
    values = np.array([v for _,v in dataset_train], float)

    covers = []
    for _, row in dff.iterrows():
        name, params = row["model"], row["params"] or {}
        # riusa i tuoi helper
        yhat, _, _details = _fit_predict_model(times, values, name, params)
        resid = values - yhat
        z = np.abs(resid) / (sigma_base + 1e-12)
        cover = float(np.mean(z > float(gamma)))
        covers.append(cover)
    dff["COVER"] = covers

    # costruisci i termini normalizzati (robusti)
    def _iqr(x):
        return float(np.percentile(x,75)-np.percentile(x,25)) or 1.0

    # 1) Accuratezza
    mae_med = float(np.median(dff["MAE"].values))
    mae_iqr = _iqr(dff["MAE"].values)
    mae_Z   = (dff["MAE"].values - mae_med) / mae_iqr

    # 2) Smoothness simmetrica: RD = |log(ROUGH)|
    RD      = np.abs(np.log(dff["ROUGH"].values + 1e-12))
    rd_med  = float(np.median(RD))
    rd_iqr  = _iqr(RD)
    rd_Z    = (RD - rd_med) / (rd_iqr or 1.0)

    # 3) Floor residui
    floor   = np.maximum(0.0, float(rho) - (dff["SIGMA_RES"].values / (sigma_base + 1e-12)))

    # 4) Coverage vicino al target
    dev_cov = np.abs(dff["COVER"].values - float(target_cov))
    dev_med = float(np.median(dev_cov))
    dev_iqr = _iqr(dev_cov)
    cover_Z = (dev_cov - dev_med) / (dev_iqr or max(1e-3, target_cov))

    # Costo totale (neutrale)
    J = (w_acc   * mae_Z
       + w_smooth* rd_Z
       + w_floor * floor
       + w_cover * cover_Z)

    idx = int(np.argmin(J))
    best = dff.iloc[idx]
    return best["model"], best["params"]


def debug_model_selection_for_ad(
    dataset_train,
    candidates=("linear","quadratic","cubic","ma3","ma5","ma7",
                "savgol_w11_p3","spline_s5.0","loess_f0.3"),
    **kwargs
):
    """
    Ritorna una tabella con tutti i termini usati dalla select_global_model_for_ad.
    Utile per capire perché un modello è stato scelto.
    """
    import numpy as np
    import pandas as pd

    # estrai stessi default usati in select_global_model_for_ad
    w_acc   = float(kwargs.get("w_acc",   0.40))
    w_smooth= float(kwargs.get("w_smooth",0.30))
    w_floor = float(kwargs.get("w_floor", 0.15))
    w_cover = float(kwargs.get("w_cover", 0.15))
    rho     = float(kwargs.get("rho",     1.05))
    gamma   = float(kwargs.get("gamma",   2.0))
    target_cov = float(kwargs.get("target_cov", 0.10))
    min_ma_window   = int(kwargs.get("min_ma_window", 5))
    min_loess_frac  = float(kwargs.get("min_loess_frac", 0.25))
    min_savgol_window = int(kwargs.get("min_savgol_window", 11))

    df = evaluate_models_on_training_extended(dataset_train, candidates)
    if df.empty:
        return pd.DataFrame()

    def _ok(row):
        m,p = row["model"], row["params"] or {}
        if m=="moving_average" and int(p.get("window",5)) < min_ma_window: return False
        if m=="loess"           and float(p.get("frac",0.2)) < min_loess_frac: return False
        if m=="savgol"          and int(p.get("window",11)) < min_savgol_window: return False
        return True
    dff = df[df.apply(_ok, axis=1)].copy()
    if dff.empty:
        dff = df.copy()

    sigma_base = _robust_scale_baseline(dataset_train)
    times  = np.array([t for t,_ in dataset_train], float)
    values = np.array([v for _,v in dataset_train], float)

    covers = []
    for _, row in dff.iterrows():
        name, params = row["model"], row["params"] or {}
        yhat, _, _ = _fit_predict_model(times, values, name, params)
        resid = values - yhat
        z = np.abs(resid) / (sigma_base + 1e-12)
        covers.append(float(np.mean(z > gamma)))
    dff["COVER"] = covers

    def _iqr(x): return float(np.percentile(x,75)-np.percentile(x,25)) or 1.0

    mae_med = float(np.median(dff["MAE"].values)); mae_iqr = _iqr(dff["MAE"].values)
    mae_Z   = (dff["MAE"].values - mae_med) / mae_iqr

    RD     = np.abs(np.log(dff["ROUGH"].values + 1e-12))
    rd_med = float(np.median(RD)); rd_iqr = _iqr(RD)
    rd_Z   = (RD - rd_med) / (rd_iqr or 1.0)

    floor  = np.maximum(0.0, rho - (dff["SIGMA_RES"].values / (sigma_base + 1e-12)))

    dev_cov = np.abs(dff["COVER"].values - target_cov)
    dev_med = float(np.median(dev_cov)); dev_iqr = _iqr(dev_cov)
    cover_Z = (dev_cov - dev_med) / (dev_iqr or max(1e-3, target_cov))

    J = (w_acc*mae_Z + w_smooth*rd_Z + w_floor*floor + w_cover*cover_Z)

    out = dff.copy()
    out["MAE_Z"]   = mae_Z
    out["RD"]      = RD
    out["RD_Z"]    = rd_Z
    out["FLOOR"]   = floor
    out["COVER"]   = dff["COVER"]
    out["COVER_Z"] = cover_Z
    out["J"]       = J
    return out.sort_values(["J","MAE"], ascending=[True,True]).reset_index(drop=True)












################################################################################# MULTIVARIATO RANK GRD SEARCH #################################################################################

def rank_grid_search_mv(
    X_mv,
    alpha_range=np.linspace(-1, 0, 10),
    model_name: str | None = None,
    model_params: dict | None = None,
    L_override_flat: np.ndarray | None = None,  # MV: length = n*C
    transform: str = "rational",
    rho: float = 1.0,
    lambda_global: float = 0.0,
):
    """
    MULTIVARIATE rank grid search.

    - Input: X_mv = [(t, vec), ...] with vec of length C
    - QUBO vars: N = n*C
    - If L_override_flat is provided, uses it for diag(M) to avoid per-batch refit.
      Builds Q batch-locally via off_diag_M_mv(X_mv).
      Monkey-patches matrix_M_mv so solve_qubo_mv uses exactly M_override.

    Selection policy:
      - Results are sorted best-first by MAX number of ones in qaoa_state (n_ones).
      - In case of ties, Python sort is stable -> earlier alpha in alpha_range stays first.
        (So "first maximum" is respected.)

    Returns:
      list[dict] with alpha, beta, qaoa_state, n_ones, approx_ratio, etc.
    """
    from AD_QAOA_ext import AD_QAOA  # local import
    import types
    import numpy as np

    model_name, model_params = normalize_model_spec(model_name, model_params)
    results = []

    # Basic shape info
    n = len(X_mv)
    if n == 0:
        return []

    # Determine C from first vector
    v0 = np.asarray(X_mv[0][1])
    C = int(v0.shape[0]) if v0.ndim > 0 else 1
    N = n * C

    # Validate override length (if provided)
    if L_override_flat is not None:
        L_override_flat = np.asarray(L_override_flat, dtype=float).reshape(-1)
        if L_override_flat.shape[0] != N:
            raise ValueError(
                f"L_override_flat length {L_override_flat.shape[0]} != n*C = {N} "
                f"(n={n}, C={C})"
            )

    for alpha in alpha_range:
        beta = 1 + alpha

        kwargs = {}
        if model_name is not None:
            kwargs["model_name"] = model_name
            kwargs["model_params"] = model_params or {}

        ad_qaoa = AD_QAOA(X_mv, alpha=float(alpha), beta=float(beta), **kwargs)

        # -----------------------
        # Build M (with/without override)
        # -----------------------
        if L_override_flat is None:
            # Full MV construction internally (may refit per batch if diag_M_mv fits inside)
            M = ad_qaoa.matrix_M_mv(
                X_mv, transform=transform, rho=rho, lambda_global=lambda_global
            )
        else:
            # Use cached global diagonal + batch-local Q
            Q = ad_qaoa.off_diag_M_mv(X_mv)
            M = np.diag(L_override_flat) + Q

            # Monkey-patch matrix_M_mv so solve_qubo_mv uses THIS M (no internal rebuild/refit)
            def _matrix_M_mv_override(self, *args, **kwargs):
                return M

            ad_qaoa.matrix_M_mv = types.MethodType(_matrix_M_mv_override, ad_qaoa)

        # -----------------------
        # QAOA solve (MV)
        # -----------------------
        states, _ = ad_qaoa.solve_qubo_mv(
            X_mv, transform=transform, rho=rho, lambda_global=lambda_global
        )
        qaoa_state = np.array(states[0], dtype=int)

        # Safety: ensure correct length
        if qaoa_state.size != N:
            raise RuntimeError(
                f"Returned state length {qaoa_state.size} != expected N={N} (n={n}, C={C})."
            )

        n_ones = int(np.sum(qaoa_state))

        # Diagnostics (kept for logging; selection is ONLY by n_ones)
        qaoa_cost = float(ad_qaoa.cost_function(M, qaoa_state))

        # Classical optimum (WARNING: exponential in N; ok only for small N)
        best_state, classical_cost = ad_qaoa.find_min_cost(M)
        classical_cost = float(classical_cost)
        approx_ratio = float(qaoa_cost / classical_cost) if classical_cost != 0 else np.nan

        # "Selected index": first 1 in the string (or argmax if none)
        sel = np.where(qaoa_state == 1)[0]
        k_sel = int(sel[0]) if sel.size > 0 else int(np.argmax(qaoa_state))

        # Decode (k -> (i,c)) and attach semantic info
        i_sel = k_sel // C
        c_sel = k_sel % C
        t_sel, vec_sel = X_mv[i_sel]
        y_sel = float(np.asarray(vec_sel)[c_sel])

        results.append({
            "alpha": float(alpha),
            "beta": float(beta),
            "qaoa_state": qaoa_state,
            "n_ones": n_ones,

            "qaoa_cost": qaoa_cost,
            "classical_cost": classical_cost,
            "approx_ratio": approx_ratio,

            "selected_index_flat": k_sel,
            "selected_i": int(i_sel),
            "selected_c": int(c_sel),
            "selected_timestamp": int(t_sel),
            "selected_value": y_sel,
        })

    # Best-first: maximize n_ones only; stable sort ensures "first maximum" wins ties
    results.sort(key=lambda r: -r["n_ones"])
    return results
