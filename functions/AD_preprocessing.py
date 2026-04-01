"Anomaly detection preprocessing functions"




import numpy as np
import pandas as pd
import math
from typing import Tuple, List, Dict
from matplotlib import pyplot as plt
from statistics import mean
import random
import gdown
import itertools




def generate_dataset(
    normal_sample_type: str,
    normal_sample_params: dict,
    outlier_sample_type,
    outlier_sample_params,
) -> tuple:
    """
    Generates a dataset containing both normal and outlier samples based on specified distributions
    and parameters. Normal and outlier samples are shuffled to randomize order, and timestamps are
    assigned sequentially.

    Backward compatible:
      - outlier_sample_type: str (old) OR list[str] (new)
      - outlier_sample_params: dict (old) OR list[dict] (new)

    Returns:
        dataset (list[(t, value)]), outlier_indices (set[int])
    """

    def _sample(dist_type: str, params: dict) -> np.ndarray:
        dist_type = dist_type.lower()
        if dist_type == "uniform":
            return np.random.uniform(**params)
        elif dist_type == "normal":
            return np.random.normal(**params)
        elif dist_type == "exponential":
            return np.random.exponential(**params)
        elif dist_type == "poisson":
            return np.random.poisson(**params)
        else:
            raise ValueError(f"Unsupported sample type: {dist_type}")

    # ---- normal ----
    normal_series = _sample(normal_sample_type, normal_sample_params)

    # ---- outliers (single or multi-component) ----
    # normalize inputs to lists
    if isinstance(outlier_sample_type, (list, tuple)):
        out_types = list(outlier_sample_type)
    else:
        out_types = [outlier_sample_type]

    if isinstance(outlier_sample_params, (list, tuple)):
        out_params_list = list(outlier_sample_params)
    else:
        out_params_list = [outlier_sample_params]

    if len(out_types) != len(out_params_list):
        raise ValueError(
            f"outlier_sample_type and outlier_sample_params must have same length "
            f"(got {len(out_types)} vs {len(out_params_list)})"
        )

    outlier_parts = []
    for ot, op in zip(out_types, out_params_list):
        outlier_parts.append(_sample(ot, op))

    outlier_series = np.concatenate(outlier_parts) if outlier_parts else np.array([], dtype=float)

    # ---- merge + shuffle (identical logic) ----
    values = np.concatenate((normal_series, outlier_series))
    indices = np.arange(len(values))

    np.random.shuffle(indices)
    values = values[indices]

    num_points = len(values)
    times = list(range(num_points))
    dataset = [(int(i), float(v)) for i, v in zip(times, values)]

    # outliers are those whose shuffled "origin index" was in the outlier block
    outlier_indices = set(np.where(indices >= len(normal_series))[0])

    return dataset, outlier_indices




def scale_dataset(dataset, new_min=1, new_max=10):
    """
    Scales the values in the dataset to a specified range [new_min, new_max], maintaining the relative 
    proportions of the original values.

    Args:
        dataset (list of tuples): A list of (timestamp, value) pairs representing the dataset (Time_series).
        new_min (float, optional): The minimum value of the scaled range. Default is 1.
        new_max (float, optional): The maximum value of the scaled range. Default is 10.

    Returns:
        scaled_dataset (list of tuples): A list of (timestamp, scaled_value) pairs, where each value 
                                         has been scaled to the specified range.
    """
    all_values = [value for _, value in dataset]

    original_min = min(all_values)
    original_max = max(all_values)

    def scale_value(value):
        return new_min + (value - original_min) * (new_max - new_min) / (original_max - original_min)

    scaled_dataset = [(timestamp, scale_value(value)) for timestamp, value in dataset]

    return scaled_dataset




def load_dataset_from_csv(file_path: str, time_column: str, value_column: str) -> tuple:
    """
    Loads a dataset from a CSV file, mapping timestamps to values and returning the dataset along 
    with the original time values (so they can be used/displayed if needed).

    Args:
        file_path (str): The path to the CSV file.
        time_column (str): The name of the column containing time data.
        value_column (str): The name of the column containing value data.

    Returns:
        dataset (list of tuples): A list of (timestamp, value) pairs with sequentially generated timestamps.
        original_times (np.ndarray): An array of original time values from the CSV file.
    """
    df = pd.read_csv(file_path)

    values = df[value_column].values

    original_times = df[time_column].values

    num_points = len(values)
    times = list(range(num_points))

    dataset = [(int(i), j) for i, j in zip(times, values)]

    return dataset, original_times




def load_partial_dataset_from_csv(file_path: str, time_column: str, value_column: str, start: int, end: int) -> tuple:
    """
    Loads a portion of the dataset from a CSV file and renumbers timestamps from 0 to (end - start).

    Args:
        file_path (str): Path to the CSV file.
        time_column (str): Name of the column containing timestamp data.
        value_column (str): Name of the column containing value data.
        start (int): Starting index of the data range to load.
        end (int): Ending index of the data range to load.

    Returns:
        dataset (list of tuples): A list of (timestamp, value) pairs with renumbered timestamps.
        original_times (np.ndarray): An array of original time values from the selected range.
    """
    df = pd.read_csv(file_path)

    values = df[value_column].values[start:end]
    original_times = df[time_column].values[start:end]

    num_points = len(values)
    times = list(range(num_points))  

    dataset = [(int(i), j) for i, j in zip(times, values)]

    return dataset, original_times




def split_dataset_with_best_batch_size(dataset, overlap=2, batch_sizes=[7, 8, 9, 10]):
    """
    Based on available batch sizes and the desired overlap between the batches, tests the dataset in order to split it in the most balanced way,
    then proceeds to actually effect the split.

    Args:
        dataset (list of tuples): The dataset to split, represented as a list of (timestamp, value) pairs.
        overlap (int, optional): The number of overlapping samples between consecutive batches. Default is 2.
        batch_sizes (list of int, optional): List of possible batch sizes to test. Default is [7, 8, 9, 10].

    Returns:
        best_batches (list of lists): A list of batches, where each batch is a list of (timestamp, value) pairs.
        best_batch_size (int): The batch size that results in the largest final batch.
    """
    num_samples = len(dataset)
    best_batch_size = None
    best_batches = []
    max_last_batch_size = 0

    for batch_size in batch_sizes:
        batches = []
        start = 0

        while start < num_samples:
            batch = dataset[start:start + batch_size]
            batches.append(batch)
            start += (batch_size - overlap)

        last_batch_size = len(batches[-1])

        if last_batch_size > max_last_batch_size:
            max_last_batch_size = last_batch_size
            best_batch_size = batch_size
            best_batches = batches

    return best_batches, best_batch_size





###################################################### da qui modifiche per il multivariato ######################################################



def generate_multivariate_segment_anomaly_dataset(
    n_points=200,
    n_channels=3,
    normal_mean=0.0,
    normal_std=0.5,
    anomaly_shift=3.0,
    anomaly_std=0.6,
    anomaly_windows=[(60, 75), (130, 145)],
    seed=None,
):
    """
    Genera una serie temporale multivariata con anomalie a finestre consecutive.

    Args:
        n_points (int): lunghezza totale della serie
        n_channels (int): numero di variabili
        normal_mean (float): media regime normale
        normal_std (float): deviazione standard regime normale
        anomaly_shift (float): shift medio durante le anomalie
        anomaly_std (float): std durante le anomalie
        anomaly_windows (list of tuples): [(start, end), ...] finestre anomale (inclusive-exclusive)
        seed (int): random seed

    Returns:
        dataset (list of tuples): [(t, x_t)] dove x_t è np.array shape (n_channels,)
        anomaly_mask (np.ndarray): boolean mask di lunghezza n_points
    """
    if seed is not None:
        np.random.seed(seed)

    # Base covariance (leggera correlazione)
    cov = normal_std**2 * (0.7 * np.ones((n_channels, n_channels)) + 0.3 * np.eye(n_channels))

    # Serie normale
    X = np.random.multivariate_normal(
        mean=np.full(n_channels, normal_mean),
        cov=cov,
        size=n_points
    )

    anomaly_mask = np.zeros(n_points, dtype=bool)

    # Inserimento finestre anomale
    for (start, end) in anomaly_windows:
        length = end - start
        X[start:end] = np.random.multivariate_normal(
            mean=np.full(n_channels, normal_mean + anomaly_shift),
            cov=(anomaly_std**2) * np.eye(n_channels),
            size=length
        )
        anomaly_mask[start:end] = True

    # Dataset nel formato temporale
    dataset = [(t, X[t]) for t in range(n_points)]

    return dataset, anomaly_mask







def generate_multivariate_segment_anomaly_dataset_v2(
    n_points=200,
    anomaly_windows=[(30, 40), (70, 80), (110, 120), (150, 160)],
    seed=53,

    # ---- canali base (regime normale) ----
    normal_mean=0.0,
    normal_std=0.5,

    # sinusoide (ch1)
    sin_amp=1.0,
    sin_period=40,
    sin_phase=0.0,

    # onda quadra (ch3)
    square_amp=1.0,
    square_period=30,
    square_duty=0.5,

    # bump gaussiana nel tempo (ch4)
    bump_amp=1.2,
    bump_center=None,
    bump_sigma=18.0,

    # rumore additivo globale
    noise_std=0.10,

    # ---- anomalie base ----
    spike_prob=0.35,        # quanti spike dentro finestra (ch0)
    square_spike_mag=5.0,   # spike iniziale (ch3)

    # ---- anomalie POTENZIATE (per rendere ch0/ch2/ch3 più leggibili) ----
    ch0_spike_mag=10.0,     # spike molto più alti su ch0
    ch2_burst_std=4.0,      # burst varianza più grande su ch2
    ch2_shift=4.0,          # mean shift più forte su ch2
    ch3_plateau_mag=6.0,    # plateau più estremo su ch3

    # anomalie ch1 (sin) e ch4 (bump) - lasciate buone come prima
    sin_flat_mag=3.5,
    sin_alt_mag=3.0,
    bump_invert_mag=2.5
):
    """
    Genera una serie multivariata (5 canali) con dinamiche diverse e anomalie a finestre.
    Output:
      dataset: [(t, x_t)] con x_t shape (5,)
      anomaly_mask: mask booleana length n_points (True nelle finestre)
    Finestre: start inclusive, end exclusive.
    """

    rng = np.random.default_rng(seed)
    t = np.arange(n_points)

    # =========================
    # STEP 1) genera 5 canali diversi
    # =========================

    # ch0: "discreto" da MVN (come tua base) -> prendo solo la prima dimensione
    cov = normal_std**2 * (0.7 * np.ones((5, 5)) + 0.3 * np.eye(5))
    X_mvn = rng.multivariate_normal(mean=np.full(5, normal_mean), cov=cov, size=n_points)
    ch0 = X_mvn[:, 0].copy()

    # ch1: sinusoide + componente cos
    ch1 = sin_amp * np.sin(2*np.pi*t/sin_period + sin_phase) + 0.25*sin_amp*np.cos(2*np.pi*t/(sin_period/2) + 0.3)

    # ch2: gauss i.i.d. + smoothing leggero
    ch2 = rng.normal(0.0, 1.0, size=n_points)
    ch2 = (np.roll(ch2, 1) + ch2 + np.roll(ch2, -1)) / 3.0

    # ch3: onda quadra
    phase = (t % square_period) / square_period
    ch3 = np.where(phase < square_duty, square_amp, -square_amp)

    # ch4: bump gaussiana nel tempo
    if bump_center is None:
        bump_center = (n_points - 1) / 2
    ch4 = bump_amp * np.exp(-0.5 * ((t - bump_center) / bump_sigma) ** 2)

    # Stack
    X = np.vstack([ch0, ch1, ch2, ch3, ch4]).T

    # normalizza per canale per scale comparabili
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    # rumore additivo globale
    X += rng.normal(0.0, noise_std, size=X.shape)

    # =========================
    # STEP 2) inserisci anomalie per finestra (coerenti con i canali)
    # =========================
    anomaly_mask = np.zeros(n_points, dtype=bool)

    for (start, end) in anomaly_windows:
        start = int(max(0, start))
        end   = int(min(n_points, end))
        if end <= start:
            continue

        anomaly_mask[start:end] = True
        L = end - start
        idx = np.arange(start, end)

        # --- ch0 (MVN-like): spike altissimi random dentro finestra ---
        spike_here = rng.random(L) < spike_prob
        if spike_here.any():
            X[idx[spike_here], 0] += ch0_spike_mag * rng.choice([-1, 1], size=spike_here.sum())

        # --- ch1 (sin): alternanza alto/basso (finestre corte) o flat costante (finestre lunghe) ---
        if L <= 12:
            alt = np.where((np.arange(L) % 2) == 0, sin_alt_mag, -sin_alt_mag)
            X[idx, 1] = alt
        else:
            level = sin_flat_mag * rng.choice([-1, 1])
            X[idx, 1] = level

        # --- ch2 (gauss): burst di varianza + mean shift (potenziato) ---
        X[idx, 2] = ch2_shift + rng.normal(0.0, ch2_burst_std, size=L)

        # --- ch3 (square): spike iniziale + plateau fisso estremo (potenziato) ---
        X[start, 3] += square_spike_mag * rng.choice([-1, 1])
        if start + 1 < end:
            X[start+1:end, 3] = ch3_plateau_mag * rng.choice([-1, 1])

        # --- ch4 (bump): inversione + offset per renderla evidente ---
        X[idx, 4] = -bump_invert_mag * np.abs(X[idx, 4]) + 0.5 * rng.choice([-1, 1])

    dataset = [(int(tt), X[int(tt)]) for tt in t]
    return dataset, anomaly_mask




def split_dataset_with_changing_offset(dataset, batch_size, offsets, drop_incomplete=True):
    """
    Crea più partizionamenti (run) della stessa serie, variando l'offset di partenza.

    Args:
        dataset: list of tuples [(t, y), ...] con t idealmente 0..N-1 e ordinato.
        batch_size: int (es. 20)
        offsets: list[int] offsets (es. [0,5,10,15])
        drop_incomplete: se True, scarta l'ultimo batch se incompleto.
                        se False, include anche batch finali più piccoli.

    Returns:
        all_batches: list of runs, dove ogni run è una lista di batch:
                    all_batches[r] = [batch0, batch1, ...]
                    batchk = [(t,y), ...] di lunghezza batch_size (o < se drop_incomplete=False)
    """
    # Assumiamo dataset ordinato per timestamp e indicizzato da 0..N-1
    N = len(dataset)
    all_runs = []

    for off in offsets:
        if off < 0 or off >= N:
            continue

        run_batches = []
        i = off
        while i < N:
            batch = dataset[i:i+batch_size]
            if len(batch) < batch_size and drop_incomplete:
                break
            run_batches.append(batch)
            i += batch_size

        all_runs.append(run_batches)

    return all_runs



































def make_batches_mv(X_mv, batch_size=12, overlap=2):
    batches = []
    step = max(1, batch_size - overlap)
    for start in range(0, len(X_mv), step):
        b = X_mv[start:start+batch_size]
        if len(b) >= 2:
            batches.append(b)
        if start + batch_size >= len(X_mv):
            break
    return batches