"Quantum Anomaly Detection"

# ===== Legacy Qiskit (monolitico 0.42.x) =====
from pandas import options
from qiskit import Aer, QuantumCircuit
from qiskit.algorithms.minimum_eigensolvers import QAOA
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, NELDER_MEAD, ADAM
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.utils import QuantumInstance

import inspect

# =============================================
import numpy as np
import math
from typing import Tuple, List, Dict
from matplotlib import pyplot as plt
from statistics import mean
import random
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
import math
from scipy.optimize import curve_fit
import itertools

# --- OPTIONAL model backends (non obbligatori, fallback sicuri) ---
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







def _get_backend():
    """
    Mantieni per compatibilità (usato altrove); non necessario con Sampler.
    Prova AER statevector, poi AER qasm.
    """
    try:
        from qiskit_aer import Aer
        try:
            return Aer.get_backend("statevector_simulator")
        except Exception:
            return Aer.get_backend("qasm_simulator")
    except Exception:
        return None




def _build_qaoa_solver(self):
    """
    QAOA compatibile con il tuo stack: preferisce AerSampler (shot-based),
    fallback al core Sampler se Aer non è disponibile.
    """
    from qiskit.algorithms.minimum_eigensolvers import QAOA
    from qiskit.algorithms.optimizers import COBYLA

    # preferisci AerSampler (evita statevector gigante)
    sampler = None
    try:
        from qiskit_aer.primitives import Sampler as AerSampler
        sampler = AerSampler(options={"shots": 2048})  # puoi settare options={"shots": 2048}
    except Exception:
        from qiskit.primitives import Sampler as CoreSampler
        sampler = CoreSampler()

    return QAOA(
        sampler=sampler,
        reps=int(getattr(self, "num_layers", 1)),
        optimizer=COBYLA(maxiter=int(getattr(self, "num_iterations", 100))),
    )



def _normalize_model_spec(name, params=None):
    """
    Normalizza in (model_name, model_params).
    Supporta alias:
      - "ma5" -> ("moving_average", {"window": 5})
      - "savgol_w11_p3" -> ("savgol", {"window":11, "polyorder":3})
      - "spline_s5.0"  -> ("spline", {"s":5.0})
      - "loess_f0.2"   -> ("loess", {"frac":0.2})
    """
    p = dict(params or {})
    if not name:
        return None, p
    s = str(name).lower().strip()

    # moving average
    if s.startswith("ma"):
        tail = s[2:]
        w = int(tail) if tail.isdigit() else int(p.get("window", 5))
        return "moving_average", {"window": max(1, int(w))}
    if s == "moving_average":
        return "moving_average", {"window": max(1, int(p.get("window", 5)))}

    # savgol
    if s.startswith("savgol"):
        if "w" in s and "_p" in s:
            try:
                w = int(s.split("w")[1].split("_")[0])
                po = int(s.split("_p")[1])
                return "savgol", {"window": w, "polyorder": po}
            except Exception:
                pass
        return "savgol", {
            "window": int(p.get("window", 11)),
            "polyorder": int(p.get("polyorder", 3)),
        }

    # spline
    if s.startswith("spline"):
        sval = p.get("s", None)
        if "s" in s and "_" not in s:
            try:
                sval = float(s.split("s")[1])
            except Exception:
                pass
        return "spline", {"s": None if sval is None else float(sval)}

    # loess
    if s.startswith("loess"):
        frac = p.get("frac", None)
        if "f" in s and "_" not in s:
            try:
                frac = float(s.split("f")[1])
            except Exception:
                pass
        if frac is None:
            frac = 0.2
        return "loess", {"frac": float(frac)}

    # polinomi o altro
    return s, p


def _resolve_distance(distance_kind, time_scale=1.0, value_scale=1.0):
    """
    Restituisce una funzione distanza (a,b) -> float.
    Supporta:
      - "euclidean", "manhattan", "chebyshev", "absolute_difference"
      - callable custom
    """
    import numpy as _np

    if callable(distance_kind):
        return distance_kind

    name = str(distance_kind).lower().strip()

    def _vec(pt):
        # pt = (t, v)
        return _np.array([time_scale * float(pt[0]), value_scale * float(pt[1])], dtype=float)

    if name == "euclidean":
        return lambda a, b: float(_np.linalg.norm(_vec(a) - _vec(b)))
    if name == "manhattan":
        return lambda a, b: float(_np.abs(_vec(a) - _vec(b)).sum())
    if name == "chebyshev":
        return lambda a, b: float(_np.abs(_vec(a) - _vec(b)).max())
    if name == "absolute_difference":
        # differenza sui valori (asse y) soltanto
        return lambda a, b: float(abs(float(a[1]) - float(b[1])))

    # fallback: euclidea
    return lambda a, b: float(_np.linalg.norm(_vec(a) - _vec(b)))



class AD_QAOA:
    """
    Anomaly detection class using QAOA.
    """
    def __init__(self,
                 X: List[Tuple[int, float]],
                 alpha: float = -0.5,
                 beta: float = 0.5,
                 radius: float = 1,
                 top_n_samples: int = 5,
                 num_iterations: int = 10,
                 model_name: str = 'quadratic',
                 model_params: dict = {},
                 radius_adjustment: bool = False,
                 num_layers: int = 1,
                 distance_kind: str = "absolute_difference",
                 debug=False):
        """
        Initializes the AD_QAOA class.
        Disclaimer: mixer selection work in progress.

        Args:
            X (List[Tuple[int, float]]): Time series data.
            alpha (float): Weight for the linear terms in the QUBO problem.
            beta (float): Weight for the quadratic terms in the QUBO problem.
            radius (float): Radius for the covering boxes.
            top_n_samples (int): Number of top samples to consider.
            num_iterations (int): Number of iterations for the COBYLA optimizer.
            model_name (str): Model selected for the detection pipeline.
            model_params (str): Model's parameters (if any).
            radius_adjustment (bool):  Enables the radius adjustment mechanism for the set covering.
            num_layers (int): Number of layers (p) to use in QAOA.
            debug (bool): Enables some debug prints throught the code.
        """
        self.X = X
        self.alpha = alpha
        self.beta = beta
        self.radius = radius
        self.top_n_samples = top_n_samples
        self.num_iterations = num_iterations
        self.model_name = model_name
        self.model_params = model_params
        self.radius_adjustment = radius_adjustment
        self.num_layers = num_layers  
        self.debug = debug
        self.distance_kind = distance_kind




    def matrix_M(self) -> np.ndarray:
        """
        Builds the matrix M for the QUBO Anomaly Detection objective function. 
        """
        n = len(self.X)
        L = self.diag_M(self.X)
        Q = self.off_diag_M(self.X)
        M = np.zeros((n, n))

        for i in range(n):
            M[i, i] = L[i]
            for j in range(i + 1, n):
                M[i, j] = Q[i, j]
                M[j, i] = Q[i, j]

        if self.debug:
          print("(debug) L:\n", L)
          print("(debug) Q:\n", Q)
          print("(debug) M:\n", M)

        return M




    # helper per scegliere un backend disponibile
    def _get_backend():
        try:
            return Aer.get_backend("statevector_simulator")
        except Exception:
            return Aer.get_backend("qasm_simulator")
        



    
    def off_diag_M(self, data):
        """
        Parte fuori-diagonale di M: distanza tra punti (con scaling opzionale).
        """
        import numpy as np

        n = len(data)
        Q = np.zeros((n, n), dtype=float)

        # usa self.distance_kind, con fallback 'euclidean'
        dk = getattr(self, "distance_kind", "euclidean")
        time_scale  = float(getattr(self, "time_scale", 1.0))
        value_scale = float(getattr(self, "value_scale", 1.0))
        dist_fn = _resolve_distance(dk, time_scale=time_scale, value_scale=value_scale)

        for i in range(n):
            for j in range(i + 1, n):
                d = dist_fn(data[i], data[j])
                Q[i, j] = d
                Q[j, i] = d
        return Q




    def compute_delta(self, data: np.ndarray, model_values: np.ndarray) -> List[float]:
        """
        Creates the diagonal terms (delta, linear contribution for the QUBO problem) for the corresponding M matrix,
        computing the difference between the data sample and the corresponding model fitting values.

        Args:
         data (np.ndarray): Time Series data.
         model_values (np.ndarray): Corresponding model values for the fitting.
         
        Returns:
         List[float]: List of absolute differences data sample/model sample.
        """
        return [abs(sample_value - model_value.item()) for sample_value, model_value in zip(data, model_values)]



        
    def inverse_transform(self, delta_values: List[float], scale_factor: float = 0.5) -> List[float]:
        """
        Applies an inverse transformation to delta values. Larger values ​​of delta become smaller, and vice versa.
        This allows the minimization problem for the QUBO formulation to correctly identify anomalies based on the highest values
        in the model fitting vector (high differences bewteen the model and the data sample). Also works as a normalization for the values.

        Args:
         delta_values (List[float]): List of absolute differences data sample/model sample.
         scale_factor (float): Scaling factor for the normalization and the transformation of the values.
         
        Returns:
         transfrmed_values (List[float]): List of transformed values to be used in the building of the diagonal of M for the QUBO problem.
        """
        transformed_values = []
        for delta in delta_values:
            if delta != 0:
                transformed_value = scale_factor / delta
            else:
                transformed_value = scale_factor  
            transformed_values.append(transformed_value)
        return transformed_values



    
    def diag_M(self, data, scale_factor=None, transform="rational", rho=1.0):
        """
        Diagonale L: f(residuo) con f decrescente.
        - Calcola yhat dal modello self.model_name/self.model_params
        - resid = |y - yhat|
        - sigma robusta (MAD) o scale_factor se fornito
        - transform:
            "rational" -> L = 1 / (1 + (resid / (rho*sigma)))
            "exp"      -> L = exp( - resid / (rho*sigma) )
        Ritorna un vettore L di lunghezza n (non una matrice).
        """
        import numpy as np

        # --- estrai times e values
        t = np.asarray([tt for tt, _ in data], dtype=float)
        y = np.asarray([vv for _, vv in data], dtype=float)
        n = len(y)

        # --- helper: fit & predict per i modelli
        def _fit_predict_model(model_name: str, model_params: dict | None):
            name = (model_name or "cubic").lower()
            params = dict(model_params or {})

            if name == "linear":
                coefs = np.polyfit(t, y, 1)
                yhat = np.polyval(coefs, t)
                return yhat
            if name == "quadratic":
                coefs = np.polyfit(t, y, 2)
                yhat = np.polyval(coefs, t)
                return yhat
            if name == "cubic":
                coefs = np.polyfit(t, y, 3)
                yhat = np.polyval(coefs, t)
                return yhat
            if name in ("moving_average", "ma"):
                w = int(params.get("window", 5))
                w = max(1, int(w))
                pad = w // 2
                ypad = np.pad(y, (pad, pad), mode="edge")
                kernel = np.ones(w) / w
                yhat = np.convolve(ypad, kernel, mode="valid")
                return yhat
            if name == "savgol":
                try:
                    from scipy.signal import savgol_filter
                except Exception as e:
                    raise RuntimeError("scipy.signal.savgol_filter richiesto per 'savgol'") from e
                w = int(params.get("window", 11))
                p = int(params.get("polyorder", 3))
                if w % 2 == 0:
                    w += 1
                if w <= p:
                    w = p + 2 + (p % 2 == 0)
                yhat = savgol_filter(y, window_length=w, polyorder=p, mode="interp")
                return yhat
            if name == "spline":
                try:
                    from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
                except Exception as e:
                    raise RuntimeError("SciPy richiesto per 'spline'") from e
                k = 3
                s = params.get("s", None)
                num_knots = params.get("num_knots", None)
                if num_knots and int(num_knots) >= 2:
                    tmin, tmax = float(t.min()), float(t.max())
                    knots = np.linspace(tmin, tmax, int(num_knots) + 2)[1:-1]
                    spl = LSQUnivariateSpline(t, y, t=knots, k=k)
                    return spl(t)
                spl = UnivariateSpline(t, y, s=s, k=k)
                return spl(t)
            if name == "loess":
                try:
                    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
                except Exception as e:
                    raise RuntimeError("statsmodels richiesto per 'loess'") from e
                frac = float(params.get("frac", 0.2))
                return _lowess(y, t, frac=frac, return_sorted=False)

            # fallback: cubic
            coefs = np.polyfit(t, y, 3)
            return np.polyval(coefs, t)

        # --- predizione + residui
        yhat = _fit_predict_model(getattr(self, "model_name", "cubic"),
                                getattr(self, "model_params", None))
        resid = np.abs(y - yhat)

        # --- scala robusta
        if scale_factor is None:
            # MAD (median absolute deviation) -> sigma ≈ 1.4826 * MAD
            med = np.median(resid)
            mad = np.median(np.abs(resid - med))
            sigma = 1.4826 * mad
            if not np.isfinite(sigma) or sigma <= 1e-12:
                sigma = max(1e-6, np.std(resid))
        else:
            sigma = float(scale_factor)
            if sigma <= 1e-12:
                sigma = 1e-6

        denom = (rho * sigma) if (rho is not None and rho > 0) else sigma
        denom = max(denom, 1e-12)

        # --- trasformazione decrescente (questa è la chiave!)
        tr = (transform or "rational").lower()
        if tr == "rational":
            L = 1.0 / (1.0 + resid / denom)
        elif tr == "exp":
            L = np.exp(-resid / denom)
        else:
            # fallback semplice: clip lineare
            L = np.maximum(0.0, 1.0 - resid / denom)

        # assicurati che sia 1-D float
        L = np.asarray(L, dtype=float).reshape(-1,)
        return L



        
            
    def distance(self, point1: np.ndarray, point2: np.ndarray, kind: str = "absolute_difference") -> float:
        """
        Calculates the (absolute) distance between two points.
    
        Args:
            point1 (np.ndarray): First point.
            point2 (np.ndarray): Second point.
            kind (str): the kind of distance to be used. Default is "absolute".
    
        Returns:
            float: the distance between point1 and point2.
    
        Raise:
            NotImplementedError: If a not supported distance type is selected.
            ValueError: if at least one of the points is not in the bidimensional array form (time_series' data point).
        """
        

        if point1.shape != (2,) or point2.shape != (2,):
            raise ValueError("Both points have to be bidimensional arrays with 2 elements (x, y).")

        if kind == "euclidean":
            return np.sqrt(np.sum((point1 - point2) ** 2))
        elif kind == "chebyshev":
            return np.max(np.abs(point1 - point2))
        elif kind == "manhattan":
            return np.sum(np.abs(point1 - point2))
        elif kind == "absolute_difference":
            return np.abs(point1[1] - point2[1])
        else:
            raise NotImplementedError




    def moving_average_expanded(self, series: np.ndarray, window_size: int = 2) -> np.ndarray:
        """
        Calculates the moving average model (expanded to match the lenght of the list of samples) for the group of sample given
        a selected window size (default is 2).

        Args:
         series (np.ndarray): Data samples used.
         window_size (int): The shifting window size for the computation.

        Returns:
         expanded_moving_avg (np.ndarray): Vector of the moving average model (with position currespondance with the original data samples).
        """

        series = np.asarray(series)

        if window_size < 1:
            raise ValueError("Window size must be at least 1.")
        if window_size > len(series):
            raise ValueError("Window size can't be greater than the length of the dataset.")

        moving_avg = np.convolve(series, np.ones(window_size) / window_size, mode='valid')
        expanded_moving_avg = np.repeat(moving_avg, window_size)
        expanded_moving_avg = expanded_moving_avg[:len(series)]

        return expanded_moving_avg



    
    def plot_model(self):
        """
        Creates the plotting showcasing the model fitting on the Time Series.

        Raises:
         ValueError: If a non supported model is selected.
        """
        data_values = np.array([value for _, value in self.X])
        times = np.array([t for t, _ in self.X]).reshape(-1, 1)

        models = {
            'linear': lambda x: np.polyval(np.polyfit(x.flatten(), data_values, 1), x),
            'quadratic': lambda x: np.polyval(np.polyfit(x.flatten(), data_values, 2), x),
            'cubic': lambda x: np.polyval(np.polyfit(x.flatten(), data_values, 3), x),
            'moving_average': lambda x: self.moving_average_expanded(data_values, self.model_params.get('window_size', 2)),
        }

        if self.model_name in models:
            model_func = models[self.model_name]
            expected_values = model_func(times)

            plt.figure(figsize=(10, 6))
            plt.plot(times, data_values, 'bo-', label='Dataset')
            plt.plot(times, expected_values, 'r-', label=f'Model ({self.model_name})')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title('Time series fitting')
            plt.legend()
            plt.show()
        else:
            raise ValueError(f"Model {self.model_name} is not supported.")



            
    def plot_time_series(self):
        """
        Creates the plotting showcasing the Time Series.
        """
        times = np.array([t for t, _ in self.X])
        values = np.array([v for _, v in self.X])

        plt.figure(figsize=(10, 6))

        plt.plot(times, values, marker='o', color='blue', linestyle='-', label='Time Series')

        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Time Series')
        if self.debug:
            plt.legend()
        plt.show()




    def plot_distances_with_arrows(self):
        """
        Creates the plotting showcasing the distance between a selected sample (first_sample_value) and the rest of the Time Series (batch).
        """
        data_values = np.array([value for _, value in self.X])
        times = np.array([t for t, _ in self.X])

        first_sample_value = data_values[3] 
        first_sample_time = times[3]

        plt.figure(figsize=(8, 6))

        plt.scatter(times, data_values, color='blue', label='Data Points')

        for i in range(0, len(times)):
          plt.arrow(first_sample_time, first_sample_value,
                    times[i] - first_sample_time, data_values[i] - first_sample_value,
                      head_width=0.05, head_length=0.1, fc='red', ec='red', alpha=0.5)

        plt.scatter([first_sample_time], [first_sample_value], color='green', s=100, label='Example')

        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Distances')
        if self.debug:
            plt.legend()
        plt.show()




    def radius_adj(self, centers):
        """
        Radius adjustment algorithm. Determinates the best radius value for the centers and batch in exam.
        Default the radius is set to 1.00 and then is tried enlarging or reducing for the covering achievement.
        Adjust the exclusive tolerance (0.33 standard) for the max_non_center_value.
        Adjust the inclusive tolerance ( /2 standard) for the normal values inclusion.

        Args:
         centers: List of centers selected as 1s in the QAOA quantum state solution.

        Returns:
         self.radius (float): The optimal radius value identified for the batch at hand.
        """
        if self.radius_adjustment:
            X_v = [(i[0], i[1]) for i in self.X]
            not_centers = [i for i in range(len(self.X)) if i not in [c[0] for c in centers]]

            if not_centers:
                max_non_center_value = max([X_v[i][1] for i in not_centers])
                r = 0
                std_dev = np.std([i[1] for i in X_v])
                data_range = max([i[1] for i in X_v]) - min([i[1] for i in X_v])

                for i, center in enumerate(centers):
                    center_coords = np.array([center[0], center[1]])

                    for not_center_index in not_centers:
                        not_center_coords = np.array([X_v[not_center_index][0], X_v[not_center_index][1]])

                        distanza = np.linalg.norm(center_coords - not_center_coords)

                        if distanza < (0.33 * max_non_center_value):   #default 0.33 (cambiato da 0.22)
                            r = max(r, distanza)

                r += (std_dev / 3)  #2 defalut, cambiato da 5

                if r <= 1.0:
                    if data_range < 0.5:
                        print(f"The radius of 1.0 is too large for this dataset range.: {data_range}")
                        self.radius = 1.0 #cambiato da 0.067
                        print("Radius:", r)
                        print("Radius adjusted check_ok")
                    else:
                        self.radius = 1.0
                        print("Radius:", r)
                        print("Radius adjusted check_ok")
                else:
                    self.radius = r
                    print("Radius:", self.radius)
                    print("Radius adjusted check_ok")

            elif not not_centers:
                self.radius = 1
                print("Radius:", self.radius)
                print("No adjustment")

        return self.radius



    
    def solve_qubo(self):
        """
        Costruisce il QUBO da M, risolve con QAOA e ritorna:
        - top_n_states: lista di bitstring (lista di 0/1)
        - variables_dict: mappa nome->valore
        """
        import numpy as np
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer

        M = self.matrix_M()
        n = len(self.X)

        # costi
        linear_terms = float(self.alpha) * np.diag(M)
        quadratic_terms = {}
        for i in range(n):
            for j in range(i + 1, n):
                if M[i, j] != 0.0:
                    quadratic_terms[(f"x{i}", f"x{j}")] = float(self.beta) * float(M[i, j])

        # QUBO
        qubo = QuadraticProgram()
        for i in range(n):
            qubo.binary_var(name=f"x{i}")
        qubo.minimize(linear=linear_terms, quadratic=quadratic_terms)

        # QAOA
        qaoa_solver = _build_qaoa_solver(self)
        meo = MinimumEigenOptimizer(qaoa_solver)
        result = meo.solve(qubo)

        # estrazione robusta top-k
        top_n = int(getattr(self, "top_n_samples", 5))
        top_n_states = []
        if hasattr(result, "samples") and result.samples:
            for s in list(result.samples)[:top_n]:
                x_bits = [int(round(b)) for b in s.x]
                top_n_states.append(x_bits)
        else:
            x_bits = [int(round(b)) for b in result.x]
            top_n_states = [x_bits]

        return top_n_states, result.variables_dict





    def centers_storage(self, state=None):
        """
        Stores the center coordinates corresponding to 1s in the QAOA solution (default is the first state if none is provided).
    
        Args:
         state (List[int], optional): A binary list representing the QAOA solution state. If None, the function will 
                                      use the first quantum state (most counts).
        Returns:
         centers (List[Tuple[int, float]]): A list of tuples, where each tuple contains a timestamp and its corresponding 
                                            value for each selected center. These centers are identified by the positions 
                                            in the solution string where the value is '1'.
        """
        if state is None:
            _, best_solution = self.solve_qubo()
            non_zero_vars = [key for key, value in best_solution.items() if value > 0.5]
            time_stamps = [self.X[int(s.replace("x", ""))][0] for s in non_zero_vars]
            selected_values = [self.X[int(s.replace("x", ""))][1] for s in non_zero_vars]
            centers = [(ts, val) for ts, val in zip(time_stamps, selected_values)]

            if self.debug:
                print('selected_centers:', centers)
            return centers
        else:
            best_solution = state
            non_zero_vars = [value for value in state if value > 0.5]
            time_stamps = [self.X[i][0] for i in range(len(state)) if non_zero_vars[i] > 0.5]
            selected_values = [self.X[i][1] for i in range(len(state)) if non_zero_vars[i] > 0.5]
            centers = [(ts, val) for ts, val in zip(time_stamps, selected_values)]

            if self.debug:
                print('selected_centers:', centers)
            return centers



    
    def detect_anomalies(self, state=None):
        """
        Detects anomalies based on the selected centers in the QAOA solution state.
    
        Args:
         state (List[int], optional): A binary list representing the QAOA solution state. If None, the function will 
                                         use the first quantum state (most counts).
    
        Returns:
         boxes (List[Tuple[int, float]]): A list of tuples representing the coverage spheres for each center, 
                                             where each box is centered around selected points and adjusted to cover 
                                             surrounding data points.
        """
        if state is None:
            _, best_solution = self.solve_qubo()
            non_zero_vars = [key for key, value in best_solution.items() if value > 0.5]
            time_stamps = [self.X[int(s.replace("x", ""))][0] for s in non_zero_vars]
            selected_values = [self.X[int(s.replace("x", ""))][1] for s in non_zero_vars]
            centers = [(ts, val) for ts, val in zip(time_stamps, selected_values)]
            
            if self.debug:
                print('selected_centers:', centers)
                
            boxes = self.covering_boxes(centers)
            return boxes
        else:
            best_solution = state
            non_zero_vars = [value for value in state if value > 0.5]
            time_stamps = [self.X[i][0] for i in range(len(state)) if non_zero_vars[i] > 0.5]
            selected_values = [self.X[i][1] for i in range(len(state)) if non_zero_vars[i] > 0.5]
            centers = [(ts, val) for ts, val in zip(time_stamps, selected_values)]
            boxes = self.covering_boxes(centers)
            return boxes

        
        
    def associate_centers_with_radius(self, state=None):
        """
        Associates each center with the corresponding calculated radius.
    
        Args:
            state (list, optional): A binary list representing the QAOA solution state. If None, the function will 
                                    retrieve the best solution available.
    
        Returns:
            centers_with_radius (list of tuples): A list of (center, radius) pairs, where each center has an associated radius.
    
        """
        centers = self.centers_storage(state=state)
    
        radius = self.radius_adj(centers)
    
        if self.debug:
            print("Radius from radius_adj:", radius)
    
        centers_with_radius = [(center, radius) for center in centers]
        
        if self.debug:
            print("List of centers and radiuses:", centers_with_radius)
    
        return centers_with_radius




    def visualize_anomalies(self, state=None):
        """
        Visualizes anomalies detected by the QAOA model, highlighting centers and coverage areas on a scatter plot.
        Adjust plot title.
        
        Args:
         state (List[int], optional): A binary list representing the QAOA solution state. If None, the function will 
                                         use the first quantum state (most counts).
        """
        plt.figure(figsize=(10, 6))

        boxes = self.detect_anomalies(state)
        X_t = [i[0] for i in self.X]
        X_v = [i[1] for i in self.X]

        for box in boxes:
            plt.plot(box[0], box[1], color="orange")

        plt.scatter(X_t, X_v)

        centers = self.centers_storage(state)
        time_stamps = [center[0] for center in centers]
        selected_values = [center[1] for center in centers]

        if self.debug:
            print("selected_values:", centers)

        plt.scatter(time_stamps, selected_values, color="red")

        plt.axis('scaled')
        plt.title('Detection')    
        plt.show()



    
    def cost_function(self, M: np.ndarray, state: np.ndarray) -> float:
        """
        Calculates the cost for a given QAOA solution state based on the matrix M.
        
        Args:
         M (np.ndarray): A square symmetric matrix representing interactions between variables. The diagonal elements 
                         represent linear terms, while the off-diagonal elements represent quadratic terms.
         state (np.ndarray): A binary vector (1D array) representing the QAOA solution state. The length of `state` 
                             should match the dimensions of `M`.
    
        Returns:
         cost (float): The calculated cost for the given state, based on the weighted sum of linear and quadratic terms.
    
        Raises:
            ValueError: If the dimension of `state` does not match the dimension of `M`.
        """
        if len(state.shape) > 1:
            state = state[0]

        if len(state) != M.shape[0]:
            raise ValueError(f"State has dimension {len(state)} and instead should have dimension {M.shape[0]}.")

        M_off_diag = M - np.diag(np.diag(M))

        linear_terms = self.alpha * np.dot(np.diag(M), state)

        quadratic_terms = self.beta * np.dot(state.T, np.dot(M_off_diag, state))

        # cost = -(linear_terms + quadratic_terms)

        cost = linear_terms + quadratic_terms

        return cost



    
    def find_max_cost(self, M: np.ndarray) -> tuple:
        """
        Finds the state with the maximum cost for a given matrix M.
    
        Args:
            M (np.ndarray): A square symmetric matrix representing interactions between variables, 
                            with diagonal elements representing linear terms and off-diagonal elements 
                            representing quadratic terms.
    
        Returns:
            best_state (np.ndarray): A binary array representing the state that maximizes the cost function.
            max_cost (float): The maximum cost value obtained with the best state.
        """
        max_cost = float('-inf')
        best_state = None
    
        for state in itertools.product([0, 1], repeat=len(M)):
            state_array = np.array(state)
            current_cost = self.cost_function(M, state_array)
    
            if current_cost > max_cost:
                max_cost = current_cost
                best_state = state_array
    
        return best_state, max_cost
    


    
    def find_min_cost(self, M: np.ndarray) -> tuple:
        """
        Finds the state with the minimum cost for a given matrix M.
    
        Args:
            M (np.ndarray): A square symmetric matrix representing interactions between variables, 
                            with diagonal elements representing linear terms and off-diagonal elements 
                            representing quadratic terms.
    
        Returns:
            best_state (np.ndarray): A binary array representing the state that minimizes the cost function.
            min_cost (float): The minimum cost value obtained with the best state.
        """
        min_cost = float('inf')
        best_state = None
    
        for state in itertools.product([0, 1], repeat=len(M)):
            state_array = np.array(state)
            current_cost = self.cost_function(M, state_array)
    
            if current_cost < min_cost:
                min_cost = current_cost
                best_state = state_array
    
        return best_state, min_cost
    






    # additional method to retrieve more information from the QAOA solution


    def solve_qubo_extended(self):
        """
        Come solve_qubo ma ritorna anche probabilità top-k se disponibili.
        """
        import numpy as np
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer

        M = self.matrix_M()
        n = len(self.X)

        linear_terms = float(self.alpha) * np.diag(M)
        quadratic_terms = {}
        for i in range(n):
            for j in range(i + 1, n):
                if M[i, j] != 0.0:
                    quadratic_terms[(f"x{i}", f"x{j}")] = float(self.beta) * float(M[i, j])

        qubo = QuadraticProgram()
        for i in range(n):
            qubo.binary_var(name=f"x{i}")
        qubo.minimize(linear=linear_terms, quadratic=quadratic_terms)

        qaoa_solver = _build_qaoa_solver(self)
        meo = MinimumEigenOptimizer(qaoa_solver)
        result = meo.solve(qubo)

        top_n = int(getattr(self, "top_n_samples", 5))
        top_n_states, topk_probs = [], []

        if hasattr(result, "samples") and result.samples:
            for s in list(result.samples)[:top_n]:
                x_bits = [int(round(b)) for b in s.x]
                bits = "".join(str(b) for b in x_bits)
                p = getattr(s, "probability", None)
                try:
                    p = float(p)
                except Exception:
                    p = 1.0 / float(top_n)
                top_n_states.append(x_bits)
                topk_probs.append((bits, p))
        else:
            x_bits = [int(round(b)) for b in result.x]
            bits = "".join(str(b) for b in x_bits)
            top_n_states = [x_bits]
            topk_probs = [(bits, 1.0)]

        return top_n_states, result.variables_dict, topk_probs








    def qubo_edges(self, tol: float = 1e-12):
        """
        Restituisce la lista di edge (i,j,w_ij) dove M[i,j] è "non nullo"
        (|M[i,j]| > tol) con i<j. Queste edge corrispondono ai termini ZZ.
        """
        import numpy as _np
        M = self.matrix_M()            # usa la tua costruzione L + Q
        n = M.shape[0]
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                w = float(M[i, j])
                if _np.isfinite(w) and abs(w) > tol:
                    edges.append((i, j, w))
        return edges




        

    def qubo_edge_index_list(self, tol: float = 1e-12):
        """
        Come qubo_edges() ma ritorna solo (i,j) senza pesi,
        utile per costruire l'ansatz QAOA e contare le RZZ per layer.
        """
        return [(i, j) for (i, j, _) in self.qubo_edges(tol=tol)]


















#################################################### DA qui upgrade per il multivariato ####################################################





    def solve_qubo_with_energy(self, top_n=None):
        """
        Come solve_qubo(), ma in più restituisce anche l'energia minima (fval)
        della soluzione ottima.

        Ritorna:
        - top_n_states: lista di bitstring (lista di 0/1), come solve_qubo
        - best_energy: energia minima (float)
        """
        import numpy as np
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer

        M = self.matrix_M()
        n = len(self.X)

        # costi
        linear_terms = float(self.alpha) * np.diag(M)
        quadratic_terms = {}
        for i in range(n):
            for j in range(i + 1, n):
                if M[i, j] != 0.0:
                    quadratic_terms[(f"x{i}", f"x{j}")] = float(self.beta) * float(M[i, j])

        # QUBO
        qubo = QuadraticProgram()
        for i in range(n):
            qubo.binary_var(name=f"x{i}")
        qubo.minimize(linear=linear_terms, quadratic=quadratic_terms)

        # QAOA
        qaoa_solver = _build_qaoa_solver(self)
        meo = MinimumEigenOptimizer(qaoa_solver)
        result = meo.solve(qubo)

        # estrazione robusta top-k (stessa logica di solve_qubo)
        if top_n is None:
            top_n = int(getattr(self, "top_n_samples", 5))

        top_n_states = []
        if hasattr(result, "samples") and result.samples:
            for s in list(result.samples)[:top_n]:
                x_bits = [int(round(b)) for b in s.x]
                top_n_states.append(x_bits)
        else:
            x_bits = [int(round(b)) for b in result.x]
            top_n_states = [x_bits]

        # energia minima (valore della funzione obiettivo)
        best_energy = float(result.fval)

        return top_n_states, best_energy
    









    def solve_qubo_with_energy_v2(self, top_n=None, k_frac=0.8, lambda_card=None):
        """
        Versione V2: come solve_qubo_with_energy(), ma aggiunge un vincolo di cardinalità
        per rendere l'energia minima un buon "window anomaly score".

        Aggiunge al costo:
            lambda_card * (sum_i z_i - k)^2
        con k = round(k_frac * n)

        Ritorna:
        - top_n_states: lista di bitstring (lista di 0/1)
        - best_energy: energia minima (float)
        """
        import numpy as np
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer

        M = self.matrix_M()
        n = len(self.X)

        # --- parametri cardinalità ---
        if lambda_card is None:
            lambda_card = float(3*self.beta)  # default sensato: stessa scala del termine quadratico #da nulla a 3xbeta

        k = int(np.clip(int(round(k_frac * n)), 0, n))  # k in [0, n]

        # --- costi base (come V1) ---
        linear_terms = float(self.alpha) * np.diag(M).astype(float)
        quadratic_terms = {}
        for i in range(n):
            for j in range(i + 1, n):
                if M[i, j] != 0.0:
                    quadratic_terms[(f"x{i}", f"x{j}")] = float(self.beta) * float(M[i, j])

        # --- aggiunta penalità: lambda*(sum z - k)^2 ---
        # Espansione:
        # (sum z - k)^2 = (sum z)^2 - 2k(sum z) + k^2
        # (sum z)^2 = sum z_i^2 + 2 sum_{i<j} z_i z_j = sum z_i + 2 sum_{i<j} z_i z_j (perché z_i^2=z_i)
        # => lambda[(1-2k) sum z_i + 2 sum_{i<j} z_i z_j] + costante
        linear_terms = linear_terms + lambda_card * (1 - 2 * k)

        for i in range(n):
            for j in range(i + 1, n):
                key = (f"x{i}", f"x{j}")
                quadratic_terms[key] = quadratic_terms.get(key, 0.0) + 2.0 * lambda_card

        # --- QUBO ---
        qubo = QuadraticProgram()
        for i in range(n):
            qubo.binary_var(name=f"x{i}")
        qubo.minimize(linear=linear_terms, quadratic=quadratic_terms)

        # --- QAOA ---
        qaoa_solver = _build_qaoa_solver(self)
        meo = MinimumEigenOptimizer(qaoa_solver)
        result = meo.solve(qubo)

        # --- estrazione robusta top-k ---
        if top_n is None:
            top_n = int(getattr(self, "top_n_samples", 5))

        top_n_states = []
        if hasattr(result, "samples") and result.samples:
            for s in list(result.samples)[:top_n]:
                x_bits = [int(round(b)) for b in s.x]
                top_n_states.append(x_bits)
        else:
            x_bits = [int(round(b)) for b in result.x]
            top_n_states = [x_bits]

        best_energy = float(result.fval)
        return top_n_states, best_energy






















###################################################################### multivariato QUBO OG ######################################################################



    def _n_channels_from_Xmv(self, X_mv):
        v0 = X_mv[0][1]
        return int(np.asarray(v0).shape[0])

    def _flat_index(self, i, c, C):
        # i in [0..n-1], c in [0..C-1]
        return i * C + c

    def _unflat_index(self, k, C):
        i = k // C
        c = k % C
        return i, c

    def _extract_channel_series(self, X_mv, c):
        # ritorna una serie monovariata (t, y_c) come la tua X originale
        out = []
        for t, vec in X_mv:
            out.append((t, float(np.asarray(vec)[c])))
        return out





    def off_diag_M_mv(self, X_mv):
        import numpy as np

        n = len(X_mv)
        C = self._n_channels_from_Xmv(X_mv)
        N = n * C
        Q = np.zeros((N, N), dtype=float)

        for c in range(C):
            Xc = self._extract_channel_series(X_mv, c)   # [(t, y_c), ...]
            Qc = self.off_diag_M(Xc)                     # tua funzione monovariata

            for i in range(n):
                for j in range(i + 1, n):
                    k = self._flat_index(i, c, C)
                    l = self._flat_index(j, c, C)
                    Q[k, l] = float(Qc[i, j])
                    Q[l, k] = float(Qc[i, j])

        return Q



    def diag_M_mv(self, X_mv, transform="rational", rho=1.0, lambda_global=0.0):
        """
        Ritorna L_flat lungo N=n*C.
        - lambda_global in [0,1]: 0 = solo residuo canale; 1 = solo residuo globale ||r_i||.
        - Non aggiunge termini quadratici cross-canale (B=0).
        """
        import numpy as np

        n = len(X_mv)
        C = self._n_channels_from_Xmv(X_mv)

        # 1) yhat per canale (riusando la logica di diag_M)
        #    ma ci serve anche il residuo (non solo L), quindi ricostruiamo resid in modo coerente.
        #    Per essere minimali e NON duplicare troppo codice:
        #    -> calcoliamo yhat replicando la parte model di diag_M,
        #       e poi applichiamo la stessa trasformazione usata in diag_M.

        # Per minimizzare modifiche: sfruttiamo diag_M per ottenere L_c,
        # ma per costruire il "globale" abbiamo bisogno dei residui.
        # Quindi facciamo un micro-fit per canale usando la stessa logica di diag_M:
        # (qui, minimal: usiamo la stessa funzione interna con polyfit/moving average ecc.
        #  ma per non riscriverla tutta, facciamo una scelta semplice: residui = inverse di L non stabile.
        #  Quindi: implementiamo qui un residuo "vero" copiando SOLO la parte fit di diag_M.)

        t = np.asarray([tt for tt, _ in X_mv], dtype=float)

        # --- helper minimale: predizione per una serie monovariata (copia ridotta da diag_M)
        def _predict_yhat_for_series(y):
            name = (getattr(self, "model_name", "cubic") or "cubic").lower()
            params = dict(getattr(self, "model_params", {}) or {})

            if name == "linear":
                coefs = np.polyfit(t, y, 1); return np.polyval(coefs, t)
            if name == "quadratic":
                coefs = np.polyfit(t, y, 2); return np.polyval(coefs, t)
            if name in ("cubic", None):
                coefs = np.polyfit(t, y, 3); return np.polyval(coefs, t)
            if name in ("moving_average", "ma"):
                w = int(params.get("window", 5))
                w = max(1, int(w))
                pad = w // 2
                ypad = np.pad(y, (pad, pad), mode="edge")
                kernel = np.ones(w) / w
                return np.convolve(ypad, kernel, mode="valid")

            # fallback cubic
            coefs = np.polyfit(t, y, 3); return np.polyval(coefs, t)

        # 2) residui per canale e residuo globale
        resid_mat = np.zeros((n, C), dtype=float)
        for c in range(C):
            y = np.asarray([float(np.asarray(v)[c]) for _, v in X_mv], dtype=float)
            yhat = _predict_yhat_for_series(y)
            resid_mat[:, c] = np.abs(y - yhat)

        # residuo globale per timestamp
        resid_global = np.linalg.norm(resid_mat, axis=1)  # shape (n,)

        # 3) trasformazione decrescente per ottenere L_{i,c}
        # usa sigma robusta per ogni canale + (opzionale) per globale
        def _sigma_robust(res):
            med = np.median(res)
            mad = np.median(np.abs(res - med))
            sig = 1.4826 * mad
            if not np.isfinite(sig) or sig <= 1e-12:
                sig = max(1e-6, float(np.std(res)))
            return sig

        sig_c = np.array([_sigma_robust(resid_mat[:, c]) for c in range(C)], dtype=float)
        sig_g = _sigma_robust(resid_global)

        tr = (transform or "rational").lower()
        lam = float(lambda_global)

        L_flat = np.zeros(n * C, dtype=float)

        for i in range(n):
            for c in range(C):
                # residuo "multivariato soft"
                r_ic = (1.0 - lam) * resid_mat[i, c] + lam * resid_global[i]

                # denom (per-canale o globale? minimal: usa sigma del canale, ma puoi anche usare mix)
                denom = max(rho * sig_c[c], 1e-12)

                if tr == "rational":
                    L = 1.0 / (1.0 + r_ic / denom)
                elif tr == "exp":
                    L = float(np.exp(-r_ic / denom))
                else:
                    L = max(0.0, 1.0 - r_ic / denom)

                k = self._flat_index(i, c, C)
                L_flat[k] = float(L)

        return L_flat








    def matrix_M_mv(self, X_mv, transform="rational", rho=1.0, lambda_global=0.0):
        import numpy as np
        n = len(X_mv)
        C = self._n_channels_from_Xmv(X_mv)
        N = n * C

        L = self.diag_M_mv(X_mv, transform=transform, rho=rho, lambda_global=lambda_global)
        Q = self.off_diag_M_mv(X_mv)

        M = np.zeros((N, N), dtype=float)
        np.fill_diagonal(M, L)
        M += Q  # Q è già simmetrica con zeri in diagonale

        if getattr(self, "debug", False):
            print("(debug) L_flat:\n", L)
            print("(debug) M shape:", M.shape)

        return M






    def solve_qubo_mv(self, X_mv, transform="rational", rho=1.0, lambda_global=0.0):
        import numpy as np
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer

        M = self.matrix_M_mv(X_mv, transform=transform, rho=rho, lambda_global=lambda_global)
        N = M.shape[0]

        linear_terms = float(self.alpha) * np.diag(M)
        quadratic_terms = {}
        for i in range(N):
            for j in range(i + 1, N):
                if M[i, j] != 0.0:
                    quadratic_terms[(f"x{i}", f"x{j}")] = float(self.beta) * float(M[i, j])

        qubo = QuadraticProgram()
        for i in range(N):
            qubo.binary_var(name=f"x{i}")
        qubo.minimize(linear=linear_terms, quadratic=quadratic_terms)

        qaoa_solver = _build_qaoa_solver(self)
        meo = MinimumEigenOptimizer(qaoa_solver)
        result = meo.solve(qubo)

        top_n = int(getattr(self, "top_n_samples", 5))
        top_n_states = []
        if hasattr(result, "samples") and result.samples:
            for s in list(result.samples)[:top_n]:
                top_n_states.append([int(round(b)) for b in s.x])
        else:
            top_n_states = [[int(round(b)) for b in result.x]]

        return top_n_states, result.variables_dict










    def decode_centers_mv(self, X_mv, state_bits, threshold=0.5):
        import numpy as np
        n = len(X_mv)
        C = self._n_channels_from_Xmv(X_mv)

        centers = []  # lista di (t, c, y_c)
        for k, b in enumerate(state_bits):
            if b >= threshold:
                i, c = self._unflat_index(k, C)
                t, vec = X_mv[i]
                centers.append((int(t), int(c), float(np.asarray(vec)[c])))
        return centers




    def centers_storage_mv(self, X_mv, state=None, threshold=0.5,
                        transform="rational", rho=1.0, lambda_global=0.0):
        """
        MV version: returns centers as (t, c, y_c) for bits set to 1.
        If state is None -> uses the best/first returned state from solve_qubo_mv.
        """
        if state is None:
            top_states, _ = self.solve_qubo_mv(
                X_mv, transform=transform, rho=rho, lambda_global=lambda_global
            )
            state = top_states[0]  # best candidate

        centers = self.decode_centers_mv(X_mv, state_bits=state, threshold=threshold)

        if getattr(self, "debug", False):
            print("selected_centers_mv:", centers)

        return centers
