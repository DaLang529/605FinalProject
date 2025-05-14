import numpy as np
import pandas as pd
import time
import os
import psutil
import ctypes
from typing import Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans as SKKMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from concurrent.futures import ThreadPoolExecutor

# Configure seaborn
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
warnings.filterwarnings('ignore')

# Constants
LIB_NAME = "kmeans_openmp.so"  # Use .dll on Windows

METHOD_COLORS = {
    'single': '#636EFA',  
    'multi': '#00CC96',   
    'sklearn': '#EF553B'  
}

class KMeansBase:
  
    def __init__(
        self, 
        n_clusters: int = 8, 
        max_iter: int = 300, 
        tol: float = 1e-4, 
        random_state: Optional[int] = None,
        n_init: int = 10
    ):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        
    def _check_fit_data(self, X):

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
        n_samples, n_features = X.shape
        if n_samples < self.n_clusters:
            raise ValueError(f"n_samples={n_samples} should be >= n_clusters={self.n_clusters}")
        return X
    
    def _initialize_centroids(self, X):

        n_samples = X.shape[0]
        if self.random_state is not None:
            np.random.seed(self.random_state)
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return X[indices].copy()
    
    def _compute_inertia(self, X, labels, centroids):

        n_samples = X.shape[0]
        inertia = 0.0
        for i in range(n_samples):
            centroid = centroids[labels[i]]
            diff = X[i] - centroid
            inertia += np.sum(diff * diff)
        return inertia
    
    def fit(self, X):
        raise NotImplementedError("Subclasses must implement fit method")
    
    def predict(self, X):
 
        if self.centroids_ is None:
            raise ValueError("Model not fitted yet. Call 'fit' before using 'predict'.")
        
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=np.int32)
        
        for i in range(n_samples):
            min_dist = float('inf')
            min_cluster = 0
            for j in range(self.n_clusters):
                dist = np.sum((X[i] - self.centroids_[j]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    min_cluster = j
            labels[i] = min_cluster
            
        return labels
    
    def fit_predict(self, X):

        self.fit(X)
        return self.labels_


class KMeansSingle(KMeansBase):
 
    def fit(self, X):

        X = self._check_fit_data(X)
        n_samples, n_features = X.shape
        
        best_inertia = float('inf')
        best_centroids = None
        best_labels = None
        best_n_iter = 0
        
        for init in range(self.n_init):
            if self.random_state is not None:
                seed = self.random_state + init
            else:
                seed = None
            
            centroids = self._initialize_centroids(X)
            labels = np.zeros(n_samples, dtype=np.int32)

            for iteration in range(self.max_iter):

                distances = np.zeros((n_samples, self.n_clusters))
                for c in range(self.n_clusters):
                    diff = X - centroids[c]
                    distances[:, c] = np.sum(diff * diff, axis=1)
                new_labels = np.argmin(distances, axis=1)

                if np.all(new_labels == labels):
                    break
                    
                labels = new_labels

                new_centroids = np.zeros((self.n_clusters, n_features))
                for c in range(self.n_clusters):
                    mask = labels == c
                    if np.any(mask):
                        new_centroids[c] = X[mask].mean(axis=0)
                    else:
                        idx = np.random.randint(n_samples)
                        new_centroids[c] = X[idx]

                centroid_shift = np.sum((centroids - new_centroids) ** 2)
                centroids = new_centroids
                
                if centroid_shift < self.tol:
                    break

            inertia = self._compute_inertia(X, labels, centroids)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids.copy()
                best_labels = labels.copy()
                best_n_iter = iteration + 1
        
        self.centroids_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        
        return self


class KMeansMulti(KMeansBase):
  
    def __init__(
        self, 
        n_clusters=8, 
        max_iter=300, 
        tol=1e-4, 
        random_state=None,
        n_init=10,
        lib_path=None
    ):

        super().__init__(n_clusters, max_iter, tol, random_state, n_init)

        self.lib_path = lib_path or os.path.join(os.path.dirname(__file__), LIB_NAME)
        self._load_library()
    
    def _load_library(self):

        try:
            self.kmeans_lib = ctypes.cdll.LoadLibrary(os.path.abspath(self.lib_path))

            self.kmeans_lib.kmeans_omp.argtypes = [
                ctypes.POINTER(ctypes.c_double),  
                ctypes.c_int,                     
                ctypes.c_int,                     
                ctypes.c_int,                     
                ctypes.c_int,                     
                ctypes.c_double,                  
                ctypes.POINTER(ctypes.c_int),     
                ctypes.POINTER(ctypes.c_double),  
                ctypes.c_int                      
            ]
            self.kmeans_lib.kmeans_omp.restype = None
            
        except (OSError, AttributeError) as e:
            raise ImportError(f"Failed to load OpenMP library: {e}. Make sure it's compiled correctly.")
    
    def fit(self, X):

        X = self._check_fit_data(X)
        n_samples, n_features = X.shape
        
        best_inertia = float('inf')
        best_centroids = None
        best_labels = None
        
        for init in range(self.n_init):
            if self.random_state is not None:
                seed = self.random_state + init
            else:
                seed = np.random.randint(0, 2**31 - 1)
                
            X_c = X.astype(np.float64)
            data_ptr = X_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            
            labels = np.zeros(n_samples, dtype=np.int32)
            labels_ptr = labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            
            centroids = np.zeros(self.n_clusters * n_features, dtype=np.float64)
            centroids_ptr = centroids.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            

            self.kmeans_lib.kmeans_omp(
                data_ptr,
                ctypes.c_int(n_samples),
                ctypes.c_int(n_features),
                ctypes.c_int(self.n_clusters),
                ctypes.c_int(self.max_iter),
                ctypes.c_double(self.tol),
                labels_ptr,
                centroids_ptr,
                ctypes.c_int(seed)
            )
            
            centroids = centroids.reshape((self.n_clusters, n_features))
            

            inertia = self._compute_inertia(X, labels, centroids)
            

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids.copy()
                best_labels = labels.copy()
        
        self.centroids_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        
        return self


class KMeansSK(KMeansBase):

    
    def fit(self, X):

        X = self._check_fit_data(X)
    
        model = SKKMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            n_init=self.n_init
        )
        model.fit(X)
        
        
        self.centroids_ = model.cluster_centers_.copy()
        self.labels_ = model.labels_.copy()
        self.inertia_ = model.inertia_
        self.n_iter_ = model.n_iter_
        
        return self


class DataGenerator:

    
    @staticmethod
    def make_blobs_data(
        n_samples=1000, 
        n_features=2, 
        n_clusters=5, 
        cluster_std=1.0,
        random_state=None
    ):

        X, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_clusters,
            cluster_std=cluster_std,
            random_state=random_state
        )
        return X, y
    
    @staticmethod
    def make_anisotropic_blobs(
        n_samples=1000,
        n_features=2,
        n_clusters=5,
        random_state=None
    ):

        X, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_clusters,
            random_state=random_state
        )
        
        if random_state is not None:
            np.random.seed(random_state)
            
        transformation = np.random.randn(n_features, n_features)
        X = np.dot(X, transformation)
        
        return X, y


class BenchmarkMetrics:
    
    
    @staticmethod
    def measure_time(func, *args, **kwargs):
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    @staticmethod
    def measure_memory(func, *args, **kwargs):

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        result = func(*args, **kwargs)
        mem_after = process.memory_info().rss
        mem_used = (mem_after - mem_before) / (1024 * 1024)  # MB
        return result, mem_used
    
    @staticmethod
    def estimate_energy(func, *args, **kwargs):

        process = psutil.Process(os.getpid())
        cpu_percent_before = psutil.cpu_percent(interval=None)
        
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        cpu_percent_after = psutil.cpu_percent(interval=None)
        avg_cpu_percent = (cpu_percent_before + cpu_percent_after) / 2
        
        # Very rough energy estimate (cpu% * time)
        energy_estimate = avg_cpu_percent * elapsed_time
        
        return result, energy_estimate
    
    @staticmethod
    def evaluate_clustering(X, labels):

        metrics = {}

        try:
            metrics['silhouette'] = silhouette_score(X, labels)
        except:
            metrics['silhouette'] = float('nan')

        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        except:
            metrics['calinski_harabasz'] = float('nan')

        try:
            metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
        except:
            metrics['davies_bouldin'] = float('nan')
            
        return metrics


class KMeansBenchmark:

    
    def __init__(self, random_state=42):

        self.random_state = random_state
        self.results = []
        
    def run_single_benchmark(
        self, 
        X, 
        method='single', 
        n_clusters=5, 
        max_iter=300, 
        tol=1e-4
    ):

        n_samples, n_features = X.shape
        result = {
            'method': method,
            'n_samples': n_samples,
            'n_features': n_features,
            'n_clusters': n_clusters
        }
        

        if method == 'single':
            model = KMeansSingle(
                n_clusters=n_clusters,
                max_iter=max_iter,
                tol=tol,
                random_state=self.random_state,
                n_init=1  
            )
        elif method == 'multi':
            model = KMeansMulti(
                n_clusters=n_clusters,
                max_iter=max_iter,
                tol=tol,
                random_state=self.random_state,
                n_init=1  
            )
        elif method == 'sklearn':
            model = KMeansSK(
                n_clusters=n_clusters,
                max_iter=max_iter,
                tol=tol,
                random_state=self.random_state,
                n_init=1  
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        

        _, time_taken = BenchmarkMetrics.measure_time(model.fit, X)
        result['time'] = time_taken
        

        _, mem_usage = BenchmarkMetrics.measure_memory(model.fit, X)
        result['mem_usage_mb'] = mem_usage
        

        _, energy_est = BenchmarkMetrics.estimate_energy(model.fit, X)
        result['energy_est'] = energy_est

        cluster_metrics = BenchmarkMetrics.evaluate_clustering(X, model.labels_)
        result.update(cluster_metrics)

        result['inertia'] = model.inertia_
        if hasattr(model, 'n_iter_'):
            result['n_iter'] = model.n_iter_
            
        return result
    
    def run_scaling_benchmark(
        self, 
        methods=['single', 'multi', 'sklearn'], 
        n_samples_range=[1000, 5000, 20000, 50000],
        n_features=2,
        n_clusters=5
    ):

        self.results = []
        

        for method in methods:
            for n_samples in n_samples_range:
                print(f"Running {method} with {n_samples} samples...")
                

                X, _ = DataGenerator.make_blobs_data(
                    n_samples=n_samples,
                    n_features=n_features,
                    n_clusters=n_clusters,
                    random_state=self.random_state
                )
                
                try:
                    result = self.run_single_benchmark(
                        X, 
                        method=method,
                        n_clusters=n_clusters
                    )
                    self.results.append(result)
                except Exception as e:
                    print(f"Error running {method} with {n_samples} samples: {e}")

        return pd.DataFrame(self.results)
    
    def plot_results(self, df=None, metrics=['time', 'mem_usage_mb', 'energy_est']):
        if df is None:
            df = pd.DataFrame(self.results)
            
        if df.empty:
            raise ValueError("No benchmark results to plot")
            
        figs = []
        
        for metric in metrics:
            if metric not in df.columns:
                warnings.warn(f"Metric {metric} not found in results")
                continue
                
            title = {
                'time': 'Execution Time vs. n_samples',
                'mem_usage_mb': 'Memory Usage vs. n_samples',
                'energy_est': 'Energy Estimate vs. n_samples',
                'silhouette': 'Silhouette Score vs. n_samples',
                'calinski_harabasz': 'Calinski-Harabasz Index vs. n_samples',
                'davies_bouldin': 'Davies-Bouldin Index vs. n_samples',
                'inertia': 'Inertia vs. n_samples'
            }.get(metric, f"{metric} vs. n_samples")
            
            y_label = {
                'time': 'Time (seconds)',
                'mem_usage_mb': 'Memory Usage (MB)',
                'energy_est': 'Energy Estimate (CPU%·s)',
                'silhouette': 'Silhouette Score',
                'calinski_harabasz': 'Calinski-Harabasz Index',
                'davies_bouldin': 'Davies-Bouldin Index',
                'inertia': 'Inertia'
            }.get(metric, metric)
            
            fig = px.line(
                df,
                x="n_samples",
                y=metric,
                color="method",
                markers=True,
                title=title,
                labels={'n_samples': 'Number of Samples', metric: y_label},
                color_discrete_map=METHOD_COLORS
            )
            
            figs.append(fig)
            
        return figs


class ClusteringVisualizer:

    
    @staticmethod
    def plot_2d_clusters(X, labels, centroids=None, title="K-means Clustering Results"):

        if X.shape[1] != 2:
            raise ValueError("This function only works with 2D data")
            
        df = pd.DataFrame({
            'x': X[:, 0],
            'y': X[:, 1],
            'cluster': labels.astype(str)
        })
        
        fig = px.scatter(
            df, 
            x='x', 
            y='y', 
            color='cluster',
            title=title
        )

        if centroids is not None:
            fig.add_trace(
                go.Scatter(
                    x=centroids[:, 0],
                    y=centroids[:, 1],
                    mode='markers',
                    marker=dict(
                        symbol='x',
                        size=15,
                        color='black',
                        line=dict(width=2)
                    ),
                    name='Centroids'
                )
            )
            
        fig.update_layout(
            legend_title_text='Cluster',
            xaxis_title='Feature 1',
            yaxis_title='Feature 2'
        )
        
        return fig
    
    @staticmethod
    def plot_comparison_dashboard(results, methods, n_samples, n_clusters=5, features=2):

        filtered = results[
            (results['n_clusters'] == n_clusters) &
            (results['n_features'] == features) &
            (results['method'].isin(methods)) &
            (results['n_samples'].isin(n_samples))
        ]
        
        if filtered.empty:
            raise ValueError("No matching results found")
        fig = make_subplots(
            rows=2, 
            cols=2,
            subplot_titles=(
                "Execution Time", 
                "Memory Usage",
                "Energy Estimate", 
                "Silhouette Score"
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        method_colors = {
            'single': '#636EFA',  
            'multi': '#00CC96',   
            'sklearn': '#EF553B'  
        }
        
        for method in methods:
            method_data = filtered[filtered['method'] == method]
            color = method_colors.get(method, '#636EFA')  
            
            fig.add_trace(
                go.Scatter(
                    x=method_data['n_samples'],
                    y=method_data['time'],
                    mode='lines+markers',
                    name=f"{method} - Time",
                    legendgroup=method,
                    showlegend=True,
                    line=dict(color=color)
                ),
                row=1, col=1
            )
            

            fig.add_trace(
                go.Scatter(
                    x=method_data['n_samples'],
                    y=method_data['mem_usage_mb'],
                    mode='lines+markers',
                    name=f"{method} - Memory",
                    legendgroup=method,
                    showlegend=False,
                    line=dict(color=color)
                ),
                row=1, col=2
            )
            

            fig.add_trace(
                go.Scatter(
                    x=method_data['n_samples'],
                    y=method_data['energy_est'],
                    mode='lines+markers',
                    name=f"{method} - Energy",
                    legendgroup=method,
                    showlegend=False,
                    line=dict(color=color)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=method_data['n_samples'],
                    y=method_data['silhouette'],
                    mode='lines+markers',
                    name=f"{method} - Silhouette",
                    legendgroup=method,
                    showlegend=False,
                    line=dict(color=color)
                ),
                row=2, col=2
            )
        

        fig.update_layout(
            title_text=f"K-means Performance Comparison (k={n_clusters}, features={features})",
            height=800,
            width=1200
        )
        

        fig.update_xaxes(title_text="Number of Samples", row=1, col=1)
        fig.update_xaxes(title_text="Number of Samples", row=1, col=2)
        fig.update_xaxes(title_text="Number of Samples", row=2, col=1)
        fig.update_xaxes(title_text="Number of Samples", row=2, col=2)
        
        fig.update_yaxes(title_text="Time (seconds)", row=1, col=1)
        fig.update_yaxes(title_text="Memory Usage (MB)", row=1, col=2)
        fig.update_yaxes(title_text="Energy Estimate", row=2, col=1)
        fig.update_yaxes(title_text="Silhouette Score", row=2, col=2)
        
        return fig



try:
    from ipywidgets import interact, FloatSlider, IntSlider, VBox, HBox, Button, Dropdown, Output
    HAS_IPYWIDGETS = True
except ImportError:
    HAS_IPYWIDGETS = False


if HAS_IPYWIDGETS:
    class InteractiveDashboard:

        
        def __init__(self):
            
            self.output = Output()
            

            self.n_samples_slider = IntSlider(
                value=10000,
                min=1000,
                max=50000,
                step=5000,
                description='n_samples',
                continuous_update=False
            )
            
            self.n_clusters_slider = IntSlider(
                value=5,
                min=2,
                max=20,
                step=1,
                description='n_clusters',
                continuous_update=False
            )
            
            self.method_dropdown = Dropdown(
                options=[
                    ("Single-Core", "single"),
                    ("Multi-Core", "multi"),
                    ("Scikit-Learn", "sklearn")
                ],
                value='single',
                description='Method'
            )
            
            self.n_samples_slider.observe(self._on_change, names='value')
            self.n_clusters_slider.observe(self._on_change, names='value')
            self.method_dropdown.observe(self._on_change, names='value')
            

            self._run_kmeans()
        
        def _on_change(self, _):

            self._run_kmeans()
        
        def _run_kmeans(self):

            n_samples = self.n_samples_slider.value
            n_clusters = self.n_clusters_slider.value
            method = self.method_dropdown.value
            
            with self.output:
                self.output.clear_output()
                print(f"Running K-means with n_samples={n_samples}, n_clusters={n_clusters}, method={method}...")

                X, _ = DataGenerator.make_blobs_data(
                    n_samples=n_samples,
                    n_features=2,
                    n_clusters=n_clusters,
                    random_state=42
                )
                

                benchmark = KMeansBenchmark(random_state=42)
                result = benchmark.run_single_benchmark(
                    X, 
                    method=method,
                    n_clusters=n_clusters
                )
                

                print("== Results ==")
                print(f"Execution Time: {result['time']:.4f} s")
                print(f"Memory Usage: {result['mem_usage_mb']:.4f} MB")
                print(f"Energy Estimate: {result['energy_est']:.4f}")
                print(f"Silhouette Score: {result['silhouette']:.4f}")
                

                if X.shape[1] == 2:
                    if method == 'single':
                        model = KMeansSingle(n_clusters=n_clusters, random_state=42)
                    elif method == 'multi':
                        model = KMeansMulti(n_clusters=n_clusters, random_state=42)
                    else:
                        model = KMeansSK(n_clusters=n_clusters, random_state=42)
                    
                    model.fit(X)
                    fig = ClusteringVisualizer.plot_2d_clusters(
                        X, 
                        model.labels_,
                        model.centroids_,
                        title=f"K-means Clustering ({method}, k={n_clusters}, n={n_samples})"
                    )
                    fig.show()
        
        def display(self):

            from IPython.display import display
            display(VBox([
                HBox([self.n_samples_slider, self.n_clusters_slider]),
                self.method_dropdown,
                self.output
            ]))

else:
    class InteractiveDashboard:

        
        def __init__(self):
            print("Warning: ipywidgets not available. Interactive dashboard disabled.")
            
        def display(self):
            print("Interactive dashboard requires ipywidgets.") 