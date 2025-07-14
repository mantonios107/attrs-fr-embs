import numpy as np
from plotly.subplots import make_subplots
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram
from gtda import diagrams


# Colors used for plotting diagrams
cols = ['#005f73', '#0a9396', '#ca6702','#ffc300']

## Redefine the max function (needed to solve a bug in th giotto-tda package)
def new_max(A):
  return np.max(A)


def compute_plot_diagrams(data,
                          metric = "euclidean",
                          homology_dimensions=[0,1],
                          rescale = False,
                          plot = False,
                          names = None,
                          show = True,):

  ## Compute Vietoris-Rips persistent homology of a given point cloud or distance matrix
  ## INPUTS
  # data: list of 2-dimensional numpy arrays of size (n,d) containing the point clouds to be processed  
  # metric: string specifying the metric used to compute distances, can be "euclidean", "cosine" 
  #         or "precomputed" if data is a distance matrix
  # homology_dimensions: list of integers containing the homology dimensions to be computed
  # rescale: boolean, whether to rescale the diagrams
  # plot: boolean, whether to plot the resulting diagrams and save the figures
  # names: list of strings, labels of the different diagrams 
  # show: boolean, whether to show the plots of the diagrams
  #
  # OUTPUTS
  # p_diagrams: list of persistence diagrams
  # fig: list of figures, if plot is False then fig is None

  persistence = VietorisRipsPersistence(
      metric = metric,
      homology_dimensions=homology_dimensions,
      n_jobs =-1,
      reduced_homology=True,
  )

  p_diagrams = persistence.fit_transform(data)

  # Rescale the diagrams
  if rescale:
    diagramScaler = diagrams.Scaler(function=new_max)
    p_diagrams = diagramScaler.fit_transform(p_diagrams)


  if plot:
    fs = []
    for n in range(len(data)):
      fs.append(plot_diagram(p_diagrams[n]))

    fig = make_subplots(rows=1, cols=len(data), subplot_titles=names)

    ## Add each figure to a subplot
    for i, fig_obj in enumerate(fs, start=1):
        for trace in fig_obj.data:
          fig.add_trace(trace, row=1, col=i)

    for j in homology_dimensions:
      fig.update_traces(
          marker=dict(color=cols[j]),
          selector=dict(type="scatter", name=f"H{j}"))

    fig.update_layout(width=300*len(p_diagrams),height = 400)
    if show:
      fig.show()

  else:
    fig = None

  return p_diagrams, fig



## PERSISTENT HOMOLOGY DIMENSION
# Functions necessary to compute the persistent homology dimension of a point cloud
# The code is taken from https://github.com/tolgabirdal/PHDimGeneralization

from scipy.spatial.distance import cdist
from threading import Thread

MIN_SUBSAMPLE = 20
INTERMEDIATE_POINTS = 5

def prim_tree(adj_matrix, alpha=1.0):
    infty = np.max(adj_matrix) + 10

    dst = np.ones(adj_matrix.shape[0]) * infty
    visited = np.zeros(adj_matrix.shape[0], dtype=bool)
    ancestor = -np.ones(adj_matrix.shape[0], dtype=int)

    v, s = 0, 0.0
    for i in range(adj_matrix.shape[0] - 1):
        visited[v] = 1
        ancestor[dst > adj_matrix[v]] = v
        dst = np.minimum(dst, adj_matrix[v])
        dst[visited] = infty

        v = np.argmin(dst)
        s += (adj_matrix[v][ancestor[v]] ** alpha)

    return s.item()

def process_string(sss):
    return sss.replace('\n', ' ').replace('  ', ' ')

class PHD():
    def __init__(self, alpha=1.0, metric='euclidean', n_reruns=3, n_points=7, n_points_min=3):
      '''
      Initializes the instance of PH-dim computer
      Parameters:
        1) alpha --- real-valued parameter Alpha for computing PH-dim (see the reference paper). Alpha should be chosen lower than
      the ground-truth Intrinsic Dimensionality; however, Alpha=1.0 works just fine for our kind of data.
        2) metric --- String or Callable, distance function for the metric space (see documentation for Scipy.cdist)
        3) n_reruns --- Number of restarts of whole calculations (each restart is made in a separate thread)
        4) n_points --- Number of subsamples to be drawn at each subsample
        5) n_points_min --- Number of subsamples to be drawn at larger subsamples (more than half of the point cloud)
      '''
      self.alpha = alpha
      self.n_reruns = n_reruns
      self.n_points = n_points
      self.n_points_min = n_points_min
      self.metric = metric
      self.is_fitted_ = False

    def _sample_W(self, W, nSamples):
        n = W.shape[0]
        random_indices = np.random.choice(n, size=nSamples, replace=False)
        return W[random_indices]

    def _calc_ph_dim_single(self, W, test_n, outp, thread_id):
        lengths = []
        for n in test_n:
            if W.shape[0] <= 2 * n:
                restarts = self.n_points_min
            else:
                restarts = self.n_points

            reruns = np.ones(restarts)
            for i in range(restarts):
                tmp = self._sample_W(W, n)
                reruns[i] = prim_tree(cdist(tmp, tmp, metric=self.metric), self.alpha)

            lengths.append(np.median(reruns))
        lengths = np.array(lengths)

        x = np.log(np.array(list(test_n)))
        y = np.log(lengths)
        N = len(x)
        outp[thread_id] = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)

    def fit_transform(self, X, y=None, min_points=50, max_points=512, point_jump=40):
      '''
      Computing the PH-dim
      Parameters:
      1) X --- point cloud of shape (n_points, n_features),
      2) y --- fictional parameter to fit with Sklearn interface
      3) min_points --- size of minimal subsample to be drawn
      4) max_points --- size of maximal subsample to be drawn
      5) point_jump --- step between subsamples
      '''
      ms = np.zeros(self.n_reruns)
      test_n = range(min_points, max_points, point_jump)
      threads = []

      for i in range(self.n_reruns):
          threads.append(Thread(target=self._calc_ph_dim_single, args=[X, test_n, ms, i]))
          threads[-1].start()

      for i in range(self.n_reruns):
          threads[i].join()

      m = np.mean(ms)
      return 1 / (1 - ms)

    def get_phd_single(self, input,):

        mx_points = input.shape[0]
        mn_points = MIN_SUBSAMPLE
        step = ( mx_points - mn_points ) // INTERMEDIATE_POINTS

        return self.fit_transform(X=input,min_points=mn_points, max_points=mx_points - step, point_jump=step)
