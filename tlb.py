import ot
import numpy as np
import scipy as sp
import utils
import tqdm
# Computes the third lower bound of Mémoli 2011 as a hierrachical transport

def tlb(c1, c2, p = 2, lambd = 1e-1):
    """
    Computes Mémoli's third lower bound as a hierarchical transport from the cost matrices of two distributions.
    """
    n = c1.shape[0]
    m = c2.shape[0]
    c = abs(np.subtract.outer(c1, c2))
    a = np.ones(n)
    b = np.ones(m)

    # # time the loop with tqdm
    # c_final = np.zeros((n, m))
    # for i in tqdm.tqdm(range(n)):
    #     for j in range(m):
    #         c_final[i, j] = ot.emd2(a, b, c[i, :, j, :], lambd)

    # Can be parallelized
    c_final = [[ot.emd2(a, b, c[i, :, j, :], lambd) for j in range(m)] for i in range(n)]
    P = ot.sinkhorn(a, b, c_final, lambd)
    return P


n = 50
m = 50

mu_s = np.array([0, 0])
cov_s = np.array([[1, 0], [0, 1]])

mu_t = np.array([4, 4])
cov_t = np.array([[1, -.8], [-.8, 1]])

source_samples = ot.datasets.make_2D_samples_gauss(n, mu_s, cov_s)
target_samples = ot.datasets.make_2D_samples_gauss(m, mu_t, cov_t)

c1 = sp.spatial.distance.cdist(source_samples, source_samples)
c2 = sp.spatial.distance.cdist(target_samples, target_samples)

tlb = tlb(c1, c2)
gw, log = ot.gromov.entropic_gromov_wasserstein(
    c1, c2, loss_fun='square_loss', epsilon=1e-1, solver='PPA',
    log=True, verbose=True)
tlb