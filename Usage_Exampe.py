import numpy as np
from fixes import _safe_svd
# Simple Example M : 2 Sensor, 4 time points
M = np.array([
    [1.0, 4.0, 2.0, 3.2],
    [3.1, 123.0, 1.3, 0.3]
])
#Lead filed 2 sensor * 3 source
G = np.array([
    [1.0, 4.0, 2.0],
    [3.1, 1.0, 1.3]
])
alpha=0.5
#Normalize data
NC=np.linalg.norm(np.dot(M,M.T),ord="fro") #Frobenius Norm
M/=np.sqrt(NC)
NC=np.linalg.norm(G,ord=np.inf)
G/=NC
eps = np.finfo(float).eps
#Initialize
n_sensor, n_source=G.shape
n_active=n_source
active_set=np.arange(n_source) #all indices of sorts initially all active
gammas = np.ones(G.shape[1], dtype=np.float64)
#gammas[1]=0.001
#ammas[2]=1e-17                #less than eps.
#_________________iteration
for itno in range(1000):
    gidx=np.abs(gammas)>eps
    active_set=active_set[gidx]
    gammas=gammas[gidx]
    #if pruned any source:
    if n_active>len(active_set):
        n_active=active_set.size
        G=G[:,gidx]  #drop columns (sources) from Leadfield matrix

    CM=np.dot(G*gammas[np.newaxis,:],G.T)
    #adds the regularization parameter alpha to each diagonal element of CM
    CM.flat[::n_sensor+1]+=alpha
    #SVM : U: singular vectors, S: singular values
    #U, S, _ = _safe_svd(CM, full_matrices=False)
    #S = S[np.newaxis, :]
    #inv = np.dot(U / (S + eps), U.T)
    comp_inv=np.linalg.pinv(CM)  #instead of doing SVM with _safe_svd this library to the same to compute pseudo inverse
    mor_inv=np.linalg.inv(CM) #Need to be n*n matrix -same result as psuedo-inverse for n*n othewise raise error
    #by using pinv _safe_svd can removed





print("normalized observed \n",M,'\n')
print("normalized leadfiled\n",G)
print("\n Active set=",active_set,",  Gammas=",gammas)
print(f"CM=",CM)
print(f"left singular vecors=",U,"singular values=",S)
print(f"inverseCM=",inv,"inverse from np",mor_inv)
