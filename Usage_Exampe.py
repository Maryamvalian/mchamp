import numpy as np
#Example M : 2 Sensor, 4 time points
M = np.array([
    [1.0, 4.0, 2.0, 3.2],
    [3.1, 123.0, 1.3, 0.3]
])
#Lead filed 2 sensor * 3 source
G = np.array([
    [1.0, 4.0, 2.0],
    [3.1, 1.0, 1.3]
])
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
#ammas[2]=1e-17
#_________________iteration
for itno in range(1000):
    gidx=np.abs(gammas)>eps
    active_set=active_set[gidx]
    gammas=gammas[gidx]




print("normalized observed \n",M,'\n')
print("normalized leadfiled\n",G)
print("\n Active set=",active_set,",  Gammas=",gammas)
print(gidx)
