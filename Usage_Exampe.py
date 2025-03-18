import numpy as np
#Example M : 2 Sensor, 4 time points
M = np.array([
    [1.0, 4.0, 2.0, 3.2],
    [3.1, 123.0, 1.3, 0.3]
])
#lead filed 2 sensor * 3 source
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

print("normalized observed \n",M,'\n')
print("normalized leadfiled\n",G)
