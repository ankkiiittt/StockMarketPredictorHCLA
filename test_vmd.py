import numpy as np
from vmdpy import VMD

# dummy signal
f = np.random.randn(100) + np.sin(np.linspace(0, 10, 100))

alpha = 2000       # moderate bandwidth constraint  
tau = 0.           # noise-tolerance (no strict fidelity enforcement)  
K = 3              # 3 modes  
DC = 0             # no DC part imposed  
init = 1           # initialize omegas uniformly  
tol = 1e-7  

u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)
print(f"Modes shape: {u.shape}")
print(f"Original shape: {f.shape}")

denoised = np.sum(u[:-1, :], axis=0) # remove highest frequency mode (last mode)
print(f"Denoised shape: {denoised.shape}")
