import jax.numpy as np

t0 = 0
tf = 0.3 # Step Period
dt = 0.005
N = round((tf-t0)/dt)

tt = np.arange(t0, tf + dt, dt)
theta_H = np.pi/6