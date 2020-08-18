import numpy as np

a = np.zeros((15))

for ii in range(1000000):
# traj_len 7 
    s_idx = np.random.randint(9)
    e_idx = s_idx + 7

    a[s_idx:e_idx] += 1

print(a)