import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sampling_sgd import generate, StochasticLinearRegressor
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
X_train = np.loadtxt("X_train.csv", delimiter=",")  
y_train = np.loadtxt("y_train.csv", delimiter=",")


trajectories = {}
batch_sizes = [1, 80, 8000, 800000]
lr=0.001
n_epochs=[100,100,200,2000]
eps=[1e-3,1e-4,1e-5,1e-5]


for i in range(4):
    model = StochasticLinearRegressor()
    params_history = model.fit(X_train, y_train, learning_rate=lr, batch_size=batch_sizes[i], n_epochs=n_epochs[i],epsilon=eps[i])
    trajectories[batch_sizes[i]] = params_history

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for batch_size in batch_sizes:
    traj = trajectories[batch_size]
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f'Batch size {batch_size}')
    ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], marker='o', color='black', s=50)  
    ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], marker='X', color='red', s=50) 

ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.set_zlabel(r'$\theta_2$')
ax.set_title('Trajectory of θ updates in 3D Parameter Space')
ax.legend()
plt.show()

