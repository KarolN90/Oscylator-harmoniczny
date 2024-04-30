# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 20:07:16 2024

@author: karol
"""

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# Stałe
m = 1   # masa
c = 0.5   # tłumienie
k = 10   # sztywnosć

# Warunki początkowe
y_0 = 1     # y(0) = 1

# Rozważany przedział czasu 0 - T
T = 10

# Rozwiązanie dokładne
def analityczny_oscylator(t):
    A = y_0
    omega = np.sqrt(k/m - c**2/4*m**2)
    phi = 0
    y_an = A * np.exp(-c*t/2*m) * np.cos(omega * t + phi)
    return y_an


# Definicja równania
def ode_oscillator(t, y):
    dy_dt = dde.grad.jacobian(y, t)
    d2y_dt2 = dde.grad.hessian(y, t)
    return m * d2y_dt2 + c * dy_dt + k * y

# Definicja domeny
geom = dde.geometry.TimeDomain(0, T)

# Warunki brzegowe

## y(t=0) = 1
ic1 = dde.icbc.IC(geom, lambda y: y_0, lambda _, on_initial: on_initial)

## dy/dt (0) = 0
def boundary_l(t, on_initial):
    return on_initial and np.isclose(t[0], 0)
def bc_func2(inputs, outputs, X):
    return dde.grad.jacobian(outputs, inputs, i=0, j=None)
ic2 = dde.icbc.OperatorBC(geom, bc_func2, boundary_l)

data = dde.data.PDE(
    geom,
    ode_oscillator,
    [ic1, ic2],
    num_domain = 200,
    num_boundary = 3,
    num_test = 200 
    )

# Definicja sieci neuronowej

depth = 3
nodes = 50
layer_size = [1] + [nodes] *depth + [1]
activation = 'tanh'
# initializer = 'Glorot normal'   'Glorot uniform'
initializer = 'Glorot uniform'

net = dde.maps.FNN(layer_size, activation, initializer)

# Budowa modelu (wybór optymalizatora, współczynnik uczenia, ilosć epok itd.)

model = dde.Model(data, net)
# loss_weights[PDE,ic1,ic2]
model.compile('adam', lr=0.01, loss_weights=[1, 1, 1], decay=("inverse time", 2500, 0.9))
# model.compile('adam', lr=0.01, decay=("inverse time", 2500, 0.9))
losshistory, train_state = model.train(iterations=20000, display_every=500)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

samples = geom.random_points(500)
result = model.predict(samples)
# Sortowanie próbek i wyników według wartości czasu t
sorted_indices = np.argsort(samples[:,0])
sorted_samples = samples[sorted_indices]
sorted_results = result[sorted_indices]

# print(result)

# Rozwiązanie analityczne

t = np.linspace(0, T, 500)
y_an = analityczny_oscylator(t)


# Narysowanie wykresu
plt.figure(dpi=200)
plt.plot(t, y_an, color='red', label='Analityczne')
plt.plot(sorted_samples[:,0], sorted_results[:,0], color='blue', label='PINN')
# plt.scatter(samples[:,0], result[:,0], s=1, marker='o', color='blue', label='PINN')
plt.xlabel('Czas')
plt.ylabel('y')
plt.title('Wykres przemieszczeń oscylatora')
plt.legend()
plt.grid(True)
plt.show()


