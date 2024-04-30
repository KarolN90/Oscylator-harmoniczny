# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:42:25 2024

@author: karol
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import datetime
from Oscylator_analityczny_wymuszony_v1 import analityczny_oscylator_wymuszony


# Stałe
m = 1   # masa
c = 0   # tłumienie
k = 10   # sztywnosć

# Wymuszenie
F0 = 2       # Amplituda siły wymuszającej
omega_f = 2  # Częstość kołowa siły wymuszającej

# Warunki początkowe
y_0 = 1     
v_0 = 0

# Rozważany przedział czasu 0 - T
T = 20
# podział przedziału czasowego
n = 600   

# Ilosć iteracji
max_iter = 90000  

# Informacja o iteracji
info_iter = 5000

# Współczynnik uczenia
lr = 1e-2

# Manualny start generatora liczb pseudolosowych
torch.manual_seed(1)

# Definicja sieci neuronowej
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.net = torch.nn.Sequential(
            nn.Linear(1,20),    # 1 input / 20 output sieć liniowa, warstwa początkowa
            nn.Tanh(),  # funkcja aktywacji
            nn.Linear(20,30),    # 20 input / 30 output sieć liniowa, warstwa 2
            nn.Tanh(),  # funkcja aktywacji
            nn.Linear(30,30),    # 30 input / 30 output sieć liniowa, warstwa 3
            nn.Tanh(),  # funkcja aktywacji
            nn.Linear(30,20),    # 30 input / 20 output sieć liniowa, warstwa 4
            nn.Tanh(),  # funkcja aktywacji
            nn.Linear(20,20),    # 20 input / 20 output sieć liniowa, warstwa 5
            nn.Tanh(),  # funkcja aktywacji
            nn.Linear(20,1)       # 20 input / 1 output sieć liniowa, warstwa końcowa
            )
    def forward(self, x):
        out = self.net(x)
        return out

# Definicja domeny 
t_domain = torch.linspace(0,1,n).view(-1,1).requires_grad_(True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = NN().to(device) # Deklaracja modelu
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

# Zmiana współczynnika uczenia w czasie
scheduler = StepLR(optimizer, step_size=5000, gamma=0.97)

results = []
loss_history = []

# Start obliczeń
t1 = datetime.datetime.now()

for i in range(max_iter):
    optimizer.zero_grad()
    
    if i < 15000:
        # Wagi poszczególnych strat [PDE, IC1, IC2]
        loss_weights = [0.5, 1, 1]
    else:
        # Wagi poszczególnych strat [PDE, IC1, IC2]
        loss_weights = [1, 0.5, 0.5]
    
    # Strata z równania różniczkowego
    y = model(t_domain)
    dy_dt  = torch.autograd.grad(y, t_domain, torch.ones_like(y), create_graph=True)[0]  # dy/dt
    d2y_dt2 = torch.autograd.grad(dy_dt,  t_domain, torch.ones_like(dy_dt),  create_graph=True)[0]   # d^2y/dt^2
    ODE = m * d2y_dt2 + c * dy_dt + k * y - F0 * torch.sin(omega_f * t_domain)
    loss_ODE = loss_weights[0] * torch.mean(ODE**2)
    
    # Strata z warunku początkowego y(0) = y_0
    IC1 = y[0] - y_0
    loss_IC1 = loss_weights[1] * torch.mean(IC1**2)
    
    # Strata z warunku początkowego dy/dt (t=0) = v_0
    IC2 = dy_dt[0] - v_0
    loss_IC2 = loss_weights[2] * torch.mean(IC2**2)
    
    # Strata całkowita
    loss = loss_ODE + loss_IC1 + loss_IC2
    loss.backward()
    optimizer.step()
    
    if i > 15000:
        scheduler.step()
    
    yr = model(t_domain).detach()
    results.append(yr)
    lh = [i, float(loss), float(loss_ODE), float(loss_IC1), float(loss_IC2)]
    loss_history.append(lh)
    
    if (i+1) % info_iter == 0:
        print(f'Iteracja {i+1}, loss: {loss}, loss_ODE: {loss_ODE}, loss_IC1: {loss_IC1}, loss_IC2: {loss_IC2}')

# Koniec obliczeń
t2 = datetime.datetime.now()
tw = t2 - t1
print(f'Czas obliczeń: {tw.total_seconds()}s')

# Próbki czasu
t = np.linspace(0, 1, n)

# Rozwiązanie analityczne
y_an = analityczny_oscylator_wymuszony(t=t*T, m=m, c=c, k=k, F0=F0, omega_f=omega_f, T=T, n=n, y_0=y_0, v_0=v_0)

# Wykres przemieszczeń
y = model(t_domain).detach()
plt.figure(dpi=200)
plt.plot(t, y_an, color='red', label='Analityczne', zorder=1, alpha=0.65)
plt.plot(t, y, color='blue', label='PINN', zorder=0.8)
for i in range(int(max_iter/info_iter)):
    yr = results[i*info_iter].tolist()
    ir = loss_history[i*info_iter][0]
    plt.plot(t, yr, color=plt.cm.viridis(i*25), label = f'Iteracja: {ir}', linewidth=0.7, zorder=0.2)
plt.xlabel('Bezwymiarowy czas')
plt.ylabel('y')
plt.title('Wykres przemieszczeń oscylatora wymuszonego')
plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1), shadow=True, ncol=1)
plt.grid(True)
plt.show()

# Wykres strat
loss_history = np.array(loss_history)
plt.figure(dpi=200)
plt.plot(loss_history[:,0], loss_history[:,1], color='red', label='Total loss', linewidth=0.6, zorder=1)
plt.plot(loss_history[:,0], loss_history[:,2], color='green', label='ODE loss', linewidth=0.3, zorder=0)
plt.plot(loss_history[:,0], loss_history[:,3], color='blue', label='IC1 loss', linewidth=0.3, zorder=0)
plt.plot(loss_history[:,0], loss_history[:,4], color='yellow', label='IC2 loss', linewidth=0.3, zorder=0)
plt.xlabel('Iteracja')
plt.yscale('log')
plt.title('Błąd')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), shadow=True, ncol=1)
plt.grid(True)
plt.show()
