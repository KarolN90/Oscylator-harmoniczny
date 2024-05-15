# -*- coding: utf-8 -*-
"""
Created on Sun May 12 14:02:51 2024

@author: karol
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 12 12:23:12 2024

@author: karol
"""

import numpy as np
import math
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
F0 = 0       # Amplituda siły wymuszającej
omega_f = 0  # Częstość kołowa siły wymuszającej

# Warunki początkowe
y_0 = 1     
v_0 = 0

# Rozważany przedział czasu 0 - T
T = 20
# podział przedziału czasowego
n = 200   

# Ilosć iteracji
max_iter = 100000  

# Informacja o iteracji
info_iter = 5000

# Współczynnik uczenia
lr = 1e-2

# Liczba neuronów w pojedynczej warstwie ukrytej
neu_nr = 50

# Manualny start generatora liczb pseudolosowych
torch.manual_seed(1)

# Inicjalizacja wag
def init_weights(m):
    if isinstance(m, nn.Linear):
        # Inicjalizacja wag przy użyciu rozkładu normalnego
        nn.init.normal_(m.weight, mean=0, std=0.1)
        #torch.nn.init.uniform_(m.weight, a=-0.3, b=0.3)
        # Inicjalizacja biasu
        nn.init.constant_(m.bias, 0.01)

# Definicja sieci neuronowej
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.net = torch.nn.Sequential(
            nn.Linear(1,neu_nr),    # 1 input / 20 output sieć liniowa, warstwa początkowa
            nn.Tanh(),  # funkcja aktywacji
            nn.Linear(neu_nr,neu_nr),    # 20 input / 30 output sieć liniowa, warstwa 2
            nn.Tanh(),  # funkcja aktywacji
            nn.Linear(neu_nr,neu_nr),    # 30 input / 30 output sieć liniowa, warstwa 3
            nn.Tanh(),  # funkcja aktywacji
            nn.Linear(neu_nr,neu_nr),    # 30 input / 30 output sieć liniowa, warstwa 3
            nn.Tanh(),  # funkcja aktywacji
            nn.Linear(neu_nr,neu_nr),    # 30 input / 30 output sieć liniowa, warstwa 3
            nn.Tanh(),  # funkcja aktywacji
            nn.Linear(neu_nr,1)       # 20 input / 1 output sieć liniowa, warstwa końcowa
            )
    def forward(self, x):
        out = self.net(x)
        return out


class RNN_NN(nn.Module):
    # Sieć rekurencyjna
    def __init__(self):
        super(RNN_NN, self).__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=neu_nr, num_layers=3, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(neu_nr, 1)

    def forward(self, x):
        # x - dane wejściowe w formie (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size, seq_len, 1)

        out, _ = self.rnn(x)
        out = out.contiguous().view(-1, neu_nr)
        out = self.fc(out)
        
        return out

class CNN_NN(nn.Module):
    # Sieć konwolucyjna
    def __init__(self):
        super(CNN_NN, self).__init__()
        self.conv1d1 = nn.Conv1d(in_channels=1, out_channels=neu_nr, kernel_size=1)
        self.conv1d2 = nn.Conv1d(in_channels=neu_nr, out_channels=neu_nr, kernel_size=1)
        self.conv1d3 = nn.Conv1d(in_channels=neu_nr, out_channels=neu_nr, kernel_size=1)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(neu_nr, 1)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size, seq_len, 1)
        
        x = self.conv1d1(x)
        x = self.tanh(x)
        x = self.conv1d2(x)
        x = self.tanh(x)
        x = self.conv1d3(x)
        x = self.tanh(x)
        
        x = x.view(batch_size, -1)
        x = self.fc(x)
        
        return x


# Definicja domeny 
t_domain = torch.linspace(0,1,n).view(-1,1).requires_grad_(True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Zwykła sieć 
# model = NN().to(device) # Deklaracja modelu

# Sieć rekurencyjna
model = RNN_NN().to(device)

# Sieć konwolucyjna
# model = CNN_NN().to(device)

# Inicjalizacja wag
# model.apply(init_weights)

optimizer = torch.optim.Adam(model.parameters(),lr=lr)


# Zmiana współczynnika uczenia w czasie
scheduler = StepLR(optimizer, step_size=2500, gamma=0.97)

results = []
loss_history = []

# Start obliczeń
t1 = datetime.datetime.now()

for i in range(max_iter):
    optimizer.zero_grad()
    
    # Wagi poszczególnych strat [PDE, IC1, IC2]
    loss_weights = [1, 1, 1]
    
    # Strata z równania różniczkowego
    y = model(t_domain)
    dy_dt  = torch.autograd.grad(y, t_domain, torch.ones_like(y), create_graph=True)[0]  # dy/dt
    d2y_dt2 = torch.autograd.grad(dy_dt,  t_domain, torch.ones_like(dy_dt),  create_graph=True)[0]   # d^2y/dt^2
    ODE = m/T**2 * d2y_dt2 + c/T * dy_dt + k * y - F0 * torch.sin(omega_f * t_domain * T)
    loss_ODE = loss_weights[0] * torch.mean(ODE**2)
    
    # Strata z warunku początkowego y(0) = y_0
    IC1 = y[0] - y_0
    loss_IC1 = loss_weights[1] * torch.mean(IC1**2)
    
    # Strata z warunku początkowego dy/dt (t=0) = v_0
    IC2 = dy_dt[0] / T - v_0
    loss_IC2 = loss_weights[2] * torch.mean(IC2**2)

    # Strata całkowita
    loss = loss_ODE + loss_IC1 + loss_IC2
    loss.backward()
    optimizer.step()
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
