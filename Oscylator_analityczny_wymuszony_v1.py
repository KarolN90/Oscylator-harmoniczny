# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 18:48:54 2024

@author: karol
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Stałe
m = 1
c = 0.5
k = 10

# Wymuszenie
F0 = 2       # Amplituda siły wymuszającej
omega_f = 2  # Częstość kołowa siły wymuszającej

# Rozważany przedział czasu 0 - T
T = 30
# podział przedziału czasowego
n = 300   

# Warunki początkowe
y_0 = 1    # y(t=0)
v_0 = 5    # dy/dt (t=0)

# Rozwiązanie dokładne
def analityczny_oscylator_wymuszony(t, m, c, k, F0, omega_f, T, n, y_0, v_0):
    # Definicja symboli
    t = sp.symbols('t')
    x = sp.Function('x')(t)
    omega_0, gamma, K, omega = sp.symbols('omega_0 gamma K omega', real=True)

    # Równanie ruchu oscylatora harmonicznego z wymuszeniem
    diff_eq = x.diff(t, 2) + 2*gamma*x.diff(t) + omega_0**2*x - K*sp.sin(omega*t)

    # Rozwiązanie symboliczne
    solution = sp.dsolve(diff_eq)

    # Dane wejciowe
    values = {omega_0: (np.sqrt(k/m)), gamma: (c/(2*m)), K: F0/m, omega: omega_f}
    specific_solution = solution.subs(values).rhs

    # Warunki brzegowe
    initial_conditions = {'x(0)': y_0, 'dx_dt(0)': v_0}
    constants = sp.solve((specific_solution.subs(t, 0) - initial_conditions['x(0)'], 
                          specific_solution.diff(t).subs(t, 0) - initial_conditions['dx_dt(0)']))
    
    result = specific_solution.subs(constants)

    # Rozwiązanie numeryczne
    y = sp.lambdify(t, result, 'numpy')
    times = np.linspace(0, T, n)
    y_an = y(times)
    return y_an

'''

# Próbki czasu
t = np.linspace(0, T, n)

# Obliczanie wartości funkcji dla próbek czasu
y_an = analityczny_oscylator_wymuszony(t, m=m, c=c, k=k, F0=F0, omega_f=omega_f, T=T, n=n, y_0=y_0, v_0=v_0)

# Narysowanie wykresu
plt.plot(t, y_an, label='y_an(t)')
plt.xlabel('Czas')
plt.ylabel('Wartość')
plt.title('Wykres analitycznego oscylatora z siłą wymuszającą')
plt.legend()
plt.grid(True)
plt.show()

'''