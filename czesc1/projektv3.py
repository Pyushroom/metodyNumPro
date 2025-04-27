import numpy as np
import matplotlib.pyplot as plt

# Parametry obwodu
R1 = 0.1   # Ohm
R2 = 10.0  # Ohm
L1 = 3.0   # H
L2 = 5.0   # H
M  = 0.8   # H
C  = 0.5   # F

# Obliczanie D1 i D2
D1 = (L1 / M) - (M / L2)
D2 = (M / L1) - (L2 / M)

# Definicje wymuszeń
def e1(t):
    if t < 3:
        return 120
    else:
        return 0

def e2(t):
    return 240 * np.sin(t)

def e3(t, f=5):
    return 210 * np.sin(2 * np.pi * f * t)

def e4(t, f=50):
    return 120 * np.sin(2 * np.pi * f * t)

# Definicja funkcji f(t, y) dla układu równań różniczkowych
def f(t, y, e_func):
    y1, y2, y3 = y

    A = np.array([
        [-R1 / (M * D1),  R2 / (L2 * D1),  -1 / (M * D1)],
        [-R1 / (L1 * D2), R2 / (M * D2),   -1 / (L1 * D2)],
        [1 / C,           0,              0]
    ])

    B = np.array([
        1 / (M * D1),
        1 / (L1 * D2),
        0
    ])

    dydt = A @ y + B * e_func(t)
    return dydt

# Implementacja metody Eulera
def euler_method(f, y0, t_span, e_func, dt):
    t = np.arange(t_span[0], t_span[1], dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(1, len(t)):
        y[i] = y[i-1] + dt * f(t[i-1], y[i-1], e_func)
        
    return t, y

# Implementacja ulepszonej metody Eulera
def improved_euler_method(f, y0, t_span, e_func, dt):
    t = np.arange(t_span[0], t_span[1], dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(1, len(t)):
        k1 = f(t[i-1], y[i-1], e_func)
        k2 = f(t[i-1] + dt, y[i-1] + dt * k1, e_func)
        y[i] = y[i-1] + (dt / 2) * (k1 + k2)

    return t, y

# Parametry symulacji
y0 = [0, 0, 0]  # Warunki początkowe
t_span = (0, 30)  # Zakres czasu
dt = 0.01  # Krok czasowy

# Wybór metody i wymuszenia
methods = ['Euler', 'Improved Euler']
e_funcs = [e1, e2, e3, e4]

for e_func in e_funcs:
    for method in methods:
        if method == 'Euler':
            t, y = euler_method(f, y0, t_span, e_func, dt)
        elif method == 'Improved Euler':
            t, y = improved_euler_method(f, y0, t_span, e_func, dt)

        # Wykres 1: Napięcie na kondensatorze Uc(t) vs t
        plt.figure(figsize=(10, 6))
        plt.plot(t, y[:, 2], label='uC(t) [V]', color='blue')
        plt.xlabel('Czas [s]')
        plt.ylabel('Napięcie na kondensatorze [V]')
        plt.title(f'Napięcie na kondensatorze Uc(t) vs t ({method}, e(t)={e_func.__name__})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'napiecie_na_kondensatorze_{method}_{e_func.__name__}.png', dpi=300)

        # Wykres 2: Prąd i1(t) i i2(t) vs t
        plt.figure(figsize=(10, 6))
        plt.plot(t, y[:, 0], label='i1(t) [A]', color='red')
        plt.plot(t, y[:, 1], label='i2(t) [A]', color='green')
        plt.xlabel('Czas [s]')
        plt.ylabel('Prąd [A]')
        plt.title(f'Prąd i1(t) i i2(t) vs t ({method}, e(t)={e_func.__name__})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'prady_i1_i_i2_{method}_{e_func.__name__}.png', dpi=300)

        # Pokaż wykresy
        plt.show()
