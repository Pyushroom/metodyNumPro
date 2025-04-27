import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from tabulate import tabulate

# Parametry obwodu
R1, R2 = 0.1, 10.0
L1, L2, M, C = 3.0, 5.0, 0.8, 0.5

D1 = (L1 / M) - (M / L2)
D2 = (M / L1) - (L2 / M)

# Funkcja wymuszenia: e(t) = 100 * sin(2 * pi * f * t)
def e_signal(t, f):
    return 100 * np.sin(2 * np.pi * f * t)

# Prekomputacja elementów macierzy A
A1_0 = -R1 / (M * D1)
A1_1 = R2 / (L2 * D1)
A1_2 = -1 / (M * D1)

A2_0 = -R1 / (L1 * D2)
A2_1 = R2 / (M * D2)
A2_2 = -1 / (L1 * D2)

A3_0 = 1 / C

# Układ równań różniczkowych (optymalizacja przez prekomputację A)
def f_system(t, y, f):
    y1, y2, y3 = y
    A = np.array([
        [A1_0, A1_1, A1_2],
        [A2_0, A2_1, A2_2],
        [A3_0, 0, 0]
    ])
    B = np.array([1 / (M * D1), 1 / (L1 * D2), 0])
    return A @ y + B * e_signal(t, f)

# Poprawiona metoda Eulera
def improved_euler(f, y0, t_span, dt, freq):
    t = np.arange(t_span[0], t_span[1] + dt, dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        k1 = f(t[i-1], y[i-1], freq)
        k2 = f(t[i-1] + dt, y[i-1] + dt * k1, freq)
        y[i] = y[i-1] + (dt / 2) * (k1 + k2)
    return t, y

# Obliczenie całkowitej mocy
def calculate_power(t, y):
    i1, i2 = y[:, 0], y[:, 1]
    power = R1 * i1**2 + R2 * i2**2
    return power

# Funkcja P(f)
def P(f, dt=0.01): #wartosc kroku
    t_span = (0, 30)
    y0 = [0, 0, 0]
    t, y = improved_euler(f_system, y0, t_span, dt, f)
    power = calculate_power(t, y)
    return simpson(power, t)

# Funkcja celu F(f) = P(f) - 406
def F(f):
    return P(f) - 406

# Metoda bisekcji
def bisection(f, a, b, tol=1e-3, max_iter=100):
    evals = 0
    for i in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        evals += 1
        if abs(fc) < tol or (b - a) / 2 < tol:
            return c, fc, i+1, evals
        if np.sign(fc) == np.sign(f(a)):
            a = c
        else:
            b = c
    return c, fc, max_iter, evals

# Metoda siecznych
def secant(f, x0, x1, tol=1e-3, max_iter=100):
    evals = 0
    for i in range(max_iter):
        f0 = f(x0)
        f1 = f(x1)
        evals += 2
        if abs(f1) < tol:
            return x1, f1, i+1, evals
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x0, x1 = x1, x2
    return x1, f1, max_iter, evals

# Metoda quasi-Newtona
def quasi_newton(f, x0, delta, tol=1e-3, max_iter=100):
    evals = 0
    for i in range(max_iter):
        Fx = f(x0)
        Fx_delta = f(x0 + delta)
        evals += 2
        derivative = (Fx_delta - Fx) / delta
        if derivative == 0:
            break
        x1 = x0 - Fx / derivative
        if abs(Fx) < tol:
            return x0, Fx, i+1, evals
        x0 = x1
    return x0, Fx, max_iter, evals

# Dobór delta f
def find_best_delta(x0=0.7):
    deltas = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    for d in deltas:
        der1 = (F(x0 + d) - F(x0)) / d
        der2 = (F(x0 + d/2) - F(x0)) / (d/2)
        if abs(der1 - der2) / abs(der1) < 0.01:  
            return d
    return deltas[-1]

# Wykres F(f)
def plot_F(f_start, f_end):
    f_vals = np.linspace(f_start, f_end, 300)
    F_vals = [F(f) for f in f_vals]
    plt.figure(figsize=(10,6))
    plt.plot(f_vals, F_vals, label="F(f)")
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Wykres funkcji celu F(f)')
    plt.xlabel('f [Hz]')
    plt.ylabel('F(f)')
    plt.grid(True)
    plt.legend()
    plt.show()

# Przedział dla f
f_start, f_end = 0.6, 0.75

# Wykres
plot_F(f_start, f_end)

# Obliczanie delty
delta_f = find_best_delta()
print(f"Optymalnie dobrane delta f dla quasi-Newtona: {delta_f}")

# Metody
bisection_result = bisection(F, f_start, f_end)
secant_result = secant(F, f_start, f_end)
quasi_newton_result = quasi_newton(F, 0.7, delta=delta_f)

# Tabela wyników
headers = ["Metoda", "Częstotliwość f [Hz]", "F(f)", "Liczba iteracji", "Liczba obliczeń P(f)"]
results = [
    ["Bisekcji", *bisection_result],
    ["Siecznych", *secant_result],
    ["Quasi-Newtona", *quasi_newton_result]
]

print(tabulate(results, headers=headers, tablefmt="grid"))
