import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from tabulate import tabulate
import os

# Parametry obwodu
R1 = 0.1   # Ohm
R2 = 10.0  # Ohm
L1 = 3.0   # H
L2 = 5.0   # H
M  = 0.8   # H
C  = 0.5   # F

# Dane pomiarowe
UL1_vals = np.array([20, 50, 100, 150, 200, 250, 280, 300])
Mj_vals = np.array([0.46, 0.64, 0.78, 0.68, 0.44, 0.23, 0.18, 0.18])

# Funkcja wymuszenia: e(t) = 100 * sin(2 * pi * f * t)
def e_signal(t, f):
    return 100 * np.sin(2 * np.pi * f * t)

D1 = (L1 / M) - (M / L2)
D2 = (M / L1) - (L2 / M)

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

# Interpolacja a) metodą Lagrange’a
def lagrange_interpolation(x_vals, y_vals, x):
    total = 0.0
    n = len(x_vals)
    for i in range(n):
        term = y_vals[i]
        for j in range(n):
            if j != i:
                term *= (x - x_vals[j]) / (x_vals[i] - x_vals[j])
        total += term
    return total

# b) interpolację funkcjami sklejanymi trzeciego stopnia
def third_degree_spline_interpolation(x_vals, y_vals, x):
    n = len(x_vals)
    h = np.diff(x_vals)
    
    A = np.zeros((n, n))
    b = np.zeros(n)
    A[0, 0] = 1
    A[-1, -1] = 1
    b[0] = b[-1] = 0

    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 6 * ((y_vals[i + 1] - y_vals[i]) / h[i] - (y_vals[i] - y_vals[i - 1]) / h[i - 1])
    
    c = np.linalg.solve(A, b)

    if x < x_vals[0]:
        return y_vals[0]  # lewa ekstrapolacja
    if x > x_vals[-1]:
        return y_vals[-1]  # prawa ekstrapolacja

    for i in range(n - 1):
        if x_vals[i] <= x <= x_vals[i + 1]:
            h_i = x_vals[i + 1] - x_vals[i]
            a = (x_vals[i + 1] - x) / h_i
            b = (x - x_vals[i]) / h_i
            y = (
                a * y_vals[i] + b * y_vals[i + 1] +
                ((a**3 - a) * c[i] + (b**3 - b) * c[i + 1]) * (h_i ** 2) / 6
            )
            return y

# Aproksymacja c) – wielomianowa metoda najmniejszych kwadratów
def least_squares_fit(x_vals, y_vals, degree):
    n = len(x_vals)
    A = np.zeros((degree + 1, degree + 1))
    b = np.zeros(degree + 1)
    for i in range(degree + 1):
        for j in range(degree + 1):
            A[i, j] = np.sum(x_vals ** (i + j))
        b[i] = np.sum(y_vals * (x_vals ** i))
    coeffs = np.linalg.solve(A, b)
    return coeffs

def evaluate_polynomial(coeffs, x):
    return sum(c * x ** i for i, c in enumerate(coeffs))

# Obliczenie całkowitej mocy
def calculate_power(t, y):
    i1, i2 = y[:, 0], y[:, 1]
    power = R1 * i1**2 + R2 * i2**2
    return power

def metoda_simpsona(fx, dt):
    if len(fx) % 2 == 0:
        fx = fx[:-1]  
    n = len(fx)
    h = dt
    return h/3 * (fx[0] + 4 * np.sum(fx[1:n-1:2]) + 2 * np.sum(fx[2:n-2:2]) + fx[n-1])

# Funkcja P(f)
def P(f, dt=0.01):
    t_span = (0, 30)
    y0 = [0, 0, 0]
    t, y = improved_euler(f_system, y0, t_span, dt, f)
    power = calculate_power(t, y)
    return metoda_simpsona(power, t[1] - t[0])

# Funkcja celu F(f) = P(f) - 406
def F(f):
    return P(f) - 406

# Metoda bisekcji
def bisection(f, a, b, tol=1e-3, max_iter=100):
    evals = 0
    fa = f(a)
    for i in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        evals += 1
        if abs(fc) < tol or (b - a) / 2 < tol:
            return c, fc, i+1, evals
        if np.sign(fc) == np.sign(fa):
            a = c
            fa = fc
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
        # Obliczamy wartość F(x0 + d) i F(x0 + d/2)
        F_x0_d = F(x0 + d)
        F_x0_d_half = F(x0 + d / 2)
        
        der1 = np.mean(F_x0_d) if isinstance(F_x0_d, np.ndarray) else F_x0_d
        der2 = np.mean(F_x0_d_half) if isinstance(F_x0_d_half, np.ndarray) else F_x0_d_half
        
        # Obliczamy różnice i porównujemy
        if abs(der1) > 1e-12 and abs(der1 - der2) / abs(der1) < 0.01:
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

# delta f
delta_f = find_best_delta()
print(f"Optymalnie dobrane delta f dla quasi-Newtona: {delta_f}")

# metody
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
