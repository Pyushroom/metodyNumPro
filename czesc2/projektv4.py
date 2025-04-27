import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d

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

# Oblicz współczynniki aproksymacji dla stopnia 3 i 5
coeffs_deg3 = least_squares_fit(UL1_vals, Mj_vals, degree=3)
coeffs_deg5 = least_squares_fit(UL1_vals, Mj_vals, degree=5)

# Definicje funkcji M(uL1)
M_poly_interp = lambda x: lagrange_interpolation(UL1_vals, Mj_vals, x)
M_spline = lambda x: third_degree_spline_interpolation(UL1_vals, Mj_vals, x)
M_poly_deg3 = lambda x: evaluate_polynomial(coeffs_deg3, x)
M_poly_deg5 = lambda x: evaluate_polynomial(coeffs_deg5, x)


# Wybór używanej funkcji do interpolacji M(uL1)
M_interp = M_spline  # testowanie metod
interp_name = "interpolacja sklejenie" 


# Wymuszenia
def e1(t): return 120 if t < 3 else 0
def e2(t): return 240 * np.sin(t)
def e3(t, f=5): return 210 * np.sin(2 * np.pi * f * t)
def e4(t, f=50): return 120 * np.sin(2 * np.pi * f * t)

def f_nonlinear(t, y, e_func, prev_y1, dt):
    y1, y2, y3 = y

    # Przybliżenie pochodnej di1/dt
    di1_dt = (y1 - prev_y1) / dt
    uL1 = L1 * di1_dt

    # Obliczanie nieliniowej indukcyjności wzajemnej
    M = float(M_interp(np.abs(uL1)))  # bezwzględna wartość napięcia

    # Obliczanie D1 i D2 dla bieżącej wartości M
    D1 = (L1 / M) - (M / L2)
    D2 = (M / L1) - (L2 / M)

    # Aktualna macierz A i wektor B
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
def euler_method_nonlinear(f, y0, t_span, e_func, dt):
    t = np.arange(t_span[0], t_span[1], dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(1, len(t)):
        prev_y1 = y[i-1][0]
        y[i] = y[i-1] + dt * f(t[i-1], y[i-1], e_func, prev_y1, dt)
        
    return t, y


# Implementacja ulepszonej metody Eulera
def improved_euler_method_nonlinear(f, y0, t_span, e_func, dt):
    t = np.arange(t_span[0], t_span[1], dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(1, len(t)):
        prev_y1 = y[i-1][0]
        k1 = f(t[i-1], y[i-1], e_func, prev_y1, dt)
        k2 = f(t[i-1] + dt, y[i-1] + dt * k1, e_func, y[i-1][0] + dt * k1[0], dt)
        y[i] = y[i-1] + (dt / 2) * (k1 + k2)

    return t, y

# Parametry symulacji
y0 = [0, 0, 0]  # Warunki początkowe: i1(0)=0, i2(0)=0, uC(0)=0
t_span = (0, 30)  # Zakres czasu
dt = 0.01  # Krok czasowy

# Wybór metody i wymuszenia
methods = ['Euler', 'Improved Euler']
#e_funcs = [e1, e2, e3, e4]
e_funcs = [e2, e4] #dwie funkce e(t) do wykresów

for e_func in e_funcs:
    for method in methods:
        if method == 'Euler':
            t, y = euler_method_nonlinear(f_nonlinear, y0, t_span, e_func, dt)
        elif method == 'Improved Euler':
            t, y = improved_euler_method_nonlinear(f_nonlinear, y0, t_span, e_func, dt)

        # Obliczanie napięcia na rezystorze R2
        uR2 = R2 * y[:, 1]  

        # Wykres 1: Napięcie na kondensatorze Uc(t) vs t
        plt.figure(figsize=(10, 6))
        plt.plot(t, y[:, 2], label='uC(t) [V]', color='blue')
        plt.xlabel('Czas [s]')
        plt.ylabel('Napięcie na kondensatorze [V]')
        plt.title(f'Napięcie na kondensatorze Uc(t) vs t ({method}, e(t)={e_func.__name__})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'napiecie_na_kondensatorze_{interp_name}_{method}_{e_func.__name__}.png', dpi=300)

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
        plt.savefig(f'prady_i1_i_i2_{interp_name}_{method}_{e_func.__name__}.png', dpi=300)

        # Wykres 3: Napięcie na rezystorze R2
        plt.figure(figsize=(10, 6))
        plt.plot(t, uR2, label='uR2(t) [V]', color='purple')
        plt.xlabel('Czas [s]')
        plt.ylabel('Napięcie na R2 [V]')
        plt.title(f'Napięcie na rezystorze R2(t) vs t ({method}, e(t)={e_func.__name__})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'napiecie_na_R2_{interp_name}_{method}_{e_func.__name__}.png', dpi=300)

        plt.show()

