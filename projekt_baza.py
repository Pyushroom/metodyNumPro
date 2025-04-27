import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parametry obwodu
R1 = 1.0   # Ohm
R2 = 2.0   # Ohm
L1 = 0.5   # H
L2 = 0.3   # H
M  = 0.1   # H
C  = 0.01  # F

# Obliczanie D1 i D2
D1 = (L1 / M) - (M / L2)
D2 = (M / L1) - (L2 / M)

# Wymuszenie: e(t)
def e(t):
    return np.sin(t)

# Definicja funkcji f(t, y) opisującej układ równań różniczkowych
def f(t, y):
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

    dydt = A @ y + B * e(t)
    return dydt

# Warunki początkowe: i1(0)=0, i2(0)=0, uC(0)=0
y0 = [0, 0, 0]

# Zakres czasu
t_span = (0, 30)  
t_eval = np.linspace(*t_span, 1000)

# Rozwiązywanie układu równań
sol = solve_ivp(f, t_span, y0, t_eval=t_eval)

# Wykres 1: Napięcie na kondensatorze Uc(t) vs t
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[2], label='uC(t) [V]', color='blue')
plt.xlabel('Czas [s]')
plt.ylabel('Napięcie na kondensatorze [V]')
plt.title('Napięcie na kondensatorze Uc(t) vs t')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('napięcie_na_kondensatorze.png', dpi=300)

# Wykres 2: Prąd i1(t) i i2(t) vs t
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='i1(t) [A]', color='red')
plt.plot(sol.t, sol.y[1], label='i2(t) [A]', color='green')
plt.xlabel('Czas [s]')
plt.ylabel('Prąd [A]')
plt.title('Prąd i1(t) i i2(t) vs t')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('prądy_i1_i_i2.png', dpi=300)


plt.show()
