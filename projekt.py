import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parametry obwodu
R1 = 1.0     # [Ohm]
R2 = 1.0     # [Ohm]
L1 = 1.0     # [H]
L2 = 1.0     # [H]
M  = 0.5     # [H]
C  = 1.0     # [F]

# Wymuszenie: e(t)
def e(t):
    return np.sin(t)

# Definicje pośrednie
D1 = L1 / M - M / L2
D2 = M / L1 - L2 / M

# Układ równań różniczkowych
def model(t, y):
    y1, y2, y3 = y
    dy1dt = (1 / M) * ( (L1 / L2 - M**2 / L2**2) * (-R1 * y1 - R2 * y2 + (1 / C) * y3 - e(t)) )
    dy2dt = (1 / M) * ( (M / L1 - L2 / M) * (-R1 * y1 - R2 * y2 + (1 / C) * y3 - e(t)) )
    dy3dt = (1 / C) * y1
    return [dy1dt, dy2dt, dy3dt]

# Warunki początkowe: i1 = 0, i2 = 0, uC = 0
y0 = [0.0, 0.0, 0.0]

# Przedział czasu
t_span = (0, 30)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Rozwiązanie
sol = solve_ivp(model, t_span, y0, t_eval=t_eval)

# Wykresy
t = sol.t
i1 = sol.y[0]
i2 = sol.y[1]
uC = sol.y[2]

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(t, uC, label='u_C')
plt.xlabel('t [s]')
plt.ylabel('u_C [V]')
plt.title('Napięcie na kondensatorze')
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, i1, label='i1')
plt.plot(t, i2, label='i2')
plt.xlabel('t [s]')
plt.ylabel('i1, i2 [A]')
plt.title('Prądy i1 oraz i2')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
