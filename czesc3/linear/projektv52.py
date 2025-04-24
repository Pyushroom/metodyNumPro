import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import os
from tabulate import tabulate  # Importowanie tabulate

# Parametry obwodu
R1, R2 = 0.1, 10.0
L1, L2, M, C = 3.0, 5.0, 0.8, 0.5

D1 = (L1 / M) - (M / L2)
D2 = (M / L1) - (L2 / M)

# Definicje wymuszeń
def e_const(t): return 1
def e1(t): return 120 if t < 3 else 0
def e2(t): return 240 * np.sin(t)
def e3(t, f=5): return 210 * np.sin(2 * np.pi * f * t)
def e4(t, f=50): return 120 * np.sin(2 * np.pi * f * t)

# Lista funkcji
e_funcs = [("e_const", e_const), ("e1", e1), ("e2", e2), ("e3", lambda t: e3(t, 5)), ("e4", lambda t: e4(t, 50))]

# Funkcja opisująca układ równań
def f(t, y, e_func):
    y1, y2, y3 = y
    A = np.array([
        [-R1 / (M * D1),  R2 / (L2 * D1),  -1 / (M * D1)],
        [-R1 / (L1 * D2), R2 / (M * D2),   -1 / (L1 * D2)],
        [1 / C,           0,              0]
    ])
    B = np.array([1 / (M * D1), 1 / (L1 * D2), 0])
    return A @ y + B * e_func(t)

# Euler i poprawiona metoda Eulera
def euler_method(f, y0, t_span, e_func, dt):
    t = np.arange(t_span[0], t_span[1] + dt, dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i-1] + dt * f(t[i-1], y[i-1], e_func)
    return t, y

def improved_euler_method(f, y0, t_span, e_func, dt):
    t = np.arange(t_span[0], t_span[1] + dt, dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        k1 = f(t[i-1], y[i-1], e_func)
        k2 = f(t[i-1] + dt, y[i-1] + dt * k1, e_func)
        y[i] = y[i-1] + (dt / 2) * (k1 + k2)
    return t, y

# Funkcja do obliczenia mocy i całki
def calculate_power(t, y):
    i1, i2 = y[:, 0], y[:, 1]
    power = R1 * i1**2 + R2 * i2**2  # Moc w watach
    return power

# Parametry czasowe
t_span = (0, 30)
dt_short = 0.001
dt_long = 0.1
y0 = [0, 0, 0]

# Wyniki
results = []

# Tworzymy katalog na wykresy
output_dir = "wykresy"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Tworzymy wykresy i zapisujemy wyniki do listy
for e_name, e_func in e_funcs:
    for dt in [dt_short, dt_long]:
        t, y = improved_euler_method(f, y0, t_span, e_func, dt)
        power = calculate_power(t, y)
        
        # Moc (średnia moc w czasie)
        P_rect = np.sum(power) * (t[1] - t[0])  # Prostokąty
        P_simpson = simpson(power, t)  # Simpson

        # Dodajemy wyniki do listy
        results.append([e_name, dt, round(P_rect, 6), round(P_simpson, 6)])

        # Rysowanie wykresów
        plt.figure(figsize=(10, 6))

        # Wykresy prądów
        plt.subplot(2, 1, 1)
        plt.plot(t, y[:, 0], label='i1(t)', color='b')
        plt.plot(t, y[:, 1], label='i2(t)', color='r')
        plt.title(f'Prądy w obwodzie: {e_name} - dt={dt}')
        plt.xlabel('Czas [s]')
        plt.ylabel('Prąd [A]')
        plt.legend()

        # Wykres mocy
        plt.subplot(2, 1, 2)
        plt.plot(t, power, label='Moc [W]', color='g')
        plt.title(f'Moc w czasie: {e_name} - dt={dt}')
        plt.xlabel('Czas [s]')
        plt.ylabel('Moc [W]')
        plt.legend()

        plt.tight_layout()

        # Zapisz wykres do pliku PNG
        filename = f"{output_dir}/{e_name}_dt={dt}.png"
        plt.savefig(filename)
        plt.close()  # Zamknięcie wykresu po zapisaniu

# Wyświetlanie tabeli z wynikami
headers = ["Wymuszenie", "dt", "Prostokąty [W]", "Simpson [W]"]
print(tabulate(results, headers=headers, tablefmt="grid"))
