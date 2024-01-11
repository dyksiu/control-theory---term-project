#Biblioteki potrzebne do realizacji projektu
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint
#############################################
#Parametry modelu i deklaracja wektoru czasu
a = 8
a1 = 8
a2 = 8
h = 6
g = 9.81
B1, B2 = 1, 1
M1, M2 = 10, 1
r1, r2 = 1, 1
m = 1
t_span = [0, 100]
t_eval = np.arange(t_span[0], t_span[1], 0.1)
#############################################
#Deklaracja list potrzebnych do zbierania wartości uchybu w osiach x i y i odpowiadającego im czasu
error_x_values = []
error_y_values = []
time_values = []
#############################################
#Zdefiniowanie regulatora PID jako klasy
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp            #Współczynnik proporcjonalny
        self.Ki = Ki            #Współczynnik całkujący
        self.Kd = Kd            #Współczynnik różniczkujący
        self.integral = 0       #Skumulowana suma błędów (część całkująca)
        self.prev_error = None  #Poprzedni błąd (do obliczeń różniczkowych)

    def update(self, error, dt):
        #Obliczanie składnika całkującego
        self.integral += error * dt

        #Obliczanie składnika różniczkującego
        derivative = 0
        if self.prev_error is not None:
            derivative = (error - self.prev_error) / dt
        self.prev_error = error

        #Obliczanie wyjścia regulatora
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return output

pid_x = PIDController(Kp=8, Ki=0.1, Kd=0.5)
pid_y = PIDController(Kp=8, Ki=0.1, Kd=0.5)
#############################################
#Funkcja modelu definiująca równania różniczkowe
def model(t, x):
    x, y, x_dot, y_dot , phi_1, phi_2 = x
    #Cel regulacji dla osi x i y
    x_target, y_target = -5, -8

    #Obliczanie błędu(różnica między docelową a rzeczywistą pozycją)
    error_x = x_target - x
    error_y = y_target - y
    error_x_values.append(error_x)
    error_y_values.append(error_y)
    time_values.append(t)
    #print(error_x)
    #print(error_y)

    #Aktualizacja sygnału sterującego z regulatora PID
    control_signal_x = pid_x.update(error_x, 0.01)  # Zakładamy krok czasowy równy 1
    control_signal_y = pid_y.update(error_y, 0.01)

    #Obliczanie prędkości kątowych
    phi_1_dot = ((y_dot  - x_dot)**2  / r1 ** 2) * 360 / (2*np.pi)
    phi_2_dot = ((y_dot  + x_dot)**2  / r2 ** 2) * 360 / (2*np.pi)

    #Zdefiniowanie równań różniczkowych opisujących model manipulatora
    x_ddot = control_signal_x + x_dot * (-B1 / (m * r1 ** 2) - B2 / (m * r2 ** 2)) + y_dot * (B1 / (m * r1 ** 2) - B2 / (m * r2 ** 2)) \
             - M1 / (m * r1) * (x + a) / (np.sqrt((x + a) ** 2 + y ** 2)) - M2 / (m * r2) * (x - a) / (np.sqrt((x + a) ** 2 + y ** 2))
    y_ddot = control_signal_y + x_dot * (B1 / (m * r1 ** 2) - B2 / (m * r2 ** 2)) + y_dot * (-B1 / (m * r1 ** 2) - B2 / (m * r2 ** 2)) \
             - M1 / (m * r1) * y / (np.sqrt((x + a) ** 2 + y ** 2)) - M2 / (m * r2) * y / (np.sqrt((x - a) ** 2 + y ** 2)) - g

    return [x_dot, y_dot, x_ddot, y_ddot,phi_1_dot,phi_2_dot]
#############################################
#Warunki początkowe i rozwiązanie równań
initial_conditions = [0,-h,0,0,0,0]
sol = solve_ivp(model, t_span, initial_conditions, t_eval=t_eval, method='RK45')
#############################################
#Zdefiniowanie funkcji narzucającej ograniczenia
def apply_limits(x, y):
    x = np.clip(x, -10, 10)   #Ogranicza x do zakresu -5 do 5
    y = np.clip(y, -20, -3)   #Ogranicza y do zakresu -20 do -3
    return x, y
#############################################
#Zastosowanie ograniczeń do rozwiązania
x_limited, y_limited = apply_limits(sol.y[0], sol.y[1])
#############################################
#Obliczenie prędkości i przyspieszenia z uwzględnieniem ograniczeń
x_dot_limited, y_dot_limited = np.gradient(x_limited, sol.t), np.gradient(y_limited, sol.t)
x_ddot_limited, y_ddot_limited = np.gradient(x_dot_limited, sol.t), np.gradient(y_dot_limited, sol.t)
#############################################
#Wykresy
fig, axs = plt.subplots(3, 2, figsize=(12, 18))
#Ustawienie tytułów i etykiet
titles = ['Pozycja x', 'Pozycja y', 'Prędkość x', 'Prędkość y', 'Przyspieszenie x', 'Przyspieszenie y']
y_labels = ['Pozycja x', 'Pozycja y', 'Prędkość x', 'Prędkość y', 'Przyspieszenie x', 'Przyspieszenie y']

for i, ax in enumerate(axs.flat):
    ax.set_title(titles[i])
    ax.set_ylabel(y_labels[i])
    ax.grid(True)
#Rysowanie wykresów
axs[0, 0].plot(sol.t, x_limited, label="x(t)", color='blue')
axs[0, 1].plot(sol.t, y_limited, label="y(t)", color='red')
axs[1, 0].plot(sol.t, x_dot_limited, label="x'(t)", color='green')
axs[1, 1].plot(sol.t, y_dot_limited, label="y'(t)", color='orange')
axs[2, 0].plot(sol.t, x_ddot_limited, label="x''(t)", color='purple')
axs[2, 1].plot(sol.t, y_ddot_limited, label="y''(t)", color='brown')
#Wykresy kątów bębnów
fig, angle_axs = plt.subplots(2, figsize=(10, 6))
angle_axs[0].plot(sol.t, sol.y[4], label='Kąt obrotu', color='blue')
angle_axs[0].set_title('Kąt \u03C61')
angle_axs[0].set_xlabel('Czas')
angle_axs[0].set_ylabel('Kąt')
angle_axs[0].grid(True)
angle_axs[0].legend()
angle_axs[1].plot(sol.t, sol.y[5], label='Kąt obrotu', color='red')
angle_axs[1].set_title('Kąt \u03C62')
angle_axs[1].set_xlabel('Czas')
angle_axs[1].set_ylabel('Kąt')
angle_axs[1].grid(True)
angle_axs[1].legend()
plt.tight_layout()
plt.show()
#Wykresy uchybów w osiach x i y
fig, error_axs = plt.subplots(2, figsize=(10, 6))
error_axs[0].plot(time_values,error_x_values, label="Uchyb w osi x",color ='blue')
error_axs[0].set_title('Uchyb w osi x')
error_axs[0].set_xlabel('Czas')
error_axs[0].set_ylabel('Wartość uchybu')
error_axs[0].grid(True)
error_axs[0].legend()
error_axs[1].plot(time_values, error_y_values, label='Uchyb w osi y', color='red')
error_axs[1].set_title('Uchyb w osi y')
error_axs[1].set_xlabel('Czas')
error_axs[1].set_ylabel('Wartość uchybu')
error_axs[1].grid(True)
error_axs[1].legend()
plt.tight_layout()
plt.show()
#Dodawanie legend
for ax in axs.flat:
    ax.legend()
#############################################
#Tworzenie animacji
fig, ax = plt.subplots()
line, = ax.plot([], [], 'ro-', lw=2)           #Ruchoma czerwona kropka
trace_line, = ax.plot([], [], 'orange', lw=2)  #Pomarańczowa linia śledząca
time_template = 'Czas = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
ax.set_xlim(-30, 20)
ax.set_ylim(-10, 10)
ax.grid()
#############################################
#Dodawanie dwóch niebieskich kulek
ax.plot(-8, -3, 'bo', markersize=10)  #Niebieska kulka w punkcie (-8, 0)
ax.plot(8, -3, 'bo', markersize=10)   #Niebieska kulka w punkcie (8, 0)
#############################################
#Deklaracja funkcji potrzebnej do wykreślenia linii łączącej kulkę z bębnami
def init():
    line.set_data([], [])
    trace_line.set_data([], [])
    time_text.set_text('')
    return line, trace_line, time_text

def animate(i):
    x = sol.y[0][i]
    y = sol.y[1][i]

    #Ograniczenie ruchu w osi y
    if y > -3:
        y = -3
    elif y < -20:
        y = -20

    line.set_data(x, y)
    trace_line.set_data([-8, x, 8], [-3, y, -3])  #Aktualizacja linii
    time_text.set_text(time_template % (i*0.01))
    return line, trace_line, time_text

ani = animation.FuncAnimation(fig, animate, len(sol.t), interval=25, blit=True, init_func=init)
ax.set_xlim(-20, 20)
ax.set_ylim(-25, 0)
plt.tight_layout()
plt.show()
