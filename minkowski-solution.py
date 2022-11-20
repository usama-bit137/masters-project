import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

plt.rcParams["font.family"] = "Times New Roman"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 2.50

ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))


def U_der(x, E):
    return 0.5 * (x ** 3 - x)

def ODE(x, v, t, E):
    """
    x = initial position.
    v = initial velocity.
    t = start time. """
    return U_der(x, E)

def RK_22(x0, t0, v0, t_range, x_range):
    """
    x0 = initial position
    t0 = initial time
    v0 = initial velocity
    t_range = the domain of the function
    x_range = maximum range cutoff

    This lovely function allows us to just put in
    the correct initial conditions and calculate the
    differential equation for a particular one. The
    purpose of the if statement is to basically cutoff
    the solution if it overshoots.
    """

    # Boxes to fill:
    xh_total = []
    t_total = []
    v_total = []

    while t0 < t_range:

        xh = x0 + dt * v0 / 2

        if abs(x0) > x_range:
            break

        vh = v0 + ODE(x0, t0, t0, E) * dt / 2
        x0 += dt * vh
        v0 += dt * ODE(xh, vh, t0 + dt / 2, E)
        t0 += dt

        # Fill the boxes:
        v_total.append(vh)
        xh_total.append(xh)
        t_total.append(t0)

    # Fewer complications:
    return np.array([t_total, xh_total, v_total])


# Fundamental values we require:
E = 1
N = 100

# Initial conditions for the Runge-Kutta algorithm.
t0 = 0
v0 = 0
dt = 0.01
x_range = 10
t_range = 100

"""Want a function for all this business: 
__________________________________________________________________"""
a_i = -0.999
a_ai = 0.999
Phi_I = RK_22(a_i, t0, v0, t_range, x_range)
Phi_AI = RK_22(a_ai, t0, v0, t_range, x_range)

"""__________________________________________________________________"""
# Saves us some computational strength:
s = 20
ax1.tick_params(axis='both', which='major', labelsize=s)
ax2.tick_params(axis='both', which='major', labelsize=s)

ax1.plot(Phi_I[0, :], Phi_I[1, :], linewidth=3)
ax1.set_ylabel('$x$', fontsize=s)
ax1.set_xlabel('$ τ $', fontsize=s)
ax1.set_title('Instanton', fontsize=s)

ax2.set_ylabel('$x$', fontsize=s)
ax2.set_xlabel('$ τ $', fontsize=s)
ax2.set_title('Anti-instanton', fontsize=s)

ax1.set_xlim(0, 16)

ax1.set_ylim(-1.2, 1.2)
ax2.set_ylim(-1.2, 1.2)
ax2.plot(Phi_AI[0, :], Phi_AI[1, :], linewidth=3)
ax2.set_xlim(0, 16)

plt.show()

