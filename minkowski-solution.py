import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy.random import randint
from scipy import integrate

% matplotlib
qt

fig1, ax1 = plt.subplots(1)
fig2, ax4 = plt.subplots(1)
fig3, (ax5, ax6) = plt.subplots(1, 2)

l = 1

alpha = 10
N = 8


def U(x, E):
    return (1 / 8) * (x ** 2 - 1) ** 2 + E * (x - 1)


def U_corr(x, E):
    x_plus = Newton_Raphson(2, E, 100)
    return U(x, E) - U(x_plus, E)


def U_der(x, E):
    return 0.5 * x ** 3 - 0.5 * x + E


def U_der2(x):
    return 1.5 * x ** 2 - 0.5


def ODE(x, v, t, E):
    """
    x = initial position.
    v = initial velocity.
    t = start time. """
    return -3 * (v / t) + U_der(x, E)


def Newton_Raphson(x, E, n):
    """
    x = the initial value of iteration.
    E = vacuum energy difference between false and true vacuum.
    n = the number of iterations.

    This is the root finding algorithm
    which allows us to find the stationary points
    of our potential - the true vacuum. Keeping the
    number of iterations high brings us very close
    to the vacua.
    """

    for i in range(n):
        x += - U_der(x, E) / U_der2(x)
    return x


def RK_22(x0, t0, v0, t_range, dt, x_range, E):
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

        vh = v0 + ODE(x0, v0, t0, E) * dt / 2
        x0 += dt * vh
        v0 += dt * ODE(xh, vh, t0 + dt / 2, E)
        t0 += dt

        # Fill the boxes:
        v_total.append(vh)
        xh_total.append(xh)
        t_total.append(t0)
    # Fewer complications:
    return np.array([t_total, xh_total, v_total])


def IntBisec(a_u, a_o, E, N):
    for i in range(N):
        amid = 0.5 * (a_u + a_o)
        Phi_mid = RK_22(amid, t0, v0, t_range, dt, x_range, E)
        if Phi_mid[-1, -1] < 0:
            a_u = amid
        else:
            a_o = amid
    return amid


def dSdxi(x, t, v, E):
    return 2 * np.pi ** 2 * t ** 3 * (0.5 * v ** 2 + U_corr(x, E))


# Parameters:
E = np.arange(0.01, 0.09, 0.01)

# Initial conditions for the Runge-Kutta algorithm:
t0 = 1e-15
v0 = 0
dt = 0.1
x_range = 2
t_range = 50

# Initial condition:
"""________________________Forward Solution:________________________"""

m = 10000

S_scipy = []
R = []
E_c = []

for n in tqdm(range(len(E))):
    a_beg = Newton_Raphson(-2, E[n], 100)
    a_end = Newton_Raphson(2, E[n], 100)
    a_HM = Newton_Raphson(0, E[n], 100)

    a_forward = np.linspace(a_beg, -1, m)
    index_o = []

    for j in range(len(a_forward)):
        Phi_mid_for = RK_22(a_forward[j], t0, v0, t_range, dt, x_range, E[n])

        # ax1.plot(Phi_mid_for[0,:], Phi_mid_for[1,:])

        # Collecting all the boundary overshoot:
        if Phi_mid_for[-1, -1] > 0:
            index_o.append(j)

    # Box for just crossovers:
    index_o_cross = []

    # Here I am trying to find the values for a_forward for which neighbouring
    # solutions are not of the same type-- this will tell us if there's a sweetspot:

    for i in np.arange(0, len(index_o) - 1):
        if index_o[i + 1] != index_o[i] + 1:
            index_o_cross.append(index_o[i])
            index_o_cross.append(index_o[i + 1])

    """_________________________________________________________________________"""

    a_best = []
    for l in range(len(index_o_cross)):
        # For the neighbouring values the
        if l % 2 == 0:
            a_o = a_forward[index_o_cross[l]]
            a_u = a_forward[index_o_cross[l] + 1]
            a_bestf = IntBisec(a_u, a_o, E[n], 1000)
            Phi_u = RK_22(a_u, t0, v0, t_range, dt, x_range, E[n])
            Phi_o = RK_22(a_o, t0, v0, t_range, dt, x_range, E[n])
            Phi_bestf = RK_22(a_bestf, t0, v0, t_range, dt, x_range, E[n])
        else:
            a_o = a_forward[index_o_cross[l]]
            a_u = a_forward[index_o_cross[l] - 1]
            a_bestf = IntBisec(a_u, a_o, E[n], 1000)
            Phi_u = RK_22(a_u, t0, v0, t_range, dt, x_range, E[n])
            Phi_o = RK_22(a_o, t0, v0, t_range, dt, x_range, E[n])
            Phi_bestf = RK_22(a_bestf, t0, v0, t_range, dt, x_range, E[n])

        a_best.append(a_bestf)

    if len(a_best) == 0:
        continue

    Phi_bestf = RK_22(a_best[0], t0, v0, t_range, dt, x_range, E[n])

    """_________________________________________________________________________"""

    x = Phi_bestf[1, :]
    v = Phi_bestf[-1, :]
    t = Phi_bestf[0, :]

    # Removing the waste end:
    ax1.plot(Phi_bestf[0, :], Phi_bestf[1, :], linewidth=2, label='$E$ = ' + str(E[n]))

    for l in np.arange(0, len(t) - 1):
        if np.round(x[l]) == 40:
            t_red = t[:l]
            x_red = x[:l]
            v_red = v[:l]
            break

    times = []

    """Calculating the radius for this bounce"""
    for k in np.arange(0, len(x) - 1):
        if np.sign(x_red[k]) != np.sign(x_red[k + 1]):
            R_i = 0.5 * (t_red[k] + t_red[k + 1])
            times.append(k)
            break

    print(times)

    if len(times) != 1:
        continue

    dSdt1 = dSdxi(x_red, t_red, v_red, E[n])

    Phi_fv = a_end * np.ones(l)
    Phi_fv_vel = np.zeros(l)
    dSdt_fv = dSdxi(Phi_fv, t_red, Phi_fv_vel, E[n])

    B_fv = integrate.cumtrapz(t_red, dSdt_fv)
    R.append(R_i)
    E_c.append(E[n])

    """Calculating the bounce"""

    S_cdl_scipy = integrate.cumtrapz(t_red, dSdt_fv, initial=0)
    S_scipy.append(S_cdl_scipy)

    ax1.axvline(x=R_i / alpha, color='red', linestyle='--', label='Radius')

    ax1.axhline(y=a_end, color='b', linestyle='--', label='$\phi_{fv}$ = ' + str(round(a_end, 6)))
    ax1.axhline(y=a_beg, color='b', linestyle='--', label='$\phi_{tv}$ = ' + str(round(a_beg, 6)))
    ax1.axhline(y=a_HM, color='b', linestyle='--', label='$\phi_{HM}$ = ' + str(round(a_beg, 6)))

R_1 = np.array(R)
S_1 = np.array(S_scipy)
E_1 = np.array(E_c)

R_ln = np.log(R_1)
S_ln = np.log(S_1)
E_ln = np.log(E_c)

# Checking the gradients of these lines to see if they're in agreement with
# Coleman's Thin-wall approximation:

R_polyfit = np.polyfit(E_c, R_1, 1)
S_polyfit = np.polyfit(E_c, S_1, 1)

m_R = R_polyfit[0]
m_S = S_polyfit[0]

ax1.plot(Phi_bestf[0, :] / alpha, 1 / np.tan(Phi_bestf[0, :] / alpha), color='orange', linestyle='--',
         label='drag term')
ax4.plot(Phi_bestf[1, :][:-2], Phi_bestf[-1, :][:-2], 'red', label='CdL Orbit')
ax4.axvline(x=a_beg, color='b', linestyle='--', label='$\phi_{tv}$ = ' + str(round(a_beg, 6)))
ax4.axvline(x=a_end, color='b', linestyle='--', label='$\phi_{fv}$ = ' + str(round(a_end, 6)))
ax5.plot(E_1, R_1, '.')
ax6.plot(E_1, S_1, '.')

"""___________________________Finishing Touches_________________________"""
ax1.set_ylabel('$\phi/a$', fontsize=20)
ax1.set_xlabel('$\lambda^{1/2}a ξ$', fontsize=20)
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.set_ylim(a_beg - 1, 1.1)
ax1.set_xlim(0, t_range / alpha)
ax1.legend(fontsize=10)

ax4.set_xlabel('$\phi(0)$', fontsize=20)
ax4.set_ylabel('$B$', fontsize=20)
ax4.set_xlim(a_beg - 0.1, a_end + 0.1)
ax4.set_ylim(-5, 5)
ax4.set_ylabel('$\dot{\phi}$', fontsize=20)
ax4.set_xlabel('$\phi$', fontsize=20)
ax4.legend(fontsize=10)

