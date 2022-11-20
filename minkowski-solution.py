# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 21:26:02 2022

@author: Usama
"""

import numpy as np
import matplotlib.pyplot as plt
import random as rnd

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 2.50
% matplotlib
qt

fig1, (ax1, ax5, ax2) = plt.subplots(3, 1)
fig2, (ax3, ax4) = plt.subplots(1, 2)
fig3, ax6 = plt.subplots(1)
fig4, ax7 = plt.subplots(1)

fig1.tight_layout()


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
    which allows us to find the maximum value
    of our potential - the true vacuum. Keeping the
    number of iterations high brings us very close
    to an undershoot.
    """

    for i in range(n):
        x += - U_der(x, E) / U_der2(x)
    return x


def dSdt(x, t, v, E):
    return 2 * (np.pi) ** 2 * (t) ** 3 * (0.5 * v ** 2 + U_corr(x, E))


def RK_22(x0, t0, v0, t_range, x_range, E):
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


def IntBisec(a_u, a_o, a_mid, E, N):
    for i in range(N):

        Phi_u = RK_22(a_u, t0, v0, t_range, x_range, E)
        amid = 0.5 * (a_u + a_o)

        Phi_mid = RK_22(amid, t0, v0, t_range, x_range, E)

        if abs(Phi_u[0, -1] - Phi_mid[0, -1]) < 1e-15:
            a_u = amid
        else:
            a_o = amid
    return amid


# Fundamental values we require:
N = 10
E = np.linspace(0.04, 0.08, N)

# Initial conditions for the Runge-Kutta algorithm.
t0 = 1e-15
v0 = 0
dt = 0.1
x_range = 2
t_range = 100

S = []
R = []
A_mid = []

for j in range(N):
    a_o = Newton_Raphson(-2, E[j], 2)
    print('An overshoot value:' + str(a_o))

    a_u = Newton_Raphson(-2, E[j], 100)
    print('An undershoot value:' + str(a_u))

    """The mid-point of the overshoot and the undershoot:"""
    a_mid = 0.5 * (a_o + a_u)
    a_mid = IntBisec(a_u, a_o, a_mid, E[j], 100)
    A_mid.append(a_mid)
    Phi_mid = RK_22(a_mid, t0, v0, t_range, x_range, E[j])

    # Inputs for the action:
    t = Phi_mid[0, :]
    x = Phi_mid[1, :]
    v = Phi_mid[-1, :]

    # Removing the waste end:
    for l in np.arange(0, len(t) - 1):
        if np.round(Phi_mid[0, l]) == 40:
            n = round(l)
            t_red = t[:n]
            x_red = x[:n]
            v_red = v[:n]
            break

    dSdt1 = 2 * (np.pi) ** 2 * (t_red) ** 3 * (0.5 * (v_red) ** 2 + U_corr(x_red, E[j]))

    M = len(dSdt1)
    S_integrated = (0.5 / M) * (t_red[0] + t_red[1]) * (dSdt1[0] + dSdt1[0])

    for m in range(M - 1):
        S_integrated += (0.5 / M) * (t_red[0] + t_red[1]) * (dSdt1[m])

    S.append(S_integrated)

    # Finding the index of intersection:
    for k in np.arange(0, len(x) - 1):
        if np.sign(Phi_mid[1, k]) != np.sign(Phi_mid[1, k + 1]):
            m = round(k)
            break

    R_i = 0.5 * (Phi_mid[0, m] + Phi_mid[0, m + 1])
    R.append(R_i)

    w = 2

    x_1 = np.linspace(-1.5, 1.5, 100)
    ax1.plot(t, x, linewidth=w)

    ax5.plot(Phi_mid[0, 1:], Phi_mid[-1, 1:], linewidth=w, label='$E$ = ' + str(round(E[j], 3)))
    ax2.plot(t_red, dSdt1, linewidth=w)
    ax6.plot(x_1, U_corr(x_1, E[j]), linewidth=3, label='$E$ = ' + str(round(E[j], 2)))
    print('ε = ' + str(E[j]))

R_1 = np.array(R)
S_1 = np.array(S)

R_ln = np.log(R_1)
S_ln = np.log(S_1)
E_ln = np.log(E)

# Checking the gradients of these lines to see if they're in agreement with
# Coleman's Thin-wall approximation:

R_polyfit = np.polyfit(E_ln, R_ln, 1)
S_polyfit = np.polyfit(E_ln, S_ln, 1)

m_R = R_polyfit[0]
m_S = S_polyfit[0]

m_S_error = (3 - abs(m_S)) / 3
m_R_error = 1 - abs(m_R)

print('The average gradient of the log(R) vs. log(E) is: ' + str(m_R) + ' with error ' + str(m_R_error))
print('The average gradient of the log(S) vs. log(E) is: ' + str(m_S) + ' with error ' + str(m_S_error))

s = 20

"""________________________________Finishing_Touches________________________"""
# Plots of derived action and radius:
ax1.set_ylim(-1.5, 1.5)
ax1.set_xlim(5, 30)
ax1.set_ylabel('$x$', fontsize=s)

ax2.set_xlim(5, 30)
ax2.set_ylim(-5e4, 1e5)
ax2.set_xlabel('$t$', fontsize=s)
ax2.set_ylabel('$\dot{\widetilde{B}}$', fontsize=s)

ax3.set_xlabel('log($E$)', fontsize=s)
ax3.set_ylabel('log($\widetilde{B}$)', fontsize=s)
ax3.plot(E_ln, S_ln, linewidth=2, label='<m> = ' + str(round(m_S, 3)))
ax3.plot(E_ln, -3 * E_ln - 4.5, '--', linewidth=2, label='Thin-Wall Action')

ax4.set_xlabel('log($E$)', fontsize=s)
ax4.set_ylabel('log($\widetilde{R}$)', fontsize=s)
ax4.plot(E_ln, R_ln, linewidth=2, label='<m> = ' + str(round(m_R, 3)))
ax4.plot(E_ln, -E_ln, '--', linewidth=2, label='Thin-Wall Radius')

ax5.set_xlim(5, 30)
ax5.set_ylim(0, 0.6)
ax5.set_ylabel('$\dot{x}$', fontsize=s)

ax6.set_xlim(-1.5, 1.5)
ax6.set_ylim(-0.2, 0.2)
ax6.set_xlabel('$x$', fontsize=s)
ax6.set_ylabel('$-\widetilde{U}$', fontsize=s)

A_polyfit = np.polyfit(E, A_mid, 1)
# ax7.plot(E, A_mid, 'o')
# ax7.plot(E, A_polyfit[0]*E + A_polyfit[1])
ax7.plot(R_1, S_1)

# ax7.set_ylabel('$\widetilde{\phi}(0)$')
# ax7.set_xlabel('$\widetilde{\epsilon}$')

ax1.tick_params(axis='both', which='major', labelsize=s)
ax2.tick_params(axis='both', which='major', labelsize=s)
ax3.tick_params(axis='both', which='major', labelsize=s)
ax4.tick_params(axis='both', which='major', labelsize=s)
ax5.tick_params(axis='both', which='major', labelsize=s)
ax6.tick_params(axis='both', which='major', labelsize=s)

ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=s)

ax3.legend(fontsize=s)
ax4.legend(fontsize=s)
ax6.legend(fontsize=s)
