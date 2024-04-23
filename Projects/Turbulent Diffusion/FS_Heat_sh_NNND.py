from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

"""
n_max = number of FS terms in X
m_max = number of FS terms in Y
c = domain length in x
d = domain length in y
x1 = lower limit of x for non-zero q (Heaviside-like)
x2 = upper limit of x for non-zero q 
y1 = lower limit of y for non-zero q
y2 = upper limit of y for non-zero q 
k = diffusion coefficient
nodes_x = number of x plotting nodes
nodes_y = number of y plotting nodes
t_max = last time value 
t_steps = number of times steps between 0 and t_max
-----------
This solution requires the source function q(x,y,t) = q1(x,y) * q2(t) to 
facilitate ease in calculating the required triple integral used to dtermine
the time-dependent Fourier series coefficients B_nm(t)
"""
# --------------------  global constants --------------------

n_max = 30
m_max = 30
c = 1.0
d = 1.0
x1 = 0.445
x2 = 0.555
y1 = 0.445
y2 = 0.555
k = 0.1
u_ref = 1000
nodes_x = 51
x_nodes = np.linspace(0, c, nodes_x, dtype=float)
nodes_y = 51
y_nodes = np.linspace(0, d, nodes_y, dtype=float)
t_max = 10
t_steps = 2
t_nodes = np.linspace(0, t_max, t_steps, dtype=float)

x_evals = np.zeros(shape=n_max, dtype=float)
for index in range(0, n_max):
    x_evals[index] = float(index*np.pi / c)


y_evals = np.zeros(shape=n_max, dtype=float)
for index in range(0, n_max):
    y_evals[index] = float((2*index+1)*np.pi / (2*d))

# ------------------------------------------------------------------


def x_ef(x, n):
    if n != 0:
        return np.cos(x_evals[n] * x) * np.sqrt(2 / c)
    else:
        return 1 / np.sqrt(c)


def y_ef(y, n):
    return np.cos(y_evals[n]*y)*np.sqrt(2 / d)


def check_othonormal(ef, max_n, ul):
    ortho_norm = True
    ip_m = np.zeros(shape=(max_n, max_n), dtype=float)
    for m in range(max_n):
        for n in range(max_n):
            ip_m[m, n] = quad(lambda x: ef(x, m) * ef(x, n), 0, ul)[0]
            if m == n and abs(ip_m[m, n] - 1.0) > 10**(-8) and abs(ip_m[m, n] - 0.0) > 10**(-8):
                ortho_norm = False
            elif m != m and abs(ip_m[m, n]) > 10**(-8):
                ortho_norm = False
    return ortho_norm, ip_m


def f(x, y):
    return 0.0


def q(x, y, tau):
    return q1(x, y) * q2(tau)


def q1(x, y):
    # return 16 * x * (c - x) * y * (d - y)
    if x1 <= x <= x2 and y1 <= y <= y2:
        return 1.0
    else:
        return 0.0


def q2(tau):
    return 1.0


def f_nm_coef():
    fc = np.zeros(shape=(n_max, m_max), dtype=float)
    for n in range(n_max):
        for m in range(m_max):
            fc[n, m] = quad(lambda x: quad(lambda y: f(x, y) * y_ef(y, m), 0, d)[0] * x_ef(x, n), 0, c)[0]
    return fc


def q_nm1_coef():
    q_nm1 = np.zeros(shape=(n_max, m_max), dtype=float)
    for n in range(n_max):
        for m in range(m_max):
            q_nm1[n, m] = quad(lambda x: quad(lambda y: q1(x, y) * y_ef(y, m), y1, y2)[0] * x_ef(x, n), x1, x2)[0]
    return q_nm1


def q_nm_coef(t, q_nm1):
    q_nm = np.zeros(shape=(n_max, m_max), dtype=float)
    for n in range(n_max):
        for m in range(m_max):
            q_nm[n, m] = q_nm1[n, m] * quad(lambda tau: q2(t) *
                                            np.exp(-k * (x_evals[n]**2 + y_evals[m]**2) * (t-tau)), 0, t)[0]
    return q_nm


def b_nm_coef(q_nm, f_nm, t):
    b_nm = np.zeros(shape=(n_max, m_max), dtype=float)
    for n in range(n_max):
        for m in range(m_max):
            b_nm[n, m] = q_nm[n, m] + (f_nm[n, m]) * np.exp(-k*(x_evals[n]**2 + y_evals[m]**2) * t)
    return b_nm


def f_s(x, y, fc):
    return sum([sum([fc[i, j] * y_ef(y, j) * x_ef(x, i) for j in range(0, m_max)]) for i in range(0, n_max)])


def q_s(x, y, q_nm):
    return sum([sum([q_nm[i, j] * y_ef(y, j) * x_ef(x, i) for j in range(0, m_max)]) for i in range(0, n_max)])


def u_s(x, y, b_nm):
    return sum([sum([b_nm[i, j] * y_ef(y, j) * x_ef(x, i) for j in range(0, m_max)]) for i in range(0, n_max)])


def plot_surface(z, title):
    from matplotlib import cm
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "3d"})
    x, y = np.meshgrid(x_nodes, y_nodes)
    ax.plot_surface(x, y, z, vmin=z.min() * 2, cmap=cm.Blues)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlim(-1.0, 1.0)
    # ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
    plt.show()


def plot_contours(z, title):
    x, y = np.meshgrid(x_nodes, y_nodes)
    ph = 6  # plot height in inches
    pw = 6 * float(c)/float(d)  # plot width in inches
    tick_space = min(0.1, 0.1)
    plt.figure('Fourier Series Solution', figsize=(pw, ph), linewidth=0)
    # u_levels = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    cs = plt.contour(x, y, z, levels=11, colors='red', linewidths=0.5)
    plt.clabel(cs, inline=1, fmt='%1.2f')
    plt.title(title, size=20)
    plt.xlabel('$x$', size=20)
    plt.ylabel('$y$', size=20)
    plt.grid(which='both', color='black', linestyle=':', linewidth=0.5)
    plt.xticks(np.arange(tick_space, c, tick_space),fontsize='small')
    plt.yticks(np.arange(tick_space, c, tick_space), fontsize='small')
    # plot_file = 'fs_sol_SH.svg'
    # plt.savefig(plot_file, facecolor='w', edgecolor='w', orientation='portrait', format='svg')
    plt.show()


def plot_wireframe(z, title):
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "3d"})
    x, y = np.meshgrid(x_nodes, y_nodes)
    ax.plot_wireframe(x, y, z)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlim(-1.0, 1.0)
    # z_ticks = np.arange(int(z.min()), int(z.max()), 1)
    # ax.set(xticklabels=x_nodes, yticklabels=y_nodes, zticklabels=z_ticks)
    plt.show()


def plot_fs(f_nm):
    z = np.zeros(shape=(len(y_nodes), len(x_nodes)), dtype=float)
    for j in range(len(x_nodes)):
        for i in range(len(y_nodes)):
            z[i, j] = f_s(x_nodes[j], y_nodes[i], f_nm)
    plot_wireframe(z, 'Fourier f surface')


def plot_qs(q_nm):
    z = np.zeros(shape=(len(y_nodes), len(x_nodes)), dtype=float)
    for j in range(len(x_nodes)):
        for i in range(len(y_nodes)):
            z[i, j] = q_s(x_nodes[j], y_nodes[i], q_nm)
    plot_wireframe(z, 'Fourier q surface')


def plot_q(t):
    z = np.zeros(shape=(len(y_nodes), len(x_nodes)), dtype=float)
    for i in range(len(y_nodes)):
        for j in range(len(x_nodes)):
            z[i, j] = q(x_nodes[j], y_nodes[i], t)
    plot_wireframe(z, 'q surface')


def plot_f():
    z = np.zeros(shape=(len(y_nodes), len(x_nodes)), dtype=float)
    for i in range(len(y_nodes)):
        for j in range(len(x_nodes)):
            z[i, j] = f(x_nodes[j], y_nodes[i])
    plot_wireframe(z, 'f surface')


def main():
    o_n = check_othonormal(x_ef, n_max, c)
    if o_n[0] is False:
        print('**** x_ef not orthonormal')
    o_n = check_othonormal(y_ef, n_max, d)
    if o_n[0] is False:
        print('**** y_ef not orthonormal')
    # plot_f()
    f_nm = f_nm_coef()
    # plot_fs(f_nm)
    q_nm1 = q_nm1_coef()
    plot_q(1.0)
    plot_qs(q_nm1)
    z_star = np.zeros(shape=(len(y_nodes), len(x_nodes)), dtype=float)
    for t in range(1, len(t_nodes)):
        q_nm = q_nm_coef(t_nodes[t], q_nm1)
        b_nm = b_nm_coef(q_nm, f_nm, t_nodes[t])
        for i in range(len(y_nodes)):
            z_star[i, :] = u_s(x_nodes, y_nodes[i], b_nm)
    #   plot_surface(z, 'Fourier u surface' )
        z = u_ref * z_star
        plot_contours(z, 'Fourier u surface')
    #   plot_wireframe(z, 'Fourier u surface')


main()
