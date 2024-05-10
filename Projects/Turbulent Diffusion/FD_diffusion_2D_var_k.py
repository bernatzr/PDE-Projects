import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ------------------------------------------------------------------------------------
# Two-dimensional finite difference code for strictly diffusion of heat or some
# other scalar quantity.
# ------------------------------------------------------------------------------------

# ------------------ Global Constants ------------------------------------------------


x_max = 1.0
y_max = 1.0
i_max = 51
j_max = 51
dt = 0.1
hx = x_max / (i_max - 1)
hy = y_max / (j_max - 1)
time_end = 2
time_step_max = int(time_end / dt)

k = 0.1 * np.ones((j_max, i_max), dtype='float')
for j in range(j_max):
    for i in range(i_max):
        k[j, i] = 0.01 + 1.0*j*(j_max-j)**2 / (j_max*j_max)

ec = np.ones((j_max, i_max), dtype=float)
wc = np.ones((j_max, i_max), dtype=float)
sc = np.ones((j_max, i_max), dtype=float)
nc = np.ones((j_max, i_max), dtype=float)
cc = np.ones((j_max, i_max), dtype=float)
cc_m = np.ones((j_max, i_max), dtype=float)
for j in range(1,j_max-1):
    for i in range(1, i_max-1):
        ec[j, i] = dt * k[j, i] / (hx * hx) + dt*(k[j, i+1] - k[j, i-1])/(4*hx*hx)
        wc[j, i] = dt * k[j, i] / (hx * hx) - dt*(k[j, i+1] - k[j, i-1])/(4*hx*hx)
        nc[j, i] = dt * k[j, i] / (hy * hy) + dt*(k[j+1, i] - k[j-1, i])/(4*hy*hy)
        sc[j, i] = dt * k[j, i] / (hy * hy) - dt*(k[j+1, i] - k[j-1, i])/(4*hy*hy)
        cc[j, i] = (1.0 + 2.0 * dt* k[j, i] / (hx * hx) + dt * 2.0 * k[j, i]/(hy* hy))
        cc_m[j, i] = 1.0


q_a = np.zeros((j_max, i_max), dtype='float')
for jj in range(j_max):
    for ii in range(i_max):
        if 0.4 <= hx*ii <= 0.6 and 0.4 <= hy*jj <= 0.6:
            q_a[jj, ii] = 0.0
        else:
            q_a[jj, ii] = 0.0


def set_ics():
    u_loc = np.zeros((j_max, i_max), dtype=float)
    for i in range(i_max):
        for j in range(j_max):
            u_loc[j, i] = 0.0*hx + 0.0*hy
    return u_loc


def set_bcs(u_m):
    # at x=0 and x = c
    for j in range(j_max):
        u_m[j, 0] = 1.0
        u_m[j, -1] = u_m[j, -2]
    # at y = 0 and y = d
    for i in range(i_max):
        u_m[0, i] = u_m[1, i]
        u_m[-1, i] = u_m[-2, i]
    return u_m


def tridag(n_max, b_vec, d_vec, a_vec, f_vec):
    u_vec = np.zeros(n_max, dtype=float)
    for n in range(2, n_max-1):
        m = b_vec[n]/d_vec[n-1]
        d_vec[n] = d_vec[n]-m*a_vec[n-1]
        f_vec[n] = f_vec[n]-m*f_vec[n-1]
    u_vec[n_max-2] = f_vec[n_max-2]/d_vec[n_max-2]
    for nn in range(n_max-3, 0, -1):
        if abs(d_vec[nn]) < 10**(-2):
            print(str(nn))
        u_vec[nn] = (f_vec[nn]-a_vec[nn]*u_vec[nn+1])/d_vec[nn]
    return u_vec


def swphrz(u_m, u_p):
    b_vec = np.zeros(j_max, dtype=float)
    d_vec = np.zeros(j_max, dtype=float)
    a_vec = np.zeros(j_max, dtype=float)
    f_vec = np.zeros(j_max, dtype=float)
    for i in range(1, i_max-1):
        for j in range(1, j_max-1):
            b_vec[j] = -sc[j, i]
            d_vec[j] = cc[j, i]
            a_vec[j] = -nc[j, i]
            f_vec[j] = ec[j, i] * u_p[j, i+1] + wc[j, i] * u_p[j, i-1] + cc_m[j, i] * u_m[j, i] + dt*q_a[j,i]
        f_vec[1] = f_vec[1] + sc[1, i] * u_p[0, i]
        f_vec[j_max-2] = f_vec[j_max-2] + nc[j_max-2, i] * u_p[j_max-1, i]
        u_vec = tridag(j_max, b_vec.copy(), d_vec.copy(), a_vec.copy(), f_vec.copy())
        u_p[1:j_max-1, i] = u_vec[1:j_max-1]
    return u_p


def swpvrt(u_m, u_p):
    b_vec = np.zeros(i_max, dtype=float)
    d_vec = np.zeros(i_max, dtype=float)
    a_vec = np.zeros(i_max, dtype=float)
    f_vec = np.zeros(i_max, dtype=float)
    for j in range(1, j_max-1):
        for i in range(1, i_max-1):
            b_vec[i] = -wc[j, i]
            d_vec[i] = cc[j, i]
            a_vec[i] = -ec[j, i]
            f_vec[i] = nc[j, i] * u_p[j+1, i] + sc[j, i] * u_p[j-1, i] + cc_m[j, i] * u_m[j, i] + dt*q_a[j, i]
        f_vec[1] = f_vec[1] + wc[j, 1] * u_p[j, 0]
        f_vec[i_max-2] = f_vec[i_max-2] + ec[j, i_max-2] * u_p[j, i_max-1]
        u_vec = tridag(i_max, b_vec.copy(), d_vec.copy(), a_vec.copy(), f_vec.copy())
        u_p[j, 1:i_max-1] = u_vec[1:i_max-1]
    return u_p


def new_u(u_m, time_step):
    max_f_iter = 100
    f_toler = 10**(-6)
    u_p = u_m.copy()
    u_i = u_m.copy()
    f_iter = 0
    converge_f = False
    while f_iter <= max_f_iter and not converge_f:
        if time_step % 2 == 0:
            u_p = swpvrt(u_m, u_p)
        else:
            u_p = swphrz(u_m, u_p)
        if np.amax(abs(u_p - u_i)) < f_toler:
            converge_f = True
        else:
            u_p = set_bcs(u_p)
            u_i = u_p.copy()
        f_iter += 1
    return u_p


def plot_contours(u, t):
    tick_space = min(0.1, 0.1)
    x_nodes = np.arange(0, x_max+hx, hx)
    y_nodes = np.arange(0, y_max+hy, hy)
    x, y = np.meshgrid(x_nodes, y_nodes)
    ph = 6  # plot height in inches
    pw = 6 * float(x_max)/float(y_max)  # plot width in inches
    plt.figure('Finite Difference Solution', figsize=(pw, ph), linewidth=0)
    u_levels = np.linspace(u.min(), u.max(), 11)
    cs = plt.contour(x, y, u, levels=u_levels, colors='red', linewidths=0.5)
    plt.xticks(np.arange(tick_space, x_max, tick_space),fontsize='small')
    plt.yticks(np.arange(tick_space, y_max, tick_space), fontsize='small')
    plt.clabel(cs, inline=1, fmt='%1.2f')
    plt.title('u Contours (i_max = ' + str(i_max) + ')', size=20)
    plt.xlabel('$x$', size=20)
    plt.ylabel('$y$', size=20)
    plt.grid(which='both', color='black', linestyle=':', linewidth='0.5')
    plot_file = 'temp_' + str(t) + '_' + str(i_max) +'.svg'
    plt.savefig(plot_file, facecolor='w', edgecolor='w', orientation='portrait', format='svg')
    plt.show()


def plot_3d(u):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    tick_space = min(0.05, 0.05)
    x_nodes = np.arange(0, x_max+hx, hx)
    y_nodes = np.arange(0, y_max+hy, hy)
    x, y = np.meshgrid(x_nodes, y_nodes)
    ax.plot_surface(x, y, u, vmin=u.min()+0.01, cmap='jet')
    ax.contour(x, y, u, cmap='jet')
    plt.show()


def ani_main(u_list):
    ph = 6  # plot height in inches
    pw = 6 * float(x_max)/float(y_max)  # plot width in inches
    figure, ax = plt.subplots(figsize=(pw, ph))
    x_nodes = np.arange(0, x_max + hx, hx)
    y_nodes = np.arange(0, y_max + hy, hy)
    tick_space = min(0.05, 0.05)
    x, y = np.meshgrid(x_nodes, y_nodes)
    # cs = ax.contour(x, y, u_list[:, :, 1], levels=11, colors='red', linewidths=0.5)
    u_star = u_list[1]
    u = u_star
    cs = ax.contour(x, y, u, levels=11, colors='red', linewidths=0.5)
    # cs = ax.contourf(x, y, u_list[:, :, 1], levels=11)
    ax.clabel(cs, cs.levels, inline=True, fontsize=10)

    def animation_function(i):
        ax.clear()
        # cs = ax.contour(x, y, u_list[:, :, i], levels=11, colors='red', linewidths=0.5)
        u_star = u_list[i+1]
        u = u_star
        cs = ax.contour(x, y, u, levels=11, colors='red', linewidths=0.5)
        ax.clabel(cs, cs.levels, inline=True, fontsize=10)
        # ax.contourf(x, y, u_list[:,:, i+1], levels=11)
        # ax.xticks(np.arange(tick_space, x_max, tick_space), fontsize='small')
        # ax.yticks(np.arange(tick_space, y_max, tick_space), fontsize='small')
        # ax.clabel(cs, inline=1, fmt='%1.2f')
        # plt.title('Contours (i_max = ' + str(i_max) + ')', size=20)
        # plt.xlabel('$x$', size=20)
        # plt.ylabel('$y$', size=20)
        # plt.grid(which='both', color='black', linestyle=':', linewidth='0.5')
        return figure

    animation = FuncAnimation(figure, func=animation_function, frames=len(u_list) - 1, interval=1000)
    plt.show()


def main():
    u_m = set_ics()
    u_m = set_bcs(u_m)
    u_p = u_m.copy()
    time_step = 0
    t_toler = 10 ** (-6)
    converge_t = False
    u_list = [u_p]
    while time_step <= time_step_max and not converge_t:
        u_p = new_u(u_m, time_step)
        u_p = set_bcs(u_p)
        d_max_t = np.amax(abs(u_p - u_m))
        if d_max_t < t_toler:
            converge_t = True
            print(time_step, abs(u_p - u_m).max())
        else:
            u_m = u_p.copy()
            u_list.append(u_p)
        time_step += 1
    plot_contours(u_p, time_step*dt)
    ani_main(u_list)
    # plot_3d(u_p)


main()
