import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
This version of convection-diffusion equation modifies the east and west coefficients in
the sweep functions.
"""


def q(x, y):
    if 0.28 <= x <= 0.31 and 0.28 <= y <= 0.31:
        return 100.0
    elif 0.48 <= x <= 0.51 and 0.48 <= y <= 0.51:
        return 50.0
    else:
        return 0.0


def coef_cartesian(hx, hy, k, dt):
    ec = k*dt/(hx*hx)
    wc = k*dt/(hx*hx)
    sc = k*dt/(hy*hy)
    nc = k*dt/(hy*hy)
    cc = 1.0 + 2.0*k*dt/(hx*hx) + 2.0*k*dt/(hy*hy)
    cc_m = 1.0
    return ec, wc, sc, nc, cc, cc_m


def q_array(x_max, y_max, hx, hy):
    x_nodes = np.arange(0, x_max+hx, hx)
    y_nodes = np.arange(0, y_max+hy, hy)
    q_a = np.zeros((len(y_nodes), len(x_nodes)), dtype='float')
    q_a[20, 10] = 100
    q_a[35, 15] = 50
    '''
    for i in range(len(x_nodes)):
        for j in range(len(y_nodes)):
            q_a[j, i] = q(x_nodes[i], y_nodes[j])
    '''
    return q_a


def set_ics(i_max, j_max, hx, hy):
    u_loc = np.zeros((j_max,i_max), dtype=float)
    for i in range(i_max):
        for j in range(j_max):
            u_loc[j, i] = 0.0*hx + 0.0*hy
    return u_loc


def set_bcs(i_max, j_max, hx, hy, u_loc):
    # at x=0 and x = c
    for j in range(j_max):
        u_loc[j, 0] = u_loc[j, 0]
        u_loc[j, i_max-1] = u_loc[j, i_max-2]
    # at y = 0 and y = d
    for i in range(i_max):
        u_loc[0, i] = u_loc[1, i]
        u_loc[j_max-1, i] = u_loc[j_max-2, i]
    return u_loc


def tridag(n_max, b_vec, d_vec, a_vec, f_vec):
    u_vec = np.zeros(n_max, dtype=float)
    for n in range(2, n_max-1):
        m = b_vec[n]/d_vec[n-1]
        d_vec[n] = d_vec[n]-m*a_vec[n-1]
        f_vec[n] = f_vec[n]-m*f_vec[n-1]
    u_vec[n_max-2] = f_vec[n_max-2]/d_vec[n_max-2]
    for n in range(n_max-3, 0, -1):
        u_vec[n] = (f_vec[n]-a_vec[n]*u_vec[n+1])/d_vec[n]
    return u_vec


def swphrz(u_m, u_p, scoef, ecoef, ncoef, wcoef, ccoef, ccoef_m, imax, jmax, q_a, hx, dt, v_vec):
    b_vec = np.zeros(jmax, dtype=float)
    d_vec = np.zeros(jmax, dtype=float)
    a_vec = np.zeros(jmax, dtype=float)
    f_vec = np.zeros(jmax, dtype=float)
    for i in range(1, imax-1):
        for j in range(1, jmax-1):
            b_vec[j] = -scoef
            d_vec[j] = ccoef
            a_vec[j] = -ncoef
            f_vec[j] = (ecoef - v_vec[j]*dt/(2*hx)) * u_p[j, i+1] \
                + (wcoef + v_vec[j]*dt/(2*hx)) * u_p[j, i-1] + ccoef_m * u_m[j, i] \
                + dt * q_a[j, i]
        f_vec[1] = f_vec[1] + scoef*u_p[0, i]
        f_vec[jmax-2] = f_vec[jmax-2] + ncoef*u_p[jmax-1, i]
        u_vec = tridag(jmax, b_vec, d_vec, a_vec, f_vec)
        u_p[1:jmax-1, i] = u_vec[1:jmax-1]
    return u_p


def swpvrt(u_m, u_p, scoef, ecoef, ncoef, wcoef, ccoef, ccoef_m, imax, jmax, q_a, hx, dt, v_vec):
    b_vec = np.zeros(imax, dtype=float)
    d_vec = np.zeros(imax, dtype=float)
    a_vec = np.zeros(imax, dtype=float)
    f_vec = np.zeros(imax, dtype=float)
    for j in range(1, jmax-1):
        for i in range(1, imax-1):
            b_vec[i] = -(wcoef + v_vec[j]*dt/(2*hx))
            d_vec[i] = ccoef
            a_vec[i] = -(ecoef - v_vec[j]*dt/(2*hx))
            f_vec[i] = ncoef * u_p[j+1, i] + scoef * u_p[j-1, i] + ccoef_m * u_m[j, i] \
                + dt * q_a[j, i]
        f_vec[1] = f_vec[1]+wcoef*u_p[j,0]
        f_vec[imax-2] = f_vec[imax-2]+ecoef*u_p[j, imax-1]
        u_vec = tridag(imax, b_vec, d_vec, a_vec, f_vec)
        u_p[j,1:imax-1] = u_vec[1:imax-1]
    return u_p


def v(jmax, hy, v_max):
    v_vec = np.zeros(jmax, dtype='float')
    for j in range(jmax):
        v_vec[j] = v_max
        # v_vec[j] = v_max * 4.0 * j * hy * (1.0 - j*hy)
        # v_vec[j] = 10*np.exp(-150*(j*hy - 0.5)**2)
    return v_vec


def conv(u, v_vec, hx, imax, jmax):
    c_a = np.zeros((jmax, imax), dtype='float')
    for j in range(1, jmax-1):
        for i in range(1, imax-1):
            c_a[j, i] = -v_vec[j] * (u[j, i+1] - u[j, i-1]) / (2*hx)
    return c_a


def plot_contours(u, x_max, y_max, i_max, j_max, v_max):
    hx = x_max/(i_max-1)
    hy = y_max/(j_max-1)
    x_tick_space = 0.2
    y_tick_space = 0.1
    x_nodes = np.arange(0, x_max+hx, hx)
    y_nodes = np.arange(0, y_max+hy, hy)
    x, y = np.meshgrid(x_nodes, y_nodes)
    ph = 10 * float(y_max)/float(x_max)   # plot height in inches
    pw = 10  # plot width in inches
    fig = plt.figure('Finite Difference Solution', figsize=(pw, ph), linewidth=0)
    x_0 = 0.15
    y_0 = 0.15
    width = 0.75
    height = 0.75
    fig.add_axes([x_0, y_0, width, height])
    u_levels = np.linspace(1.0, u.max(), 16)
    cs = plt.contour(x, y, u, levels=u_levels, colors='red', linewidths=0.5)
    plt.xticks(np.arange(x_tick_space, x_max, x_tick_space), fontsize='small')
    plt.yticks(np.arange(y_tick_space, y_max, y_tick_space), fontsize='small')
    plt.clabel(cs, inline=1, fmt='%1.2f')
    plt.title('u(x, y) Contours (i_max = ' + str(i_max) + ')', size=10)
    plt.xlabel('$x$', size=10)
    plt.ylabel('$y$', size=10)
    plt.grid(which='both', color='black', linestyle=':', linewidth='0.5')
    plot_file = 'u(x,y)_' + str(v_max) + '_' + str(i_max) + '.png'
    plt.savefig(plot_file, facecolor='w', edgecolor='w', orientation='portrait', format='png')
    plt.show()


def plot_3d(u, x_max, y_max, i_max, j_max, v_max):
    fig = plt.figure()
    ax = Axes3D(fig)
    hx = x_max/(i_max-1)
    hy = y_max/(j_max-1)
    # tick_space = min(0.05, 0.05)
    x_nodes = np.arange(0, x_max+hx, hx)
    y_nodes = np.arange(0, y_max+hy, hy)
    x, y = np.meshgrid(x_nodes, y_nodes)
    ax.plot_wireframe(x, y, u, rstride=1, cstride=1, color='black')
    # ax.plot_surface(x, y, u, rstride=1, cstride=1, cmap='jet')
    ax.contour(x, y, u, cmap='jet')
    plt.show()


def plot_surface(u, x_max, y_max, i_max, j_max, v_max):
    from matplotlib import cm
    hx = x_max / (i_max-1)
    hy = y_max / (j_max-1)
    # tick_space = min(0.05, 0.05)
    x_nodes = np.arange(0, x_max+hx, hx)
    y_nodes = np.arange(0, y_max+hy, hy)
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "3d"})
    x, y = np.meshgrid(x_nodes, y_nodes)
    ax.plot_surface(x, y, u, cmap=cm.Blues)
    ax.set_title('Pollution Surface')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlim(u.min()-0.1, u.max()+0.1)
    ax.contour(x, y, u, cmap='jet')
    plt.show()


def main(time_steps):
    x_max = 5.0
    y_max = 1.0
    i_max = 101
    j_max = 51
    dt = 1.0
    k = 6.0 / 10000.
    v_max = 20.0 / 1000.
    hx = x_max/(i_max - 1)
    hy = y_max/(j_max - 1)
    q_a = q_array(x_max, y_max, hx, hy)
    v_vec = v(j_max, hy, v_max)
    ec, wc, sc, nc, cc, cc_m = coef_cartesian(hx, hy, k, dt)
    u_m = set_ics(i_max, j_max, hx, hy)
    u_m = set_bcs(i_max, j_max, hx, hy, u_m)
    u_i = u_m.copy()
    u_p = u_m.copy()
    max_iter = 10
    f_toler = 10**(-6)
    t_toler = 10**(-2)
    time_step = 1
    converge_t = False
    d_max_t = 10.0
    while time_step <= time_steps and not converge_t:
        f_iter = 0
        converge_f = False
        while (f_iter <= max_iter) and (not converge_f):
            if time_step % 2 == 0:
                u_p = swpvrt(u_m, u_p, sc, ec, nc, wc, cc, cc_m, i_max, j_max, q_a, hx, dt, v_vec)
            else:
                u_p = swphrz(u_m, u_p, sc, ec, nc, wc, cc, cc_m, i_max, j_max, q_a, hx, dt, v_vec)
            if np.amax(abs(u_p - u_i)) < f_toler:
                converge_f = True
            else:
                u_i = u_p.copy()
            f_iter += 1
        d_max_t = np.amax(abs(u_p - u_m)[:, i_max-2])
        i, j = np.unravel_index(abs(u_p - u_m).argmax(), u_p.shape)
        # print(i, j)
        # print(abs(u_p - u_m)[i, j])
        if 0.001 < d_max_t < t_toler:
            converge_t = True
            print(time_step, abs(u_p - u_m).max())
        else:
            u_p = set_bcs(i_max, j_max, hx, hy, u_p)
            u_i = u_p.copy()
            u_m = u_p.copy()
        time_step += 1
    print(str(converge_t), d_max_t)
    u_p = set_bcs(i_max, j_max, hx, hy, u_p)
    plot_contours(u_p, x_max, y_max, i_max, j_max, v_max)
    # plot_surface(u_p, x_max, y_max, i_max, j_max, v_max)


main(4000)
