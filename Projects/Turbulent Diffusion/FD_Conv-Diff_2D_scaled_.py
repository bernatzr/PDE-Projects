import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------------
# Two-dimensional finite difference code for convective-diffusion of heat or some
# other scalar quantity. Version 2 uses scaled quantities based on reference values
# ------------------------------------------------------------------------------------

j_m = 51
i_m = 101
v_x = 1
v_y = 0.5
u_ref = 1000  # kg/meter-squared
l_ref = 1000  # length in meters
v_ref = 20    # m/second
t_ref = l_ref / v_ref  # seconds 1000 / 20 = 50

q_a = np.zeros((j_m, i_m), dtype='float')

for j in range(j_m):
    for i in range(i_m):
        if i in [20] and j in [11]:
            q_a[j, i] = 1000 / u_ref


def k(i_max, j_max):
    kxy = np.ones((j_max, i_max), dtype='float')
    for i in range(i_max):
        for j in range(j_max):
            kxy[j, i] = (0.000005 + j*(0.01 - 0.000005) / (j_max-1))
            # kxy[j, i] = 0.005
            '''
            if i < 25:
                kxy[j, i] = .1
            else:
                kxy[j, i] = 1
            '''
            # kxy[j, i] = 4.0 * np.exp((j_max-j)/10)
            # kxy[j, i] = 0.1
            # kxy[i, j] = 0.001 * ran.randrange(90, 110, 1)
            # kxy[j, i] = 2.0 - j*1.0/j_max
    return kxy


def coef_cartesian(i_max, j_max, hx, hy, dt, kxy):
    ec = np.ones((j_max, i_max), dtype='float')
    wc = np.ones((j_max, i_max), dtype='float')
    sc = np.ones((j_max, i_max), dtype='float')
    nc = np.ones((j_max, i_max), dtype='float')
    cc = np.ones((j_max, i_max), dtype='float')
    cc_m = np.ones((j_max, i_max), dtype='float')
    for i in range(1,i_max-1):
        for j in range(1,j_max-1):
            ec[j, i] = (+kxy[j, i+1] + 4*kxy[j, i] - kxy[j, i-1])*dt/(4*hx*hx) - dt*v_x
            wc[j, i] = (-kxy[j, i+1] + 4*kxy[j, i] + kxy[j, i-1])*dt/(4*hx*hx) + dt*v_x
            sc[j, i] = (-kxy[j+1, i] + 4*kxy[j, i] + kxy[j-1, i])*dt/(4*hy*hy) + dt*v_y
            nc[j, i] = (+kxy[j+1, i] + 4*kxy[j, i] - kxy[j-1, i])*dt/(4*hy*hy) - dt*v_y
            cc[j, i] = 1.0 + 2.0*kxy[j, i]*dt/(hy*hy) + 2.0*kxy[j, i]*dt/(hx*hx)
    return ec, wc, sc, nc, cc, cc_m


def set_ics(i_max, j_max, hx, hy):
    u_loc = np.zeros((j_max, i_max), dtype=float)
    for i in range(i_max):
        for j in range(j_max):
            u_loc[j, i] = 0.0*hx + 0.0*hy
    return u_loc


def set_bcs(i_max, j_max, u_m, t):
    # at x=0 and x = c
    for j in range(j_max):
        u_m[j, 0] = u_m[j, 1]
        u_m[j, -1] = u_m[j, -2]
    # at y = 0 and y = d
    for i in range(i_max):
        u_m[0, i] = u_m[1, i]
        # u_m[i, -1] = 5 * np.sin(2*np.pi*t/365)
        u_m[-1, i] = 0.0
    return u_m


def tridag(n_max, b_vec, d_vec, a_vec, f_vec):
    u_vec = np.zeros(n_max, dtype=float)
    for n in range(2, n_max-1):
        m = b_vec[n]/d_vec[n-1]
        d_vec[n] = d_vec[n]-m*a_vec[n-1]
        f_vec[n] = f_vec[n]-m*f_vec[n-1]
    u_vec[n_max-2] = f_vec[n_max-2]/d_vec[n_max-2]
    for nn in range(n_max-3, 0, -1):
        u_vec[nn] = (f_vec[nn]-a_vec[nn]*u_vec[nn+1])/d_vec[nn]
    return u_vec


def swphrz(u_m, u_p, scoef, ecoef, ncoef, wcoef, ccoef, ccoef_m, imax, jmax, dt):
    b_vec = np.zeros(jmax, dtype=float)
    d_vec = np.zeros(jmax, dtype=float)
    a_vec = np.zeros(jmax, dtype=float)
    f_vec = np.zeros(jmax, dtype=float)
    for i in range(1, imax-1):
        for j in range(1, jmax-1):
            b_vec[j] = -scoef[j, i]
            d_vec[j] = ccoef[j, i]
            a_vec[j] = -ncoef[j, i]
            f_vec[j] = ecoef[j, i] * u_p[j, i+1] + wcoef[j, i] * u_p[j, i-1] + ccoef_m[j, i] * u_m[j, i] + dt*q_a[j, i]
        f_vec[1] = f_vec[1] + scoef[1, i] * u_p[0, i]
        f_vec[jmax-2] = f_vec[jmax-2] + ncoef[jmax-2, i] * u_p[jmax-1, i]
        u_vec = tridag(jmax, b_vec.copy(), d_vec.copy(), a_vec.copy(), f_vec.copy())
        u_p[1:jmax-1, i] = u_vec[1:jmax-1]
    return u_p


def swpvrt(u_m, u_p, scoef, ecoef, ncoef, wcoef, ccoef, ccoef_m, imax, jmax, dt):
    b_vec = np.zeros(imax, dtype=float)
    d_vec = np.zeros(imax, dtype=float)
    a_vec = np.zeros(imax, dtype=float)
    f_vec = np.zeros(imax, dtype=float)
    for j in range(1, jmax-1):
        for i in range(1, imax-1):
            b_vec[i] = -wcoef[j, i]
            d_vec[i] = ccoef[j, i]
            a_vec[i] = -ecoef[j, i]
            f_vec[i] = ncoef[j, i] * u_p[j+1, i] + scoef[j, i] * u_p[j-1, i] + ccoef_m[j, i] * u_m[j, i] + dt*q_a[j, i]
        f_vec[1] = f_vec[1] + wcoef[j, 1] * u_p[j, 0]
        f_vec[imax-2] = f_vec[imax-2] + ecoef[j, imax-2] * u_p[j, imax-1]
        u_vec = tridag(imax, b_vec.copy(), d_vec.copy(), a_vec.copy(), f_vec.copy())
        u_p[j, 1:imax-1] = u_vec[1:imax-1]
    return u_p


def plot_contours(u_star, x_max, y_max, i_max, j_max, t):
    u = u_ref * u_star
    hx = x_max/(i_max-1)
    hy = y_max/(j_max-1)
    tick_space = min(0.05, 0.05)
    x_nodes = np.arange(0, x_max+hx, hx)
    y_nodes = np.arange(0, y_max+hy, hy)
    x, y = np.meshgrid(x_nodes, y_nodes)
    ph = 6  # plot height in inches
    pw = 6 * float(x_max)/float(y_max)  # plot width in inches
    plt.figure('Finite Difference Solution', figsize=(pw, ph), linewidth=0)
    x_0 = 0.1
    y_0 = 0.1
    width = 0.85
    height = 0.85
    u_levels = np.linspace(u.min(), u.max(), 11                        )
    # cs = plt.contour(x, y, u, 10, colors='red', linewidths=0.5)
    # cs = plt.contour(x, y, u, 10, colors='red', linewidths=0.5)
    cs = plt.contour(x, y, u, levels=u_levels, colors='red', linewidths=0.5)
    plt.xticks(np.arange(tick_space, x_max, tick_space),fontsize='small')
    plt.yticks(np.arange(tick_space, y_max, tick_space), fontsize='small')
    plt.clabel(cs, inline=1, fmt='%1.2f')
    plt.title('u(x, y) Contours (i_max = ' + str(i_max) + ')', size=20)
    plt.xlabel('$x$', size=20)
    plt.ylabel('$y$', size=20)
    plt.grid(which='both', color='black', linestyle=':', linewidth='0.5')
    plot_file = 'temp_' + str(t) + '_' + str(i_max) +'.svg'
    plt.savefig(plot_file, facecolor='w', edgecolor='w', orientation='portrait', format='svg')
    plt.show()


def plot_3d(u, x_max, y_max, i_max, j_max):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # fig.add_axes(ax)
    hx = x_max/(i_max-1)
    hy = y_max/(j_max-1)
    tick_space = min(0.05, 0.05)
    x_nodes = np.arange(0, x_max+hx, hx)
    y_nodes = np.arange(0, y_max+hy, hy)
    x, y = np.meshgrid(x_nodes, y_nodes)
    # ax.plot_wireframe(x, y, u, rstride=1, cstride=1, color='black')
    # ax.plot_surface(x, y, u, rstride=1, cstride=1, cmap='jet')
    ax.plot_surface(x, y, u, vmin=u.min()+0.01, cmap='jet')
    # ax.contour(x, y, u, rstride=1, cstride=1, cmap='jet')
    ax.contour(x, y, u, cmap='jet')
    plt.show()


def new_u(u_m, i_max, j_max, time_step, hx, hy, dt, kxy):
    ec, wc, sc, nc, cc, cc_m = coef_cartesian(i_max, j_max, hx, hy, dt, kxy)
    max_f_iter = 100
    f_toler = 10**(-6)
    u_p = u_m.copy()
    u_i = u_m.copy()
    f_iter = 0
    converge_f = False
    while f_iter <= max_f_iter and not converge_f:
        if time_step % 2 == 0:
            u_p = swpvrt(u_m, u_p, sc, ec, nc, wc, cc, cc_m, i_max, j_max, dt)
        else:
            u_p = swphrz(u_m, u_p, sc, ec, nc, wc, cc, cc_m, i_max, j_max, dt)
        if np.amax(abs(u_p - u_i)) < f_toler:
            converge_f = True
        else:
            u_p = set_bcs(i_max,j_max,u_p,time_step*dt)
            u_i = u_p.copy()
        f_iter += 1
    return u_p


def main():
    x_max = 1.0
    y_max = 0.5
    i_max = 101
    j_max = 51
    dt = 1
    # k = 0.1
    kxy = k(i_max, j_max)
    hx = x_max/(i_max - 1)
    hy = y_max/(j_max - 1)
    u_m = set_ics(i_max, j_max, hx, hy)
    u_m = set_bcs(i_max, j_max, u_m, 0)
    u_p = u_m.copy()
    max_t_iter = 100
    t_step = 0
    time_iter = 1
    t_toler = 10 ** (-12)
    converge_t = False
    # plot_contours(u_p, x_max, y_max, i_max, j_max, t_step * dt)
    while time_iter <= max_t_iter and not converge_t:
        u_p = new_u(u_m, i_max, j_max, t_step, hx, hy, dt, kxy)
        u_p = set_bcs(i_max, j_max, u_p, t_step*dt)
        d_max_t = np.amax(abs(u_p - u_m))
        if d_max_t < t_toler:
            converge_t = True
            print(time_iter, abs(u_p - u_m).max())
        else:
            print(time_iter, abs(u_p - u_m).max())
            u_m = u_p.copy()
        time_iter += 1
    plot_contours(u_p, x_max, y_max, i_max, j_max, t_step*dt)
    # plot_3d(u_p, x_max, y_max, i_max, j_max)


main()
