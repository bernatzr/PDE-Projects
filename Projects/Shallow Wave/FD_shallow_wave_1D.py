import numpy as np
import matplotlib.pyplot as plt


def u_ics(max_n, hx):
    u = np.zeros(max_n, dtype='float')
    for i in range(max_n):
        u[i] = 0.0
    return u


def e_ics(max_n, hx):
    e = np.zeros(max_n+1, dtype='float')
    lo_x = 0.5*max_n*hx - 0.5
    hi_x = lo_x + 1.0
    for i in range(max_n):
        if lo_x < i*hx < hi_x:
            e[i] = 0.5*(i*hx - lo_x)*(hi_x - i*hx)
    return e


def u_bcs(t_step, dt,max_n, u):
    return u[max_n-1]


def e_bcs(t_step, dt,max_n, e):
    return e[1], e[max_n]


def u_new(u_m, e_i, max_n, hx, dt, g):
    u_i = np.zeros(max_n, dtype='float')
    for i in range(max_n):
        u_i[i] = u_m[i] - g * dt * (e_i[i+1] - e_i[i]) / hx
    return u_i


def e_new(e_m, u_i, max_n, hx, dt, cap_h):
    e_i = np.zeros(max_n+1, dtype='float')
    for i in range(1, max_n):
        e_i[i] = e_m[i] - cap_h * dt * (u_i[i] - u_i[i-1]) / hx
    e_i[0] = e_i[1]
    e_i[max_n] = e_i[max_n-1]
    return e_i


def main():
    lgt = 10
    max_n = 201
    hx = lgt/(max_n-1)
    dt = 0.01
    t_steps = 1000
    g = 1.5
    cap_h = 1.0
    lim_iter = 10000
    f_toler = 10**(-10)
    t_toler = 10**(-10)
    u_m = u_ics(max_n, hx)
    u_i = u_m.copy()
    u_p = u_m.copy()
    e_m = e_ics(max_n, hx)
    e_i = e_m.copy()
    e_p = e_m.copy()
    t_conv = False
    t_step = 0
    e_max = 0
    u_max = 0
    while t_step < t_steps and not t_conv:
        e_m[0], e_m[max_n] = e_bcs(t_step,dt,max_n,e_m)
        f_iter = 0
        f_conv = False
        while f_iter < lim_iter and not f_conv:
            u_p = u_new(u_m, e_i, max_n, hx, dt, g)
            e_p = e_new(e_m, u_i, max_n, hx, dt, cap_h)
            d_u = max([abs(u_p[i] - u_i[i]) for i in range(max_n)])
            d_e = max([abs(e_p[i] - e_i[i]) for i in range(max_n)])
            if d_u < f_toler and d_e < f_toler:
                f_conv = True
            else:
                f_iter += 1
                u_i = u_p.copy()
                e_i = e_p.copy()
        # print(f_conv, f_iter)
        d_u = max([abs(u_p[i] - u_m[i]) for i in range(max_n)])
        d_e = max([abs(e_p[i] - e_m[i]) for i in range(max_n)])
        if d_u < t_toler and d_e < t_toler:
            t_conv = True
        else:
            t_step += 1
            u_p[0] = 0.0
            e_p[0], e_p[max_n] = e_bcs(t_step, dt, max_n, e_p)
            u_m = u_p.copy()
            e_m = e_p.copy()
            u_max = max(max([abs(u_p[i]) for i in range(len(u_p))]), u_max)
            e_max = max(max([abs(e_p[i]) for i in range(len(u_p))]), e_max)
        # print(t_step, 'u = ', u_p)
        # print(t_step, 'e = ', e_p)
        plt.plot([0.5 * hx + i * hx for i in range(max_n - 1)], e_p[1:-1], 'k-')
        plt.plot([i * hx for i in range(max_n)], u_p, 'r--')
        plt.pause(0.01)
        plt.clf()
        plt.axis([0, 10, -0.1, 0.1])
    print(t_step)
    print(u_p)
    print(e_p)
    print('u_max = ', u_max)
    print('e_max = ', e_max)
    # plt.plot([0.5*hx + i*hx for i in range(max_n-1)], e_p[1:-1], 'k-')
    # plt.plot([i*hx for i in range(max_n)], u_p, 'r--')
    plt.show()


main()
