import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------------------------------------------------------------
# This version
# ------------------------------------------------------------------------------------


def coef_cartesian(hx, hy, k, dt):
    ec = k*dt/(hx*hx)
    wc = k*dt/(hx*hx)
    sc = k*dt/(hy*hy)
    nc = k*dt/(hy*hy)
    cc = 1.0 + 2.0*k*dt/(hx*hx) + 2.0*k*dt/(hy*hy)
    cc_m = 1.0
    return ec, wc, sc, nc, cc, cc_m


def set_ics(i_max,j_max,hx,hy):
    u_loc = np.zeros((j_max,i_max), dtype=float)
    for i in range(i_max):
        for j in range(j_max):
            u_loc[j,i] = 0.0*hx + 0.0*hy
    return u_loc


def set_bcs(i_max,j_max,u_loc, t):
    # at x=0 and x = c
    for j in range(j_max):
        u_loc[0, j] = u_loc[1, j]
        u_loc[-1, j] = u_loc[-2,j]
    # at y = 0 and y = d : Neumann
    for i in range(i_max):
        u_loc[i, 0] = u_loc[i,1]
        u_loc[i, -1] = 5*np.sin(2*np.pi*t/365)
    return u_loc


def tridag(n_max,b_vec,d_vec,a_vec,f_vec):
    u_vec = np.zeros(n_max, dtype=float)
    for n in range(2,n_max-1):
        m=b_vec[n]/d_vec[n-1]
        d_vec[n]=d_vec[n]-m*a_vec[n-1]
        f_vec[n]=f_vec[n]-m*f_vec[n-1]
    u_vec[n_max-2]=f_vec[n_max-2]/d_vec[n_max-2]
    for n in range(n_max-3,0,-1):
        u_vec[n]=(f_vec[n]-a_vec[n]*u_vec[n+1])/d_vec[n]
    return u_vec


def swphrz(u_m,u_p,scoef,ecoef,ncoef,wcoef,ccoef,ccoef_m,imax,jmax):
    b_vec = np.zeros(jmax, dtype=float)
    d_vec = np.zeros(jmax, dtype=float)
    a_vec = np.zeros(jmax, dtype=float)
    f_vec = np.zeros(jmax, dtype=float)
    for i in range(1,imax-1):
        for j in range(1,jmax-1):
            b_vec[j]=-scoef
            d_vec[j]=ccoef
            a_vec[j]=-ncoef
            f_vec[j]=ecoef*u_p[j,i+1]+wcoef*u_p[j,i-1]+ccoef_m*u_m[j,i]
        f_vec[1]=f_vec[1]+scoef*u_p[0,i]
        f_vec[jmax-2]=f_vec[jmax-2]+ncoef*u_p[jmax-1,i]
        u_vec=tridag(jmax,b_vec,d_vec,a_vec,f_vec)
        u_p[1:jmax-1,i] = u_vec[1:jmax-1]
    return u_p


def swpvrt(u_m,u_p,scoef,ecoef,ncoef,wcoef,ccoef,ccoef_m,imax,jmax):
    b_vec = np.zeros(imax, dtype=float)
    d_vec = np.zeros(imax, dtype=float)
    a_vec = np.zeros(imax, dtype=float)
    f_vec = np.zeros(imax, dtype=float)
    for j in range(1,jmax-1):
        for i in range(1,imax-1):
            b_vec[i]=-wcoef
            d_vec[i]=ccoef
            a_vec[i]=-ecoef
            f_vec[i]=ncoef*u_p[j+1,i]+scoef*u_p[j-1,i]+ccoef_m*u_m[j,i]
        f_vec[1]=f_vec[1]+wcoef*u_p[j,0]
        f_vec[imax-2]=f_vec[imax-2]+ecoef*u_p[j,imax-1]
        u_vec=tridag(imax,b_vec,d_vec,a_vec,f_vec)
        u_p[j,1:imax-1] = u_vec[1:imax-1]
    return u_p


def plot_contours(u, x_max, y_max, i_max, j_max, t):
    hx = x_max/(i_max-1)
    hy = y_max/(j_max-1)
    tick_space = min(0.05, 0.05)
    x_nodes=np.arange(0, x_max+hx, hx)
    y_nodes=np.arange(0, y_max+hy, hy)
    x, y = np.meshgrid(x_nodes, y_nodes)
    ph = 8  # plot height in inches
    pw = 8 * float(x_max)/float(y_max)  # plot width in inches
    fig = plt.figure('Finite Difference Solution', figsize=(pw, ph), linewidth=0)
    x_0 = 0.1
    y_0 = 0.1
    width = 0.85
    height = 0.85
    axes_color = '#ffffff'
    # fig.add_axes([x_0, y_0, width,height], axisbg=axes_color)
    u_levels = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    # cs = plt.contour(x, y, u, 10, colors='red', linewidths=0.5)
    cs = plt.contour(x, y, u, 10, colors='red', linewidths=0.5)
    plt.xticks(np.arange(tick_space, x_max, tick_space),fontsize='small')
    plt.yticks(np.arange(tick_space, y_max, tick_space), fontsize='small')
    plt.clabel(cs, inline=1, fmt='%1.2f')
    plt.title('Temperature Contours (i_max = ' + str(i_max) + ')', size=20)
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
    x_nodes=np.arange(0, x_max+hx, hx)
    y_nodes=np.arange(0, y_max+hy, hy)
    x, y = np.meshgrid(x_nodes, y_nodes)
    # ax.plot_wireframe(x, y, u, rstride=1, cstride=1, color='black')
    ax.plot_surface(x, y, u, rstride=1, cstride=1, cmap='jet')
    # ax.contour(x, y, u, rstride=1, cstride=1, cmap='jet')
    ax.contour(x, y, u, cmap='jet')
    plt.show()


def new_u(u_m,i_max,j_max,time_step,hx,hy,dt,k):
    ec, wc, sc, nc, cc, cc_m = coef_cartesian(hx,hy,k,dt)
    max_f_iter = 200
    max_t_iter = 1
    f_toler = 10**(-6)
    t_toler = 10**(-2)
    u_p = u_m.copy()
    u_i = u_m.copy()
    time_iter = 1
    converge_t = False
    while time_iter <= max_t_iter and not converge_t:
        f_iter = 0
        converge_f = False
        while f_iter <= max_f_iter and not converge_f:
            if time_iter % 2 == 0:
                u_p = swpvrt(u_m,u_p,sc,ec,nc,wc,cc,cc_m,i_max,j_max)
            else:
                u_p = swphrz(u_m,u_p,sc,ec,nc,wc,cc,cc_m,i_max,j_max)
            if np.amax(abs(u_p - u_i)) < f_toler:
                converge_f = True
            else:
                u_p = set_bcs(i_max,j_max,u_p,time_step*dt)
                u_i = u_p.copy()
            f_iter += 1
        d_max_t = np.amax(abs(u_p - u_m))
        if d_max_t < t_toler:
            converge_t = True
            print(time_iter, abs(u_p - u_m).max())
        else:
            u_i = u_p.copy()
            u_m = u_p.copy()
        time_iter += 1
    return u_p


def main(t_steps):
    x_max = 1.0
    y_max = 1.0
    i_max = 21
    j_max = 21
    dt = 1.0
    k = 0.5
    hx = x_max/(i_max - 1)
    hy = y_max/(j_max - 1)
    u_m = set_ics(i_max,j_max,hx,hy)
    u_m = set_bcs(i_max,j_max,u_m,0)
    u_p = u_m.copy()
    t_step = 0
    for t_step in range(t_steps):
        u_p = new_u(u_m, i_max, j_max, t_step, hx, hy, dt,k)
        u_p = set_bcs(i_max,j_max,u_p,t_step*dt)
        u_m = u_p.copy()
    plot_contours(u_p, x_max, y_max, i_max, j_max, t_step*dt)
    plot_3d(u_p, x_max, y_max, i_max, j_max)


main(100)
