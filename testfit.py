import numpy as np
import scipy
from matplotlib import pyplot as plt

def filtfunc(t, A0, tp, h):
    """'Fifth order semi-gaussian anti-aliasing filter' with parameters from
    1804.02583."""
    tf = np.clip((t - h) / tp, 0, None) # this introduces tiny extra error
                                        # since function is not 0 at 0?
    l = np.outer(np.array([1.19361, 2.38722, 2.5928, 5.18561]), tf)
    sl = np.sin(l)
    cl = np.cos(l)
    e = np.exp(np.outer(np.array([-2.94809, -2.82833, -2.40318]), tf))
    print(l)
    print(sl)
    print(cl)
    print(e)
    return A0 * ( \
            4.31054 * e[0]
            - 2.6202 * e[1] * (cl[0] + cl[0] * cl[1] + sl[0] * sl[1])
            + 0.464924 * e[2] * (cl[2] + cl[2] * cl[3] + sl[2] * sl[3])
            + 0.762456 * e[1] * (sl[0] - cl[1] * sl[0] + cl[0] * sl[1])
            - 0.327684 * e[2] * (sl[2] - cl[3] * sl[2] + cl[2] * sl[3]))

def main():
    x = np.linspace(0, 25, 51)
    y = filtfunc(x, 599, 2.15, 9.5)
    x2 = np.linspace(0, 25, 10001)
    y2 = filtfunc(x2, 599, 2.15, 9.5)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(x, y, s=4)
    ax.plot(x2, y2, linewidth=0.5)
    fig.savefig('testfit2.png')
    plt.close()
    print(x)
    print(y)
    print(x2[np.argmax(y2)])
    print(np.max(y2))

if __name__ == '__main__':
    main()
