import numpy as np

def f(t, A0, tp, h, b):
    """'Fifth order semi-gaussian anti-aliasing filter' with parameters from
    1804.02583."""
    tf = np.clip((t - h) / tp, 0, None)
    l = np.outer(np.array([1.19361, 2.38722, 2.5928, 5.18561]), tf)
    sl = np.sin(l)
    cl = np.cos(l)
    e = np.exp(np.outer(np.array([-2.94809, -2.82833, -2.40318]), tf))
    return b + A0 * ( \
            4.31054 * e[0]
            - 2.6202 * e[1] * (cl[0] + cl[0] * cl[1] + sl[0] * sl[1])
            + 0.464924 * e[2] * (cl[2] + cl[2] * cl[3] + sl[2] * sl[3])
            + 0.762456 * e[1] * (sl[0] - cl[1] * sl[0] + cl[0] * sl[1])
            - 0.327684 * e[2] * (sl[2] - cl[3] * sl[2] + cl[2] * sl[3]))
