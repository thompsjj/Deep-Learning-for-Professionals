def estimate_gradients(f, wrt, at, h=1e-5):
    d = {}
    for variable in wrt:
        dfdw  = f(**{param: value+h if param == variable else value for param, value in at.items()})
        dfdw_ = f(**{param: value-h if param == variable else value for param, value in at.items()})
        d[variable] = (dfdw-dfdw_) / (2*h)
    return d
