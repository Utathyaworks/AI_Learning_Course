def broadcast_shape(*shapes):
    n = max((len(s) for s in shapes), default=0)
    out = []
    for i in range(1, n + 1):
        dims = {s[-i] for s in shapes if len(s) >= i} - {1}
        if len(dims) > 1:
            raise ValueError("incompatible shapes")
        out.append(dims.pop() if dims else 1)
    return tuple(reversed(out))
