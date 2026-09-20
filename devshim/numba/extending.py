def overload(*a, **k):
    def deco(f):
        return f
    return deco


def register_jitable(*a, **k):
    if len(a) == 1 and callable(a[0]) and not k:
        return a[0]
    def deco(f):
        return f
    return deco


def intrinsic(*a, **k):
    def deco(f):
        return f
    return deco


def overload_method(*a, **k):
    def deco(f):
        return f
    return deco
