def __getattr__(name):
    def f(*a, **k):
        raise NotImplementedError
    return f
