class _Obj:
    def __init__(self, *a, **k):
        object.__setattr__(self, "_d", dict(k))
    def __getattr__(self, item):
        if item.startswith("_"):
            raise AttributeError(item)
        return _Obj
    def __call__(self, *a, **k):
        return _Obj(*a, **k)
    def update(self, *a, **k):
        return self
    def add_trace(self, *a, **k):
        return self


class _Namespace:
    Template = _Obj
    Annotation = _Obj
    Shape = _Obj
    def __getattr__(self, item):
        return _Obj


layout = _Namespace()


def __getattr__(name):
    return _Obj
