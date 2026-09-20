class tqdm:
    def __init__(self, iterable=None, *a, **k):
        self.iterable = iterable
    def __iter__(self):
        return iter(self.iterable or [])
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def update(self, *a, **k): pass
    def close(self): pass
    def set_description(self, *a, **k): pass
