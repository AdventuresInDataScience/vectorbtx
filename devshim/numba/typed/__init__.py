class List(list):
    @classmethod
    def empty_list(cls, *a, **k):
        return cls()


class Dict(dict):
    @classmethod
    def empty(cls, *a, **k):
        return cls()
