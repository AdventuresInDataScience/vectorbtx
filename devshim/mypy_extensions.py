def VarArg(t=None): return t
def KwArg(t=None): return t
def Arg(t=None, name=None): return t
def DefaultArg(t=None, name=None): return t
def NamedArg(t=None, name=None): return t
def DefaultNamedArg(t=None, name=None): return t
class TypedDict(dict): pass
def trait(c): return c
def mypyc_attr(*a, **k):
    def deco(c): return c
    return deco
