def listof(type_):
    return lambda s: list(map(type_, s.split(",")))


def str2bool(v):
    assert v.lower() in ["true", "false"]
    return v.lower() == "true"


def optional(type_):
    return lambda s: None if s.lower() in ["null", "none"] else type_(s)


def union(*types):
    def loader(s):
        for typ in types:
            try:
                return typ(s)
            except:
                pass
        raise TypeError(s)

    return loader
