import types
import operator


def ignore_future_arguments(parser, ignored):
    call = parser.add_argument

    def add_argument(*args, **kwargs):
        name = args[0]
        if name not in ignored:
            call(name, *args[1:], **kwargs)

    parser.add_argument = add_argument

    return parser


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


class lambda_:
    def __init__(self, literal):
        self.literal = literal
        self.func = eval(literal)
        assert isinstance(self.func, types.LambdaType)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __str__(self):
        return self.literal
