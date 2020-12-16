from zouqi.parsing import *


def optional(fn):
    def parse(s):
        if s.lower() in ["null", "none"]:
            return None
        return fn(s)

    return parse
