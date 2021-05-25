from zouqi.typing import *


class _Scheduled(str):
    pass


Scheduled = Annotated[str, Parser(type=_Scheduled)]
