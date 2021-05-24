from zouqi.typing import *


class _Scheduled(str):
    pass


Scheduled = Annotated[str, dict(type=_Scheduled)]
