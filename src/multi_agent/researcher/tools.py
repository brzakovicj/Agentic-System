from typing import TypedDict
from pydantic import Field

class EvaluatorDecision(TypedDict):
    need_web_search: bool