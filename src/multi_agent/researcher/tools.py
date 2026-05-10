from pydantic import BaseModel, Field

class EvaluatorDecision(BaseModel):
    need_web_search: bool = Field(
        description="True if web search is needed, False if retrieved documents are sufficient"
    )
    reason: str = Field(
        description="Brief explanation of the decision"
    )