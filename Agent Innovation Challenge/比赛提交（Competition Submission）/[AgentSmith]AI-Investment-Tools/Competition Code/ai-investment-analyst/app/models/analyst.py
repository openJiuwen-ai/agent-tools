from pydantic import BaseModel


class LineItem(BaseModel):
    ticker: str
    report_period: str
    period: str
    currency: str

    # Allow additional fields dynamically
    model_config = {"extra": "allow"}


class ValueInvestorPromptResponse(BaseModel):
    prompt: str  # The prompt ready for LLM
    facts: dict  # The analysis facts
