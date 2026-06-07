from pydantic import BaseModel


class Claim(BaseModel):
    claim_text: str


class ExtractionResult(BaseModel):
    abstract_id: str
    claims: list[Claim]
