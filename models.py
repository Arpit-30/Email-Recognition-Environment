from pydantic import BaseModel

class EmailObservation(BaseModel):
    email_text: str
    sender: str
    subject: str

class EmailAction(BaseModel):
    action: str  # spam / important / reply

class EmailReward(BaseModel):
    score: float