# The triage assistant as a web service (run: uvicorn triage_service:app --port 8000).
from fastapi import FastAPI
from pydantic import BaseModel, Field
from triage import Triage

app = FastAPI(title="Ticket triage assistant")
triage = Triage()                                          # load all the parts once


class Ticket(BaseModel):
    text: str = Field(min_length=3, max_length=2000)       # validated automatically by FastAPI


@app.post("/triage")
def triage_ticket(ticket: Ticket):
    result = triage([ticket.text])[0]
    result["model"] = "triage-ensemble-v1"
    return result
