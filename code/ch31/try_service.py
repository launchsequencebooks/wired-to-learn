# Send a few new tickets (written for this test, not from the dataset) through the service.
from fastapi.testclient import TestClient
from triage_service import app

client = TestClient(app)
tickets = ["I bought something in a shop yesterday and was charged two times for it",
           "my phone got stolen at the train station and the app is on it",
           "How long does it take for money I send to Germany to arrive?",
           "hi, can you help me"]
for text in tickets:
    r = client.post("/triage", json={"text": text}).json()
    print(f"{text!r}\n  -> {r['team']} / {r['intent']} (confidence {r['team_confidence']}, {r['route']})")
    print(f"     similar: {r['similar_past_tickets'][0]!r}")
print("empty ticket ->", client.post("/triage", json={"text": ""}).status_code)
