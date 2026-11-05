# Call the recommender service in-process.
from fastapi.testclient import TestClient
from rec_service import app

client = TestClient(app)
recs = client.get("/users/2/recommendations", params={"k": 3}).json()
print("GET /users/2/recommendations?k=3 ->", [r["movie"] for r in recs])
sim = client.get("/movies/49/similar", params={"k": 3}).json()          # movie 49: Star Wars (1977)
print("GET /movies/49/similar?k=3 ->", [(s["movie"], s["similarity"]) for s in sim])
print("GET /users/5000/recommendations ->", client.get("/users/5000/recommendations").status_code)
