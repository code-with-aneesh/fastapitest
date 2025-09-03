# main.py
from fastapi import FastAPI

app = FastAPI()

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

# Example endpoint with parameters
@app.get("/greet/{name}")
def greet(name: str):
    return {"message": f"Hello, {name}!"}
