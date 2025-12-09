from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

# import your function
from modular_code import get_response      #

app = FastAPI()

class InputModel(BaseModel):
    query_dict: Dict[str, Any]

@app.post("/run")
def run_filters(payload: InputModel):
    result = get_response({"query_dict": payload.query_dict})
    return {"result": result}

