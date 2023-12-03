import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from utils import *
import uvicorn


with open('pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)


app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    print(item)
    item_df = pd.DataFrame([item.model_dump()])
    prediction = np.round(np.exp(pipeline.predict(item_df)), 2)
    return prediction


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    items_df = pd.DataFrame([item.dict() for item in items])
    predictions = np.round(np.exp(pipeline.predict(items_df)), 2)
    return predictions



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)