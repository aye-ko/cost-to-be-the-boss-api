import os
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import tempfile
from paddleocr import PaddleOCR


app = FastAPI()

ocr_engine = PaddleOCR(use_textline_orientation=True, lang='en')


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cost-to-be-the-boss-vue.vercel.app",
                "http://localhost:3000",
                ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecipeInput(BaseModel):
    name: str
    servingPrice: float
    servingsPerBatch: int
    batchesPerMonth:float
    
    
class SimulationRequest(BaseModel):
    recipes: list[RecipeInput]
    revenueGoal: float
    
class SimulationResponse(BaseModel):
    successProbability: float
    bestCase: float
    worstCase: float
    realisticRange: list[float]

@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/simulate")
def simulate(request: SimulationRequest) -> SimulationResponse:
    numSimulations = 10000
    totalRevenues = []
    
    for _ in range(numSimulations):
        runRevenue = 0
        for recipe in request.recipes:
            waste = np.random.normal(loc=0.30, scale=0.07)  # Simulate waste percentage
            waste = np.clip(waste, 0, 1)  # Ensure waste is between 0 and 100%
            recipeRevenue= (recipe.batchesPerMonth
                             * recipe.servingsPerBatch
                            * recipe.servingPrice
                            * (1 - waste))
            runRevenue += recipeRevenue
        totalRevenues.append(runRevenue)
        
    totalRevenues = np.array(totalRevenues)
    successProbability = np.mean(totalRevenues >= request.revenueGoal)
    bestCase = np.percentile(totalRevenues, 95)
    worstCase = np.percentile(totalRevenues, 5)
    realisticRange = [np.percentile(totalRevenues, 25), np.percentile(totalRevenues, 75)]
    
    return SimulationResponse(
        successProbability = float(successProbability),
        bestCase = float(bestCase),
        worstCase = float(worstCase),
        realisticRange=[float(realisticRange[0]), float(realisticRange[1])]
    )
    

@app.post("/ocr")

def ocr(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        tmp.write(file.file.read())   # the uploads byes onto disk
        path =tmp.name  # remember where it is
    try:
        result = ocr_engine.predict(path)
        lines = []
        for res in result:
            lines.extend(res['rec_texts'])
        return {"lines": lines}
    finally:
        os.remove(path)
