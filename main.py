from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np


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

app = FastAPI()

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