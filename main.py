import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

# Load CSV file
csv_file = "products.csv"
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
else:
    df = pd.DataFrame(columns=["Title", "cleanDescription", "Image Src", "Product URL"])

@app.get("/")
def home():
    return {"message": "API is running!"}

@app.get("/recommend")
def recommend(query: str):
    if df.empty:
        raise HTTPException(status_code=500, detail="No data available")
    
    # Filter matching products
    results = df[df["Title"].str.contains(query, case=False, na=False)].to_dict(orient="records")
    
    return {"recommendations": results[:5]}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # Ensure Render assigns a port
    uvicorn.run(app, host="0.0.0.0", port=port)
