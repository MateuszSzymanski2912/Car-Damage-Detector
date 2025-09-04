from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from resnet import process_image
from chatgpt import summarize_with_chatgpt


app = FastAPI(title="Car Damage Detection API")

@app.get("/")
def root():
    return {"message": "Car Damage Detection API is running"}

@app.post("/predict")
async def predict(files: list[UploadFile] = File(...), heatmap: bool = False, summary: bool = False, return_original: bool = False):
    results = []
    for file in files:
        image_bytes = await file.read()
        try:
            result = process_image(image_bytes, generate_heatmap=heatmap, return_original=return_original)
            result["filename"] = file.filename
            results.append(result)
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
    
    chatgpt_summary = None
    if summary:
        chatgpt_summary = summarize_with_chatgpt(results)
    
    print("ChatGPT Summary:", chatgpt_summary)
    
    return JSONResponse({
        "results": results,
        "summary": chatgpt_summary
    })

