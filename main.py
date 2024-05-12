from fastapi import FastAPI, UploadFile, File
import uvicorn

app = FastAPI()


@app.post("/upload")
async def upload_photo(photo: UploadFile = File(...)):
    with open(f"photos/{photo.filename}", "wb") as f:
        f.write(photo.file.read())
    return {"filename": photo.filename}

if __name__ == "__main__":
    uvicorn.run(app, host="87.242.101.70", port=8000)
