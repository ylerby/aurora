from fastapi import FastAPI, UploadFile, File

app = FastAPI()


@app.post("/upload")
async def upload_photo(photo: UploadFile = File(...)):
    with open(f"photos/{photo.filename}", "wb") as f:
        f.write(photo.file.read())
    return {"filename": photo.filename}
