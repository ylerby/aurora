from fastapi import FastAPI, UploadFile, File
import uvicorn

app = FastAPI()


@app.post("/upload")
async def upload_photo(photo: UploadFile = File(...)):
    with open(f"photos/{photo.filename}", "wb") as f:
        f.write(photo.file.read())
    return {"filename": photo.filename}

if __name__ == "__main__":
    u_config = uvicorn.Config("main:app", port=8080, log_level="info", reload=True)
    server = uvicorn.Server(u_config)
    server.serve()
