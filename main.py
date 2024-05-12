import sys

from fastapi import FastAPI, UploadFile, File
import uvicorn
import asyncio

app = FastAPI()


@app.post("/upload")
async def upload_photo(photo: UploadFile = File(...)):
    with open(f"photos/{photo.filename}", "wb") as f:
        f.write(photo.file.read())
    return {"filename": photo.filename}


async def run_server():
    u_config = uvicorn.Config("main:app", port=8080, log_level="info", reload=True)
    server = uvicorn.Server(u_config)

    await server.serve()


async def main():
    tasks = [
        run_server(),
    ]

    await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
        loop.run_forever()
        loop.close()
    except KeyboardInterrupt:
        sys.exit(0)
