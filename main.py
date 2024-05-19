import sys
import os
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
import uvicorn
import asyncio
from aurora_cv import get_answer

app = FastAPI()

users = {
    "buyanov": "hui"
}

tests = {}

if "USERS" in os.environ:
    users.update({k: v for k, v in [pair.split(":") for pair in os.environ["USERS"].split(",")]})

@app.post("/upload")
async def upload_photo(test_number: int, photo: UploadFile = File(...)):
    with open(f"photos/{photo.filename}", "wb") as f:
        f.write(photo.file.read())

    answer = get_answer(f"photos/{photo.filename}")

    if answer is None:
        return {"error": "invalid photo format"}

    return {"answer": answer, "test_number": test_number}

@app.post("/auth")
async def auth(request: Request):
    data = await request.json()
    login = data.get("login")
    password = data.get("password")
    test_number = data.get("test", {}).get("number")
    answers = data.get("test", {}).get("answers")

    if login not in users or users[login] != password:
        raise HTTPException(status_code=401, detail="Invalid login or password")

    if test_number not in tests:
        tests[test_number] = {}

    for answer in answers:
        question = answer.get("question")
        answer_text = answer.get("answer")
        tests[test_number][question] = answer_text

    return {"result": "ok"}

async def run_server():
    u_config = uvicorn.Config("main:app", host="0.0.0.0", port=8088, log_level="info", reload=True)
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