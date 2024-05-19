import sys
import os
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
import uvicorn
import asyncio
from aurora_cv import get_answer

app = FastAPI()

users = {
    "buyanov": "hui",
    "login": "123"
}

tests = {}

if "USERS" in os.environ:
    users.update({k: v for k, v in [pair.split(":") for pair in os.environ["USERS"].split(",")]})


@app.post("/upload")
async def upload_photo(test_number: int, photo: UploadFile = File(...)):
    if test_number not in tests:
        return {"error": "Test number not found"}

    photos_dir = "photos"
    if not os.path.exists(photos_dir):
        os.makedirs(photos_dir)

    with open(os.path.join(photos_dir, photo.filename), "wb") as f:
        f.write(photo.file.read())

    correct_answers = [{"question": str(i), "correct_answer": answer} for i, answer in tests[test_number].items()]
    answer = get_answer(os.path.join(photos_dir, photo.filename), correct_answers)

    if answer is None:
        return {"error": "invalid photo format"}

    result = {
        "answers": answer["answer"],
        "total-correct-answers": answer["total-correct-answers"],
        "total-incorrect-answers": answer["total-incorrect-answers"],
        "test_number": test_number
    }

    return result


@app.post("/auth")
async def auth(request: Request):
    data = await request.json()
    login = data.get("login")
    password = data.get("password")
    test_number = int(data.get("number"))
    answers = data.get("test")

    if login not in users or users[login] != password:
        raise HTTPException(status_code=401, detail="Invalid login or password")

    if test_number not in tests:
        tests[test_number] = {}

    for answer in answers:
        question = answer.get("question")
        correct_answer = answer.get("correct_answer")
        tests[test_number][question] = correct_answer

    return {"result": "ok", "test_data": tests[test_number]}


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
