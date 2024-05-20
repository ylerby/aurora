import sys
import os
from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Query, status
import uvicorn
import asyncio
from aurora_cv import get_answer

app = FastAPI()

users: dict = {
    "teacher_1": "pass_1",
}

tests: dict = {}

if "USERS" in os.environ:
    users.update({k: v for k, v in [pair.split(":") for pair in os.environ["USERS"].split(",")]})


@app.post("/upload")
async def upload_photo(test_number: int = Query(None), photo: UploadFile = File(...)):
    """
    :param test_number: query param, integer
    :param photo: uploaded photo
    :return:
        Test results info:
            1) Answers: (question, your answer, correct answer
            2) Total correct answers
            3) Total incorrect answers
        Example:
            {
                "answers": [
                    {
                        "question": "4",
                        "answer": "D",
                        "correct-answer": "K"
                    },
                    {
                        "question": "2",
                        "answer": "B",
                        "correct-answer": "B"
                    }
            ],
            "total-correct-answers": 4,
            "total-incorrect-answers": 5,
            "test_number": 1
        }
    """
    if test_number is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid query parameter")

    if test_number not in tests:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="test number not found")

    photos_dir = "photos"
    if not os.path.exists(photos_dir):
        os.makedirs(photos_dir)

    with open(os.path.join(photos_dir, photo.filename), "wb") as f:
        f.write(photo.file.read())

    correct_answers = [{"question": str(i), "correct_answer": answer} for i, answer in tests[test_number].items()]
    try:
        answer = get_answer(os.path.join(photos_dir, photo.filename), correct_answers)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"image processing error: {e}")

    if answer is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid photo format")

    if "answer" not in answer or "total-correct-answers" not in answer or "total-incorrect-answers" not in answer:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid photo format")

    result = {
        "answers": answer["answer"],
        "total-correct-answers": answer["total-correct-answers"],
        "total-incorrect-answers": answer["total-incorrect-answers"],
        "test_number": test_number
    }

    return result


@app.post("/auth")
async def auth(request: Request):
    """
     Authenticate a user, send correct test answers.

    :param request:
        JSON with:
            1) login
            2) password
            3) test number
            4) answers
        Example:

            {
                "login": "login",
                "password": "123",
                "number": 1,
                "test": [
                        {"question": 1, "correct_answer": "A"},
                        {"question": 7, "correct_answer": "A"}
                ]
            }


    :returns:
        Processing status (ok/error).

        Example:
            {
                "result": "ok"
            }
    """

    data = await request.json()
    login = data.get("login")
    password = data.get("password")
    test_number = int(data.get("number"))
    answers = data.get("test")

    if login not in users or users[login] != password:
        raise HTTPException(status_code=401, detail="invalid login or password")

    if test_number not in tests:
        tests[test_number] = {}

    for answer in answers:
        question = answer.get("question")
        correct_answer = answer.get("correct_answer")
        tests[test_number][question] = correct_answer

    return {"result": "ok"}


async def run_server():
    """
    Server initializing, running
    """
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
