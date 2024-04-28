FROM python:3.9-slim-bullseye

WORKDIR /code

ENV FLASK_APP=calc.py

RUN pip install flask

COPY . .

CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
