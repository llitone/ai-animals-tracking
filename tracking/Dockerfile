FROM python:3.11

WORKDIR api/

COPY . .

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install -r requirements.txt

CMD ["flask", "run"]