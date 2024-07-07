FROM python:3.11

WORKDIR site/

COPY . .

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install numpy~=1.26.4
RUN pip install -r requirements.txt

CMD ["flask", "run"]