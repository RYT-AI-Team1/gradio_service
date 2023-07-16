FROM python:3.7.16
MAINTAINER namhai1810k2003@gmail.com
WORKDIR /app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN set -xe && apt-get -yqq update && apt-get -yqq install python3-pip && pip3 install --upgrade pip
COPY . /app
RUN pip install -r requirements.txt

# EXPOSE 5201

CMD ["python", "./server/app.py"] 