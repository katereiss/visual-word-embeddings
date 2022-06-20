# syntax=docker/dockerfile:1
FROM python:3.8-slim-buster
EXPOSE 8080
WORKDIR /streamlit
COPY requirements.txt streamlit/requirements.txt
RUN pip3 install -r streamlit/requirements.txt
COPY . /streamlit
CMD [ "python3", "-m" , "streamlit", "run", "--host=0.0.0.0"]
