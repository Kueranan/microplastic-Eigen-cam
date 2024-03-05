# For Dockerfile
FROM python:3.9

RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

WORKDIR /app

CMD ["streamlit", "run", "app_micro.py"]

