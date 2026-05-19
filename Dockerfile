FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install flask mlflow pandas scikit-learn

EXPOSE 5000

CMD ["python", "app.py"]
