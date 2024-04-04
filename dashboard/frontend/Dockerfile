FROM python:3.10

COPY . .

RUN python3.10 -m pip install -r requirements.txt -t .

EXPOSE 8050

CMD ["python", "dashboard.py"]