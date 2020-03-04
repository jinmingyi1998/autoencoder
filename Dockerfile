FROM python:3.6.10

COPY . /app
RUN pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
CMD python3 -u /app/app.py