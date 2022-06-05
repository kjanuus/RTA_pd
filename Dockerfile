FROM python:3.8
WORKDIR /app
COPY . .
RUN pip install Flask Jinja2
RUN pip3 install numpy
RUN pip freeze
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]