FROM python:3.12-bullseye
WORKDIR /app
COPY . .
RUN pip install .
ENTRYPOINT ["/bin/bash", "train"]
