FROM python:3.10

WORKDIR /llm-api

COPY ./requirements.txt /llm-api/requirements.txt
ENV FORCE_CMAKE "1"
ENV CMAKE_ARGS "-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./app /llm-api/app
ENV PYTHONPATH "/llm-api"

CMD ["python", "./app/main.py"]
