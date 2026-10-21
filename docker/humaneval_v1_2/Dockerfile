FROM python:3.10-slim
COPY third_party/human-eval /opt/human-eval
RUN python -m pip install --no-cache-dir /opt/human-eval
WORKDIR /work
ENTRYPOINT ["evaluate_functional_correctness"]

