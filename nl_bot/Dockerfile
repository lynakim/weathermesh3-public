FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir -r requirements.txt

COPY lambda_function.py query_agent.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]
