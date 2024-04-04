FROM public.ecr.aws/lambda/python:3.10

COPY requirements.txt ./

RUN python3.10 -m pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu -t .

COPY ST_MGCN_best_model.pkl /opt/ml/ST_MGCN_best_model.pkl

COPY adj_matrices2.pkl /opt/ml/adj_matrices2.pkl

COPY carreaux_disconnect_function.pkl /opt/ml/carreaux_disconnect_function.pkl

COPY app.py ./

COPY preprocessor.py ./

COPY stmgcn.py ./

CMD ["app.lambda_handler"]
