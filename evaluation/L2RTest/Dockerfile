FROM python:3.7-slim

ADD /surface-distance /surface-distance

RUN groupadd -r evaluator && useradd -m --no-log-init -r -g evaluator evaluator

RUN mkdir -p /opt/evaluation /input /output \
    && chown evaluator:evaluator /opt/evaluation /input /output

USER evaluator
WORKDIR /opt/evaluation

ENV PATH="/home/evaluator/.local/bin:${PATH}"

RUN python -m pip install --user -U pip



COPY --chown=evaluator:evaluator ground-truth /opt/evaluation/ground-truth

COPY --chown=evaluator:evaluator requirements.txt /opt/evaluation/
RUN python -m pip install --user -rrequirements.txt

COPY --chown=evaluator:evaluator evaluation.py /opt/evaluation/
COPY --chown=evaluator:evaluator utils.py /opt/evaluation/

ENTRYPOINT "python" "-m" "evaluation" "-c" "ground-truth/LungCT_Val_evaluation_config.json" "-i" "/input/" "-d" "/opt/evaluation/ground-truth/" "-o" "/output/metrics.json"
