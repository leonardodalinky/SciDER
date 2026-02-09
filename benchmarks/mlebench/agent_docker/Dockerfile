FROM mlebench-env

# where to put submission.csv, will be extracted
ARG SUBMISSION_DIR
ENV SUBMISSION_DIR=${SUBMISSION_DIR}
# where to put any logs, will be extracted
ARG LOGS_DIR
ENV LOGS_DIR=${LOGS_DIR}
# where to put any code, will be extracted
ARG CODE_DIR
ENV CODE_DIR=${CODE_DIR}
# where to put any other agent-specific files, will not be necessarily extracted
ARG AGENT_DIR
ENV AGENT_DIR=${AGENT_DIR}

RUN mkdir ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR}

ARG CONDA_ENV_NAME=agent

# Requirements for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# put remaining necessary agent files in the expected location
COPY additional_notes.txt /home/additional_notes.txt
COPY start.sh ${AGENT_DIR}

# copy just `agent_docker/requirements.txt`, so that we can cache conda separately from the agent files
COPY benchmarks/mlebench/agent_docker/requirements.txt ${AGENT_DIR}/requirements.txt

# create conda environment and install the requirements to it
RUN conda run -n ${CONDA_ENV_NAME} pip install -r ${AGENT_DIR}/requirements.txt && \
    conda clean -afy

COPY scievo/ ${AGENT_DIR}/SciEvoBench/scievo/
COPY bench_workflows/ ${AGENT_DIR}/SciEvoBench/bench_workflows/
COPY requirements.txt ${AGENT_DIR}/SciEvoBench/requirements.txt
RUN conda run -n ${CONDA_ENV_NAME} pip install -r ${AGENT_DIR}/SciEvoBench/requirements.txt && \
    conda clean -afy

# set working directory to SciEvoBench
WORKDIR ${AGENT_DIR}/SciEvoBench
