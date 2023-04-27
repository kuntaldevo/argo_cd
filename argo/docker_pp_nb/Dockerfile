# ARGO docker container

# Install default environment
FROM dockerhub.paypalcorp.com/core-data-platform/ppmagics-spark230:latest
FROM python:3.7
ENV LANG C.UTF-8
RUN apt-get update && apt-get install -y python python-dev python-pip virtualenv libssl-dev libpq-dev git build-essential libfontconfig1 libfontconfig1-dev
RUN pip install setuptools pip --upgrade --force-reinstall

# Copy ARGO directory
RUN mkdir /argo
COPY ./ /argo

# Set up working directory
WORKDIR /
RUN mkdir /workdir/
RUN mkdir /workdir/examples

# Copy over files from argo folder 
COPY ./examples /workdir/examples
COPY ./argo/correlation_reduction/examples /workdir/examples/subpackage_examples/correlation_reduction
COPY ./argo/read_data/examples /workdir/examples/subpackage_examples/read_data
COPY ./argo/rule_application/examples /workdir/examples/subpackage_examples/rule_application
COPY ./argo/rule_filtering/examples /workdir/examples/subpackage_examples/rule_filtering
COPY ./argo/rule_generation/examples /workdir/examples/subpackage_examples/rule_generation
COPY ./argo/rule_optimisation/examples /workdir/examples/subpackage_examples/rule_optimisation
COPY ./argo/rule_scoring/examples /workdir/examples/subpackage_examples/rule_scoring
COPY ./argo/rules/examples /workdir/examples/subpackage_examples/rules
COPY ./argo/simility_apis/examples /workdir/examples/subpackage_examples/simility_apis
COPY ./argo/simility_requests/examples /workdir/examples/subpackage_examples/simility_requests
COPY ./argo/system_config_generation/examples /workdir/examples/subpackage_examples/system_config_generation

# Install sub-packages
RUN pip install /argo/argo/argo_utils/.
RUN pip install /argo/argo/correlation_reduction/.
RUN pip install /argo/argo/read_data/.
RUN pip install /argo/argo/rule_application/.
RUN pip install /argo/argo/rule_filtering/.
RUN pip install /argo/argo/rule_generation/.
RUN pip install /argo/argo/rule_optimisation/.
RUN pip install /argo/argo/rule_scoring/.
RUN pip install /argo/argo/rules/.
RUN pip install /argo/argo/simility_apis/.
RUN pip install /argo/argo/simility_requests/.
RUN pip install /argo/argo/system_config_generation/.
# Install ARGO
RUN pip install /argo/.

# Install jupyter packages
RUN pip install jupyterlab
RUN pip install ipykernel

# ENTRYPOINT jupyter lab --ip=0.0.0.0 --port=8080 --allow-root --no-browser --notebook-dir /workdir/ --NotebookApp.token=''