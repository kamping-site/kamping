# Container image that runs your code
FROM ubuntu:latest

ENV TZ=Europe/Berlin
RUN DEBIAN_FRONTEND="noninteractive" apt update

# Copies your code file from your action repository to the filesystem path `/` of the container
COPY entrypoint.sh /entrypoint.sh

# Code file to execute when the docker container starts up (`entrypoint.sh`)
ENTRYPOINT ["/entrypoint.sh"]