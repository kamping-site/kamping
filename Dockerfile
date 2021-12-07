FROM ubuntu:21.10

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get install git gcc clang curl ca-certificates cmake wget gnupg lsb-release doxygen graphviz -y --no-install-recommends
RUN curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh
