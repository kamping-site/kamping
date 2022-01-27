FROM ubuntu:21.10

# This Dockerfile is not directly used in the workflows. It is the one used on our self-hosted runner though and only here for documentation.

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y --fix-missing
RUN apt-get upgrade
RUN apt-get install git gcc clang curl ca-certificates cmake wget gnupg lsb-release doxygen graphviz build-essential libopenmpi-dev texlive-full -y
RUN curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh

# install cmake
ARG CMAKE_VERSION=3.21.0
RUN wget https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION-linux-x86_64.sh && \
    sh ./cmake-$CMAKE_VERSION-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-$CMAKE_VERSION-linux-x86_64.sh
