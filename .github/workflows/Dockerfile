FROM ubuntu:jammy

# This Dockerfile is not directly used in the workflows. It is the one used on our self-hosted runner though and only here for documentation.

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y --fix-missing
RUN apt-get upgrade -y

# Build dependencies
RUN apt-get install git gcc clang curl ca-certificates cmake wget gnupg lsb-release build-essential libopenmpi-dev -y

# Doxygen dependencies
RUN apt-get install flex bison graphviz texlive-full -y 

RUN curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh

# install Doxygen (newer versions that available in the package repositories)
ARG DOXYGEN_VERSION=1.10.0
RUN wget https://www.doxygen.nl/files/doxygen-$DOXYGEN_VERSION.src.tar.gz && \
    tar -xzf doxygen-$DOXYGEN_VERSION.src.tar.gz && \
    cd doxygen-$DOXYGEN_VERSION && \
    mkdir build && cd build && cmake -G "Unix Makefiles" .. && \
    make -j && make install

# install cmake
ARG CMAKE_VERSION=3.28.3
RUN wget https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION-linux-x86_64.sh && \
    sh ./cmake-$CMAKE_VERSION-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-$CMAKE_VERSION-linux-x86_64.sh

# Formatting dependencies
RUN apt-get install clang-format cmake-format -y
