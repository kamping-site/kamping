#!/bin/sh -l

apt-get update
apt-get install -y
apt install libopenmpi-dev
echo "Hello $1"
time=$(date)
echo "::set-output name=time::$time"
