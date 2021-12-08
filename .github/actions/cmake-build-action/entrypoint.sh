#!/bin/sh -l

apt-get update > /dev/null
apt-get install -y libopenmpi-dev
echo "Hello $1"
time=$(date)
echo "::set-output name=time::$time"
