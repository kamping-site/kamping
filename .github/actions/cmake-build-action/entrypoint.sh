#!/bin/sh -l

apt update -y
echo "Hello $1"
time=$(date)
echo "::set-output name=time::$time"
