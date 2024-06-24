#!/bin/bash

commands=("./testtlbo" "./testtlbosw" "./testtlboe" "./testtlbob" "./testtlboebsw" "./testsolis")
args=(10 30 50)

for cmd in "${commands[@]}"; do
    for arg in "${args[@]}"; do
        $cmd $arg
    done
done
