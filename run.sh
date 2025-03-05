#!/usr/bin/env bash
echo '' > output.txt

# Check the number of arguments
if [ "$#" -eq 0 ]; then
    echo "No arguments provided. using default arguments to run"
    make run 
else
    echo "Number of arguments: $#"
    ./bin/imageRotationNPP $* 
fi
