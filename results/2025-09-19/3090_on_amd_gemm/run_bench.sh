#!/bin/bash

# Script to run 3090_full.py 10 times and save data folders sequentially

echo "Starting 10x benchmark runs..."
echo "================================"

for i in {1..10}; do
    echo "Run $i/10 starting at $(date)"

    # Remove any existing data folder to start fresh
    if [ -d "data" ]; then
        rm -rf data
    fi

    # Run the Python benchmark
    python3 3090_full.py

    # Check if data folder was created
    if [ -d "data" ]; then
        # Move data folder to numbered version
        mv data data_$i
        echo "Run $i completed - data saved to data_$i/"
    else
        echo "WARNING: Run $i failed - no data folder created"
    fi

    # Cool-down pause between runs (except after last run)
    if [ $i -lt 10 ]; then
        echo "Cooling down for 60 seconds..."
        sleep 60
    fi

    echo "--------------------------------"
done

echo "All 10 benchmark runs completed!"
echo "Data folders: data_1/ through data_10/"
ls -la data_*/
