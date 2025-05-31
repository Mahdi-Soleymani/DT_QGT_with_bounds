#!/bin/bash

LOG_FILE="test_all_dlen.log"
mkdir -p logs
echo "Starting test sweep..." > "$LOG_FILE"

for i in {1..10}; do
    echo "Running with --des_len=$i" | tee -a "$LOG_FILE"

    python atari/test_QGT.py \
        --des_len $i \
        --num_iter 1000 \
        --num_cores 6 \
        --k 10 \
        --checkpoint revived-feather-12.pth \
        --mode DT \
        >> "$LOG_FILE" 2>&1
done

echo "All tests completed!" | tee -a "$LOG_FILE"
