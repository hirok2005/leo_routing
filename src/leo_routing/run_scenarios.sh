#!/bin/bash

duration=250
iterations=10


recovery_rate=0.05

failure_params=("10 0.0055" "20 0.0125" "30 0.0214" "40 0.0333" "50 0.0500" "60 0.0750" "70 0.1166" "80 0.2000" "90 0.4500")

bandwidth=15000
message_size=1500
max_queue_size=100

congestion_params=(
    "10 1.0" 
    "20 0.5" 
    "30 0.3333" 
    "40 0.25" 
    "50 0.2" 
    "60 0.1666" 
    "70 0.1428" 
    "80 0.125" 
    "90 0.1111"
)

for param in "${failure_params[@]}"; do
    read -r percentile failure_rate <<< "$param"
    
    echo "Running failure scenario: Percentile $percentile, Failure Rate $failure_rate"
    python src/leo_routing/protocol_comparison.py --scenario-name failure_${percentile} --duration $duration --iterations $iterations --pairs all --failure-rate $failure_rate --recovery-rate $recovery_rate --data-size $message_size --interval 5 -l --results-dir "results/link_failure/"
done



 for param in "${congestion_params[@]}"; do
     read -r percentile message_interval <<< "$param"
    
     echo "Running congestion scenario: Percentile $percentile, Message Interval $message_interval"
     python src/leo_routing/protocol_comparison.py --scenario-name congestion_${percentile} --duration $duration --iterations $iterations --pairs all --data-size $message_size --interval $message_interval --model-bandwidth --bandwidth $bandwidth --max-queue-size $max_queue_size --poisson-traffic -l --results-dir "results/congestion/"
 done