#!/bin/bash

# memory profiling script for each path finding algorithm

mkdir -p ./results/memory/bmssp
mkdir -p ./results/memory/dijkstra

 for algo in bmssp dijkstra; do
     planes=4
     sats=2
    
     for scale_idx in {1..7}; do
         constellation_size=$((planes * sats))
         echo "=========================================================="
         echo "Testing $algo with Constellation Size: $constellation_size ($planes planes, $sats sats)"
         echo "=========================================================="
        
         for run_idx in {1..3}; do
            
             # output: ALGO-iteration-number-constellationsize
             output_file="./results/memory/${algo}/${algo}-${run_idx}-${constellation_size}.bin"
             echo "-> Run $run_idx (output: $output_file)"
            
             memray run -o "$output_file" \
                 static_speed_test.py \
                 --solver "$algo" \
                 --iterations 1 \
                 --starting-planes $planes \
                 --starting-sats $sats \
                 --scales 1
         done
        
         planes=$((planes * 2))
         sats=$((sats * 2))
     done
 done


 for file in ./results/memory/bmssp/*.bin; do
     report_file="${file%.bin}.html"
     echo "Generating report for $file -> $report_file"
     memray flamegraph "$file" -o "$report_file"
 done

for file in ./results/memory/dijkstra/*.bin; do
    report_file="${file%.bin}.html"
    echo "Generating report for $file -> $report_file"
    memray flamegraph "$file" -o "$report_file"
done