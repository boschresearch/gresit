#!/bin/bash

export CASTLE_BACKEND='pytorch'

NSAMPLES=("1000" "2000" "5000")
NNODES=("10" "15")
GS=("2" "5")

for nodes in "${NNODES[@]}"
do
        for samples in "${NSAMPLES[@]}"
        do
                for gs in "${GS[@]}"
                do
                        echo "Run $samples, $gs, $nodes"
                        python3 simulation.py \
                                --n_samples="$samples" \
                                --n_runs=20 \
                                --n_nodes="$nodes" \
                                --group_size="$gs"
                done
        done
done
