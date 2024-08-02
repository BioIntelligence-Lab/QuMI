#!/bin/sh

experiments_dir=experiments/ISVLSI

run_experiment()
{
    classifier="$1"
    n_labels="$2"
    gpu_num="$3"
    
    printf "Train $classifier-$n_labels model on GPU $gpu_num? (y/n)"
    read answer
    if [ "$answer" != "${answer#[Yy]}" ] ; then
        mkdir -p "$experiments_dir"
        for seed in {1..5} ; do
            /usr/bin/time -v -o "${experiments_dir}/time_${classifier}_${n_labels}_${seed}.txt" -- \
                python src/main.py train \
                    -c "$classifier" \
                    -n "$n_labels" \
                    --experiment-dir "$experiments_dir" \
                    -f 1 \
                    -g "$gpu_num" \
                    --seed "$seed" \
                    --threads 4 \
                    # this line intentionally left blank
        done
    else
        exit 1
    fi

}

run_experiment $@
