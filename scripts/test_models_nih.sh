#!/bin/sh

experiments_dir=experiments/ISVLSI

run_experiment()
{
    classifier="$1"
    n_labels="$2"
    gpu_num="$3"

    printf "Test nih w/ $classifier-$n_labels on GPU $gpu_num? (y/n)"
    read answer
    if [ "$answer" != "${answer#[Yy]}" ] ; then
        mkdir -p "$experiments_dir"
        for seed in {1..5} ; do
            python src/main.py test \
                -c "$classifier" \
                -n "$n_labels" \
                --experiment-dir "$experiments_dir" \
                -f 1 \
                -g "$gpu_num" \
                --seed "$seed" \
                --threads 1 \
                --clobber \
                # oops, testing requires "clobbering" a directory
                # this line intentionally left blank
        done
    else
        exit 1
    fi

}

for classifier in classical quantum ; do
    for labels in 8 14 19 ; do
        echo yes | run_experiment $classifier $labels 0
    done
done
