#!/bin/bash

# test run. 20% data should still give at best 75% AUC

# TODO: semaphore
for classifier in quantum classical ; do
	for n_labels in 19 14 8 ; do
		python src/main.py train -c $classifier -n $n_labels -f 0.2 --experiment-dir 'experiments/test' --clobber &
		sleep 60  # wait for GPU memory to fill so that the auto allocation mechanism works
	done
done

wait $(jobs -p)
