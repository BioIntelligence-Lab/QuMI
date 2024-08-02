#!/bin/bash

# https://unix.stackexchange.com/a/436713

enqueue() {
    (
        echo "Running $@"
	"$@"
    ) &

    # allow to execute up to $N jobs in parallel
    if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
        # now there are $N jobs already running, so wait here for any job
        # to be finished so there is a place to start next one.
        wait -n
    fi
}
#task() { 
#	sleep $(( RANDOM % 3 + 10)) 
#	echo $1
#}
#N=4
#for i in {a..z} ; do
#	enqueue task $i
#done
