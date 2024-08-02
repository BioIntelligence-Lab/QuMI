#!/bin/bash
# https://unix.stackexchange.com/a/216475

# initialize a semaphore with a given number of tokens
open_sem(){
    tmpdir="$(mktemp -d)"
    fifo="${tmpdir}/pipe-$$"
    mkfifo "$fifo"
    exec 3<>"$fifo"
    rm "$fifo"
    rmdir "$tmpdir"
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}

# run the given command asynchronously and pop/push tokens
run_with_lock(){
    local x
    # this read waits until there is something to read
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    # push the return code of the command to the semaphore
    printf '%.3d' $? >&3
    )&
}

