#!/bin/bash
shopt -s globstar
for i in ../models/**/metrics.csv ; do python plot-learning-progress.py "$i" "$(dirname $i)/progress.png" ; done
