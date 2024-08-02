#!/bin/bash

set -o pipefail  # to catch ruff errors
# https://github.com/astral-sh/ruff/issues/2743
export CLICOLOR_FORCE=1

if ! ruff check --diff | less -R ; then
	echo "Fix?"
	select yn in "Yes" "No"; do
	    case $yn in
		Yes ) ruff check --fix ; break;;
		No ) break ;;
	    esac
	done
fi


if ! ruff format --diff | less -R ; then
	echo "Format?"
	select yn in "Yes" "No"; do
	    case $yn in
		Yes ) ruff format ; break ;;
		No ) break ;;
	    esac
	done
fi
