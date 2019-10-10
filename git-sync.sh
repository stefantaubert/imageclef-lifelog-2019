#!/bin/bash

if [ "$1" == "" ]; then
	echo "Please add a commit message!"
else
	echo $1
	git status
	git add .
	git commit -am "$1"
	git pull
	git push
fi
