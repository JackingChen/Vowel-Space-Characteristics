#!/usr/bin/env bash


for i in {1..10};do
	python Main_regression.py --suffix $i
done
