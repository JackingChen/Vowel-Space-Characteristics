#!/usr/bin/env bash


for i in {1..10};do
	python Analyze_DOCKID_syncrony.py --dataset_role ASD_DOCKID --reFilter True
	python Analyze_DOCKID_syncrony.py --dataset_role TD_DOCKID --reFilter True

	python Statistical_tests.py --epoch $i
done
