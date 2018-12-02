N_KEYPOINTS=$1
python scripts/test.py --experiment-name aflw-"$1"pts-finetune --train-dataset aflw --test-dataset aflw