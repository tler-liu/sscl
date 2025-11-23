# script to run CIFAR10 with 10% labeled data using hyperparameters = 1
# this script was used to generate results in the hyperparameter tuning section of the paper

python train.py \
	--dataset cifar10 \
	--labeled-fraction 0.1 \
	--lambda-sup 1.0 \
	--lambda-cont 1.0 \
	--tau 1.0 \
	--epochs 256 \
	--image-size 32 \
	--save-dir ./checkpoints \
    --final-model c10_01_ht1.pth \
    --val-metrics c10_01_ht1.json \
    --train-plot c10_01_ht1.png