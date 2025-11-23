# script to run CIFAR10 with 10% labeled data using l_sup=1.0, l_cont=1.0, tau=0.2
# this script was used to generate results in the hyperparameter tuning section of the paper

python train.py \
	--dataset cifar10 \
	--labeled-fraction 0.1 \
	--lambda-sup 1.0 \
	--lambda-cont 1.0 \
	--tau 0.2 \
	--epochs 256 \
	--image-size 32 \
	--save-dir ./checkpoints \
    --final-model c10_01_ht2.pth \
    --val-metrics c10_01_ht2.json \
    --train-plot c10_01_ht2.png