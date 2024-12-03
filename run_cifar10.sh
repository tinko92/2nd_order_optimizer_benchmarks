
for seed in 0 1 2; do
	for lr in '1e-3' '1e-2' '1e-1'; do
		for optim in SophiaG; do #FAdam optim.SGD optim.Adam SophiaG optim.RMSprop; do
			for model in models.swin_v2_s models.vgg19 models.wide_resnet50_2 models.densenet121 models.vgg11 models.vgg13 models.resnet18; do
				for loss in 'nn.CrossEntropyLoss'; do
					python test.py $optim $model $loss CIFAR10 $lr 2 $seed
				done
			done
		done
	done
done
