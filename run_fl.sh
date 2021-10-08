#python train_FL.py config.json -d /hpi/fs00/share/fg-arnrich/datasets/xray_FL/ -o /scratch/joceline.ziegler/bugfix
# python3 train_FL.py config.json -d ./ -df mendeley_xray/ -o ./test --no_gpu

#python train_FL.py config.json -d /mnt/dsets/ChestXrays/CheXpert/ -df ~/netstore/data_files/ -o ~/netstore/densenet_lr0_001_0110

python train_FL.py densenet_config.json -d /mnt/dsets/ -df ~/netstore/data_files/mendeley_files/ -o ~/netstore/densenet_mendeley -m ../fl_densenet_bin_lr0_01/global_7rounds.pth.tar

python train_FL.py resnet_config.json -d /mnt/dsets/ -df ~/netstore/data_files/mendeley_files/ -o ~/netstore/resnet_mendeley -m ../resnet_bin/global_5rounds.pth.tar
