#python train_FL.py config.json -d /hpi/fs00/share/fg-arnrich/datasets/xray_FL/ -o /scratch/joceline.ziegler/bugfix
# python3 train_FL.py config.json -d ./ -df mendeley_xray/ -o ./test --no_gpu

#python train_FL.py config.json -d /mnt/dsets/ChestXrays/CheXpert/ -df ~/netstore/data_files/ -o ~/netstore/densenet_lr0_001_0110

python train_FL.py densenet_config.json -d /mnt/dsets/ -df ~/netstore/data_files/combined_files/ -o ~/netstore/densenet_combined --combine

python train_FL.py resnet_config.json -d /mnt/dsets/ -df ~/netstore/data_files/combined_files/ -o ~/netstore/resnet_combined --combine
