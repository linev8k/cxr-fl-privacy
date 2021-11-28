# example bash command for private training

python train_FL.py resnet_config.json -dp resnet_dp_config.json -d /mnt/dsets/ -df ~/netstore/data_files/combined_files_less/ -o ~/netstore/resnet_dp_eps3 --combine
