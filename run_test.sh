# example bash command for model testing

python test_model.py resnet_config.json -d /mnt/dsets/ -df ~/netstore/data_files/combined_files_less -o ./model_test_results -of resnet_dp_eps3.csv -m ../resnet_dp_eps3/global_3rounds.pth.tar --combine
