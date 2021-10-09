#python test_model.py config.json -d /hpi/fs00/share/fg-arnrich/datasets/xray_FL/ -o /scratch/joceline.ziegler/fl_baseline_mendeley/ -m /scratch/joceline.ziegler/fl_baseline_mendeley/round7_client1/1-epoch_FL.pth.tar --val

#python test_model.py config.json -d ./ -df mendeley_xray/ -o ./test/ --val

#python test_model.py config.json -d /mnt/dsets/ChestXrays/CheXpert/ -df ~/netstore/data_files/ -o ./test --val -m ../resnet_mendeley/global_14rounds.pth.tar

python test_model.py resnet_config.json -d /mnt/dsets/ -df ~/netstore/data_files/mendeley_files/ -o ./example_output -m ../resnet_mendeley/global_14rounds.pth.tar
