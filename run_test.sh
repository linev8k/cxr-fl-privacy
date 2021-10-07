#python test_model.py config.json -d /hpi/fs00/share/fg-arnrich/datasets/xray_FL/ -o /scratch/joceline.ziegler/fl_baseline_mendeley/ -m /scratch/joceline.ziegler/fl_baseline_mendeley/round7_client1/1-epoch_FL.pth.tar --val

# python3 test_model.py config.json -d ./ -df mendeley_xray/ -o ./test/ --no_gpu --val

python test_model.py config.json -d /mnt/dsets/ChestXrays/CheXpert/ -df ~/netstore/data_files/ -o ./test --val
