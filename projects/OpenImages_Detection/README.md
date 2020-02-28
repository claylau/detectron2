# Train Process

```sh
ln -s /path/to/pen/1400-79-186 datasets 
```

get like this directories tree:

├── base_rcnn_c4.yaml
├── datasets -> /home/lauz/datasets/OpenImagesDataSet/Pen/1400-79-186/
├── data.py
└── train_net.py

then run `python3 train_net.py --configs-file base_rcnn_c4.yaml`
