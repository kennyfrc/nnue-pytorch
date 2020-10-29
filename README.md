# Setup
```
python3 -m venv env
source env/bin/activate
pip install pytorch-lightning
pip install python-chess
pip install tensorboard
```

# Run

```
source env/bin/activate
python3 train.py
```

# Logging

```
tensorboard --logdir=logs
```

# Thanks

* syzygy - http://www.talkchess.com/forum3/viewtopic.php?f=7&t=75506
* https://github.com/connormcmonigle/seer-nnue
* https://github.com/DanielUranga/TensorFlowNNUE
* https://hxim.github.io/Stockfish-Evaluation-Guide/