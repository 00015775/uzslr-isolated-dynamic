## Model Development

Model architecture and training are defined here at `modeling/`.
Training is performed in the cloud using the same codebase,
with configuration files controlling experiments.

### Initial Template
<pre>
.
└──modeling/
    ├── README.md
    ├── configs/
    │   ├── base.yaml
    │   ├── experiment_001.yaml
    │   └── experiment_002.yaml
    ├── datasets/
    │   └── uzslr_dataset.py
    ├── models/
    │   ├── __init__.py
    │   ├── cnn.py
    │   ├── transformer.py
    │   └── rnn.py
    ├── training/
    │   ├── train.py
    │   ├── loss.py
    │   └── metrics.py
    ├── evaluation/
    │   └── evaluate.py
    └── utils/
        ├── logging.py
        ├── checkpoints.py
        └── seed.py
</pre>
