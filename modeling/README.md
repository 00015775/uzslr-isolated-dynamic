## Model Development

Model architecture and training are defined here at `modeling/`.
Training is performed in the cloud using the same codebase,
with configuration files controlling experiments.

### Conda environment

```shell
# Create and activate the environment
conda env create -f environment-uzslr-signs.yml
conda activate uzslr-signs
```

> [!IMPORTANT]
> This environment, `uzslr-signs`, is currently configured to support all preprocessing steps.
> This preprocessing will be used for model training and evaluation once finalized.
> The [`environment-uzslr-signs.yml`](../environment-uzslr-signs.yml) will be expanded as new dependencies are added during the model development phase.


### Initial Template
<pre>
.
└──modeling/
    ├── README.md
    ├── configs/
    │   ├── base.yaml
    │   ├── experiment_001.yaml
    │   └── experiment_002.yaml
    ├── notebooks/
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
