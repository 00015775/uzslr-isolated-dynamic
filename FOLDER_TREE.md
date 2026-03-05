## Folder Tree

This document is intended for the repository owner to correctly place the dataset, as it is not included in the GitHub repository. Proper placement ensures that the model can be trained on other machines without issues. Note that the folder structure shown here may not exactly match the current project layout, as updates and new features may change it. The main purpose is to indicate where the dataset should go, since manually modifying dataset paths in notebooks or scripts can be time-consuming and error-prone.

### `tree -L 3 .`
```
tree -L 3 .
.
├── conda_deps_from_history.yml
├── CONDA_INFO.md
├── data
│   ├── test
│   │   ├── assalomu_alaykum
│   │   ├── bahor
│   │   ├── birga
│   │   ├── bo'sh
│   │   ├── bosh_kiyim
│   │   ├── boshlanishi
│   │   ├── bozor
│   │   ├── eshik
│   │   ├── futbol
│   │   ├── iltimos
│   │   ├── internet
│   │   ├── javob
│   │   ├── jismoniy_tarbiya
│   │   ├── karam
│   │   ├── kartoshka
│   │   ├── kichik
│   │   ├── kitob
│   │   ├── ko'prik
│   │   ├── likopcha
│   │   ├── maktab
│   │   ├── mehmonxona
│   │   ├── mehribon
│   │   ├── metro
│   │   ├── musiqa
│   │   ├── o'simlik_yog'i
│   │   ├── o'ynash
│   │   ├── ochish
│   │   ├── ot
│   │   ├── ovqat_tayyorlash
│   │   ├── oxiri
│   │   ├── poezd
│   │   ├── pomidor
│   │   ├── qidirish
│   │   ├── qish
│   │   ├── qo'ziqorin
│   │   ├── qor
│   │   ├── qorong'i
│   │   ├── quyon
│   │   ├── restoran
│   │   ├── sariyog'
│   │   ├── shokolad
│   │   ├── sovun
│   │   ├── stakan
│   │   ├── televizor
│   │   ├── tosh
│   │   ├── toza
│   │   ├── turish
│   │   ├── yomg'ir
│   │   ├── yopish
│   │   └── yordam_berish
│   ├── train
│   │   ├── assalomu_alaykum
│   │   ├── bahor
│   │   ├── birga
│   │   ├── bo'sh
│   │   ├── bosh_kiyim
│   │   ├── boshlanishi
│   │   ├── bozor
│   │   ├── eshik
│   │   ├── futbol
│   │   ├── iltimos
│   │   ├── internet
│   │   ├── javob
│   │   ├── jismoniy_tarbiya
│   │   ├── karam
│   │   ├── kartoshka
│   │   ├── kichik
│   │   ├── kitob
│   │   ├── ko'prik
│   │   ├── likopcha
│   │   ├── maktab
│   │   ├── mehmonxona
│   │   ├── mehribon
│   │   ├── metro
│   │   ├── musiqa
│   │   ├── o'simlik_yog'i
│   │   ├── o'ynash
│   │   ├── ochish
│   │   ├── ot
│   │   ├── ovqat_tayyorlash
│   │   ├── oxiri
│   │   ├── poezd
│   │   ├── pomidor
│   │   ├── qidirish
│   │   ├── qish
│   │   ├── qo'ziqorin
│   │   ├── qor
│   │   ├── qorong'i
│   │   ├── quyon
│   │   ├── restoran
│   │   ├── sariyog'
│   │   ├── shokolad
│   │   ├── sovun
│   │   ├── stakan
│   │   ├── televizor
│   │   ├── tosh
│   │   ├── toza
│   │   ├── turish
│   │   ├── yomg'ir
│   │   ├── yopish
│   │   └── yordam_berish
│   └── validation
│       ├── assalomu_alaykum
│       ├── bahor
│       ├── birga
│       ├── bo'sh
│       ├── bosh_kiyim
│       ├── boshlanishi
│       ├── bozor
│       ├── eshik
│       ├── futbol
│       ├── iltimos
│       ├── internet
│       ├── javob
│       ├── jismoniy_tarbiya
│       ├── karam
│       ├── kartoshka
│       ├── kichik
│       ├── kitob
│       ├── ko'prik
│       ├── likopcha
│       ├── maktab
│       ├── mehmonxona
│       ├── mehribon
│       ├── metro
│       ├── musiqa
│       ├── o'simlik_yog'i
│       ├── o'ynash
│       ├── ochish
│       ├── ot
│       ├── ovqat_tayyorlash
│       ├── oxiri
│       ├── poezd
│       ├── pomidor
│       ├── qidirish
│       ├── qish
│       ├── qo'ziqorin
│       ├── qor
│       ├── qorong'i
│       ├── quyon
│       ├── restoran
│       ├── sariyog'
│       ├── shokolad
│       ├── sovun
│       ├── stakan
│       ├── televizor
│       ├── tosh
│       ├── toza
│       ├── turish
│       ├── yomg'ir
│       ├── yopish
│       └── yordam_berish
├── Data_Numpy_Arrays_RSL_UzSL
├── data-preprocessed
├── dataset-prep
│   ├── dataset-checks
│   │   ├── 01_check_frames.py
│   │   ├── 02_count_repetitions.py
│   │   ├── 03_verify_dataset_splits.py
│   │   └── 04_check_frames_after_dataset_splits.py
│   ├── README.md
│   ├── step01_reorganize_dataset.py
│   └── step02_train_val_test_split.py
├── docs
│   ├── gifs
│   │   ├── both_eyes.gif
│   │   ├── both_hand.gif
│   │   ├── face.gif
│   │   ├── full_body.gif
│   │   ├── inference_usage.gif
│   │   ├── left_hand.gif
│   │   ├── lip.gif
│   │   ├── nose.gif
│   │   ├── pose.gif
│   │   └── right_hand.gif
│   └── images
│       ├── augment_v1_data_flow.png
│       ├── data_preprocess_augment_v1.png
│       ├── layers(sohn-h).png
│       ├── model_architecture(sohn-h).png
│       ├── model_results(sohn-h).png
│       ├── norm_reference_point(sohn-h).png
│       ├── preprocess_v1_data_flow.png
│       └── training_config(sohn-h).png
├── environment-uzslr-signs.yml
├── inferencing
│   ├── __pycache__
│   │   ├── inference_config.cpython-39.pyc
│   │   ├── inference_model.cpython-39.pyc
│   │   ├── inference_preprocess.cpython-39.pyc
│   │   ├── inference01_config.cpython-314.pyc
│   │   ├── inference01_config.cpython-39.pyc
│   │   ├── inference02_preprocess.cpython-39.pyc
│   │   └── inference03_model.cpython-39.pyc
│   ├── best_model.pth
│   ├── inference01_config.py
│   ├── inference02_preprocess.py
│   ├── inference03_model.py
│   ├── inference04_main.py
│   └── README.md
├── modeling
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-39.pyc
│   ├── notebooks
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── 02_ak_preprocess_v2.ipynb
│   │   ├── 03_ak_model_dev_v1.ipynb
│   │   ├── best_model.pth
│   │   ├── checkpoint.pth
│   │   └── fake_data
│   └── README.md
├── preprocessing
│   ├── notebooks
│   │   ├── 01_ak_exploratory_analysis.ipynb
│   │   ├── 02_ak_preprocess_v1.ipynb
│   │   ├── 02_ak_preprocess_v2.ipynb
│   │   └── fake_data
│   ├── README.md
│   └── scripts
├── README.md
├── REPRODUCIBILITY.md
├── requirements.txt
├── show-50-signs
│   ├── README.md
│   └── signs
│       ├── assalomu_alaykum
│       ├── bahor
│       ├── birga
│       ├── bo'sh
│       ├── bosh_kiyim
│       ├── boshlanishi
│       ├── bozor
│       ├── eshik
│       ├── futbol
│       ├── iltimos
│       ├── internet
│       ├── javob
│       ├── jismoniy_tarbiya
│       ├── karam
│       ├── kartoshka
│       ├── kichik
│       ├── kitob
│       ├── ko'prik
│       ├── likopcha
│       ├── maktab
│       ├── mehmonxona
│       ├── mehribon
│       ├── metro
│       ├── musiqa
│       ├── o'simlik_yog'i
│       ├── o'ynash
│       ├── ochish
│       ├── ot
│       ├── ovqat_tayyorlash
│       ├── oxiri
│       ├── poezd
│       ├── pomidor
│       ├── qidirish
│       ├── qish
│       ├── qo'ziqorin
│       ├── qor
│       ├── qorong'i
│       ├── quyon
│       ├── restoran
│       ├── sariyog'
│       ├── shokolad
│       ├── sovun
│       ├── stakan
│       ├── televizor
│       ├── tosh
│       ├── toza
│       ├── turish
│       ├── yomg'ir
│       ├── yopish
│       └── yordam_berish
├── uzslr_environments_with_hash.yml
├── venv
│   ├── bin
│   │   ├── activate
│   │   ├── activate_this.py
│   │   ├── activate.csh
│   │   ├── activate.fish
│   │   ├── activate.nu
│   │   ├── activate.ps1
│   │   ├── pip
│   │   ├── pip-3.12
│   │   ├── pip3
│   │   ├── pip3.12
│   │   ├── python -> /usr/local/bin/python3.12
│   │   ├── python3 -> python
│   │   └── python3.12 -> python
│   ├── lib
│   │   └── python3.12
│   └── pyvenv.cfg
├── video-collector
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── config.cpython-39.pyc
│   │   ├── mod01_config.cpython-39.pyc
│   │   ├── mod02_storage.cpython-39.pyc
│   │   ├── mod03_recorder.cpython-39.pyc
│   │   ├── mod04_ui.cpython-39.pyc
│   │   ├── recorder.cpython-39.pyc
│   │   ├── storage.cpython-39.pyc
│   │   └── ui.cpython-39.pyc
│   ├── Data_Numpy_Arrays_RSL_UzSL
│   │   ├── DATASET_INFO.txt
│   │   ├── signer01
│   │   ├── signer02
│   │   ├── signer03
│   │   ├── signer04
│   │   ├── signer05
│   │   ├── signer06
│   │   ├── signer07
│   │   ├── signer08
│   │   ├── signer09
│   │   └── signer10
│   ├── dataset-checks
│   │   ├── 01_check_sign_count_per_signer.py
│   │   ├── 02_count_repetitions_per_sign.py
│   │   ├── 03_check_rep_consistency.py
│   │   ├── 04_visualize_landmarks.py
│   │   ├── 05_verify_npy_shapes.py
│   │   └── 06_trash_unwanted_sign.py
│   ├── environment-video-collector.yml
│   ├── mod01_config.py
│   ├── mod02_storage.py
│   ├── mod03_recorder.py
│   ├── mod04_ui.py
│   ├── mod05_main.py
│   └── README.md
└── web_app
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-312.pyc
    │   └── __init__.cpython-39.pyc
    ├── backend
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── config.py
    │   ├── llm_client.py
    │   ├── main.py
    │   ├── model.py
    │   └── preprocess.py
    ├── best_model.pth
    ├── Dockerfile
    ├── Dockerfile.llm
    ├── environment-web-uzslr-signs.yml
    ├── frontend
    │   ├── app.js
    │   ├── index.html
    │   ├── signs.html
    │   └── style.css
    ├── ollama-models
    │   ├── blobs
    │   ├── manifests
    │   └── models
    ├── README.md
    ├── requirements-docker.txt
    └── requirements-local.txt

252 directories, 115 files

```

### tree -L 2 .

```
tree -L 2 .
.
├── conda_deps_from_history.yml
├── CONDA_INFO.md
├── data
│   ├── test
│   ├── train
│   └── validation
├── Data_Numpy_Arrays_RSL_UzSL
├── data-preprocessed
├── dataset-prep
│   ├── dataset-checks
│   ├── README.md
│   ├── step01_reorganize_dataset.py
│   └── step02_train_val_test_split.py
├── docs
│   ├── gifs
│   └── images
├── environment-uzslr-signs.yml
├── inferencing
│   ├── __pycache__
│   ├── best_model.pth
│   ├── inference01_config.py
│   ├── inference02_preprocess.py
│   ├── inference03_model.py
│   ├── inference04_main.py
│   └── README.md
├── modeling
│   ├── __init__.py
│   ├── __pycache__
│   ├── notebooks
│   └── README.md
├── preprocessing
│   ├── notebooks
│   ├── README.md
│   └── scripts
├── README.md
├── REPRODUCIBILITY.md
├── requirements.txt
├── show-50-signs
│   ├── README.md
│   └── signs
├── uzslr_environments_with_hash.yml
├── venv
│   ├── bin
│   ├── lib
│   └── pyvenv.cfg
├── video-collector
│   ├── __init__.py
│   ├── __pycache__
│   ├── Data_Numpy_Arrays_RSL_UzSL
│   ├── dataset-checks
│   ├── environment-video-collector.yml
│   ├── mod01_config.py
│   ├── mod02_storage.py
│   ├── mod03_recorder.py
│   ├── mod04_ui.py
│   ├── mod05_main.py
│   └── README.md
└── web_app
    ├── __init__.py
    ├── __pycache__
    ├── backend
    ├── best_model.pth
    ├── Dockerfile
    ├── Dockerfile.llm
    ├── environment-web-uzslr-signs.yml
    ├── frontend
    ├── ollama-models
    ├── README.md
    ├── requirements-docker.txt
    └── requirements-local.txt

34 directories, 37 files
```