# Conda environments for effective and reproducible research

* While the command `conda env export > environments.yml` does produce the list of all packages downloaded by both `conda` and `pip`, it certainly comes with its flaws. Basically, the packages listed within `conda env export > environments.yml` because apart from mentioning the specific package version, for some packages it can specify the system specific package version. For instance, instead of writing as `1.23.4` for package version, it can for some packages write as `hbd8a1cb_0` for both manually installed and pre-installed packages. 

* This strange package versioning is called [build variant hash](https://docs.conda.io/projects/conda-build/en/latest/resources/variants.html#differentiating-packages-built-with-different-variants), and this versioning appears differently for different operating systems. The implication is that an environment file that contains a build variant hash for one or more of the packages **cannot be used on a different operating system (OS)** to the one it was created on. <u>**Basically includes system-specific low level packages, hence `conda env export > environments.yml` is not OS agnostic.**</u>

* In order to solve listing of OS specific packages, ultimately producing reproducible OS agnostic conda environment for other people to use regarding of their operating system, we can add `--from-history` argument to the `conda env export` command. The full command is `conda env export --from-history -f conda_deps.yml`. The difference being with this command, is that it will only include the packages a person has explicitly installed with `conda install` and the version a person has requested to be installed. For example if you installed `numpy=1.24` this will be listed, but if you installed pandas without specifing the version conda will automatically try to install the latest pandas version of the given time, and pandas will be listed in your environment file <u>without a version number so anyone using your environment file will get the latest version which may not match the version you used</u>. This is one reason to explicitly state the version of a package you wish to install.

* Without `--from-history` the output may on some occasions include the build variant hash (which can alternatively be removed by editing the environment file). These are often specific to the operating system and including them in your environment file means **it will not necessarily work if someone is using a different operating system**.

<div class="alert alert-block alert-danger", style="font-weight: 600">
  Be aware that <code>--from-history</code> will omit any packages you have installed using <code>pip</code>. This may be addressed in future releases. However in the meantime, editing your exported environment files by hand is sometimes the best option. Include the package version explicitly as well.
</div>

* It is important to note that the exported file includes the `prefix:` entry which records the location the environment is installed on your system. However, **If you are to share this file with colleagues you should remove this line before doing so as it is highly unlikely their virtual environments will be in the same location**. <u>You should also ideally remove `prefix:` the line before committing to GitHub</u>. This `prefix:` is usually located at the bottom of the `.yml` file, and looks like that: 

`prefix: /opt/anaconda3/envs/CONDA_ENV_NAME`

* Since the conda env export `--from-history -f conda_deps.yml` does not include packages installed with `pip`, we can include them manually. We can create the `requirements.txt` file of all the `pip` packages installed along with their versions, with the command of `pip list --format=freeze > requirements.txt` 

* Sometimes you may not specify the package versioning by simply using `conda install` or `pip install` along with the package name, the command of `conda env export --from-history -f conda_deps.yml` also will not specify the package versioning. This can be manually solved, by running `conda env export > environments.yml`, we can see the actual package version and simply copy and paste to the main `environments.yml`.

<div class="alert alert-block alert-success", style="font-weight: 600">
  Creating your project’s Conda environment from a single environment file is a Conda "best practice". Not only do you have a file to share with collaborators but you also have a file that can be placed under version control which further enhances the reproducibility of your research project and workflow.
</div>

```code
name: conda_env
channels:
  - conda-forge
  - defaults
  - https://repo.anaconda.com/pkgs/main
  - https://repo.anaconda.com/pkgs/r
dependencies:
  - python=3.11.14
  - ipykernel=7.1.0
  - jupyter=1.1.1
  - streamlit=1.51.0
  - matplotlib=3.10.8
  - seaborn=0.13.2
  - ucimlrepo=0.0.7
  - matplotlib-venn=1.1.2
  - xgboost=3.1.1
  - lightgbm=4.6.0
  - scikit-learn=1.7.2
  - pandas=2.3.3
  - numpy=1.26.4
  - scipy=1.16.3
  - joblib=1.5.2
  - pip:
      - altair==5.5.0
      - anyio==4.11.0
      - appnope==0.1.4
      - argon2-cffi==25.1.0
      - argon2-cffi-bindings==25.1.0
      - arrow==1.4.0
      - asttokens==3.0.1
```


* Givent that `environments.yml` file is setup properly, the following code must be run inside of the directory where the `environments.yml` is located, usually at the root directory.
```code
conda env create -f environments.yml
```
* After getting the required packages and their respective versions, we can now activate the conda environment with the following code, but since for other projects the name of environment will be different for different projects, check the `name: conda_env_name` at the top of `environments.yml`.
```code
conda activate conda_env
```

**Referece:**\
Information learned from\
[University of Sheffield Research Software Engineering (2023) - Conda environments for effective and reproducible research](https://rse.shef.ac.uk/conda-environments-for-effective-and-reproducible-research/04-sharing-environments/index.html)

