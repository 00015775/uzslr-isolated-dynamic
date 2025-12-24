# Conda environments

Jupyter runs the user's code in a separate process called *kernel*. The kernel can be a different Python installation (in a different conda environment or virtualenv or Python 2 instead of Python 3) or even an interpreter for a different language (e.g. Julia or R). Kernels are configured by specifying the interpreter and a name and some other parameters (see [Jupyter documentation][5]) and configuration can be stored system-wide, for the active environment (or virtualenv) or per user. If `nb_conda_kernels` is used, additional to statically configured kernels, a separate kernel for each conda environment with `ipykernel` installed will be available in Jupyter notebooks.

In short, there are three options how to use a conda environment and Jupyter:

## Option 1: Run Jupyter server and kernel inside the conda environment

Do something like:
```bash
    conda create -n my-conda-env python=x.x   # creates new virtual env
    conda activate my-conda-env          # activate environment in terminal
    conda install jupyter                # install jupyter + notebook
    jupyter notebook                     # start server + kernel inside my-conda-env
```

Jupyter will be completely installed in the conda environment. Different versions of Jupyter can be used
for different conda environments, but this option might be a bit of overkill. It is enough to
include the kernel in the environment, which is the component wrapping Python which runs the code.
The rest of Jupyter notebook can be considered as editor or viewer and it is not necessary to
install this separately for every environment and include it in every `env.yml` file. Therefore one
of the next two options might be preferable, but this one is the simplest one and definitely fine.

## Option 2: Create special kernel for the conda environment

Do something like:
```bash
    conda create -n my-conda-env python=x.x                    # creates new virtual env
    conda activate my-conda-env                                # activate environment in terminal
    conda install ipykernel                                    # install Python kernel in new conda env
    ipython kernel install --user --name=my-conda-env-kernel --display-name="Python (My Project Env)"  # configure Jupyter to use Python kernel
```
Then run jupyter from the system installation or a different conda environment:
```bash
    conda deactivate          # this step can be omitted by using a different terminal window than before
    conda install jupyter     # optional, might be installed already in system e.g. by 'apt install jupyter' on debian-based systems
    jupyter notebook          # run jupyter from system
```
Name of the kernel and the conda environment are independent from each other, but it might make sense to use a similar name.

Only the Python kernel will be run inside the conda environment, Jupyter from system or a different conda environment will be used - it is not installed in the conda environment. By calling `ipython kernel install` the jupyter is configured to use the conda environment as kernel, see [Jupyter documentation][5] and [IPython documentation][6] for more information. In most Linux installations this configuration is a `*.json` file in `~/.local/share/jupyter/kernels/my-conda-env-kernel/kernel.json`:
```bash
    {
     "argv": [
      "/opt/miniconda3/envs/my-conda-env/bin/python",
      "-m",
      "ipykernel_launcher",
      "-f",
      "{connection_file}"
     ],
     "display_name": "my-conda-env-kernel",
     "language": "python"
    }
```
## Option 3: Use nb_conda_kernels to use a kernel in the conda environment


When the [package `nb_conda_kernels`](https://github.com/Anaconda-Platform/nb_conda_kernels#nb_conda_kernels) is installed, a separate kernel is available automatically for each
conda environment containing the conda package `ipykernel` or a different kernel (R, Julia, ...).
```bash
    conda activate my-conda-env    # this is the environment for your project and code
    conda install ipykernel
    conda deactivate
    
    conda activate base            # could be also some other environment
    conda install nb_conda_kernels
    jupyter notebook
```
You should be able to choose the Kernel `Python [conda env:my-conda-env]`. Note that `nb_conda_kernels` seems to be available only via conda and not via pip or other package managers like apt.

# Troubleshooting

Using Linux/Mac the command `which` on the command line will tell you which jupyter is used, if you
are using option 1 (running Jupyter from inside the conda environment), it should be an executable
from your conda environment:
```bash
    $ which jupyter
    /opt/miniconda3/envs/my-conda-env/bin/jupyter
    $ which jupyter-notebook   # this might be different than 'which jupyter'! (see below)
    /opt/miniconda3/envs/my-conda-env/bin/jupyter-notebook
```
Inside the notebook you should see that Python uses Python paths from the conda environment:
```bash
    [1] !which python
    /opt/miniconda3/envs/my-conda-env/bin/python
    [2] import sys; sys.executable
    '/opt/miniconda3/envs/my-conda-env/bin/python'
    ['/home/my_user',
     '/opt/miniconda3/envs/my-conda-env/lib/python37.zip',
     '/opt/miniconda3/envs/my-conda-env/lib/python3.7',
     '/opt/miniconda3/envs/my-conda-env/lib/python3.7/lib-dynload',
     '',
     '/opt/miniconda3/envs/my-conda-env/lib/python3.7/site-packages',
     '/opt/miniconda3/envs/my-conda-env/lib/python3.7/site-packages/IPython/extensions',
     '/home/my_user/.ipython']
```
Jupyter provides the command `jupyter-troubleshoot` or in a Jupyter notebook:
```bash
    !jupyter-troubleshoot
```
This will print a lot of helpful information about including the outputs mentioned above as well as installed libraries and others. When
asking for help regarding Jupyter installations questions, it might be good idea to provide this information in bug reports or questions.

To list all configured Jupyter kernels run:
```bash
    jupyter kernelspec list
```

# Common errors and traps

## Jupyter notebook not installed in conda environment

Note: symptoms are not unique to the issue described here.

**Symptoms:** ImportError in Jupyter notebooks for modules installed in the conda environment (but
not installed system wide), but no error when importing in a Python terminal

**Explaination:** You tried to run jupyter notebook from inside your conda environment
(option 1, see above), there is no configuration for a kernel for this conda environment (this
would be option 2) and nb_conda_kernels is not installed (option 3), but jupyter notebook is not (fully)
installed in the conda environment, even if `which jupyter` might make you believe it was.

In GNU/Linux you can type `which jupyter` to check which executable of Jupyter is run.

This means that system's Jupyter is used, probably because Jupyter is not installed:
```bash
    (my-conda-env) $ which jupyter-notebook
    /usr/bin/jupyter
```
If the path points to a file in your conda environment, Jupyter is run from inside Jupyter:
```bash
    (my-conda-env) $ which jupyter-notebook
    /opt/miniconda3/envs/my-conda-env/bin/jupyter-notebook
```
Note that when the conda package `ipykernel` is installed, an executable `jupyter` is shipped, but
no executable `jupyter-notebook`. This means that `which jupyter` will return a path to the conda
environment but `jupyter notebook` will start system's `jupyter-nootebook` (see also [here][7]):
```bash
     $ conda create -n my-conda-env
     $ conda activate my-conda-env
     $ conda install ipykernel
     $ which jupyter            # this looks good, but is misleading!
     /opt/miniconda3/envs/my-conda-env/bin/jupyter
     $ which jupyter-notebook   # jupyter simply runs jupyter-notebook from system...
     /usr/bin/jupyter-notebook
```
This happens because `jupyter notebook` searches for `jupyter-notebook`, finds
`/usr/bin/jupyter-notebook` and
[calls it](https://github.com/jupyter/jupyter_core/blob/340b63a466736764b2931b5d7d910113eb1d94fd/jupyter_core/command.py#L120)
starting a new Python process. The shebang in `/usr/bin/jupyter-notebook` is `#!/usr/bin/python3`
and [not a dynamic](https://stackoverflow.com/questions/1530702/dont-touch-my-shebang)
`#!/usr/bin/env python`.
Therefore Python manages to break out of the conda environment. I guess jupyter could call
`python /usr/bin/jupyter-notebook` instead to overrule the shebang, but mixing
system's bin files and the environment's python path can't work well anyway.

**Solution:** Install jupyter notebook inside the conda environment:
```bash
     conda activate my-conda-env
     conda install jupyter
     jupyter notebook
```

## Wrong kernel configuration: Kernel is configured to use system Python

Note: symptoms are not unique to the issue described here.

**Symptoms:** ImportError in Jupyter notebooks for modules installed in the conda environment (but
not installed system wide), but no error when importing in a Python terminal

**Explanation:** Typically the system provides a kernel called python3 (display name "Python 3")
configured to use `/usr/bin/python3`, see e.g. `/usr/share/jupyter/kernels/python3/kernel.json`.
This is usually overridden by a kernel in the conda environment, which points to the environments
python binary `/opt/miniconda3/envs/my-conda-env/bin/python`. Both are generated by the package
`ipykernel` (see [here](https://github.com/ipython/ipykernel/blob/d283077435138f6bb5d77624e887a2552574d773/setup.py#L130)
and [here](https://github.com/ipython/ipykernel/blob/d283077435138f6bb5d77624e887a2552574d773/ipykernel/kernelspec.py#L19)).

A user kernel specification in `~/.local/share/jupyter/kernels/python3/kernel.json` might override
the system-wide and environment kernel. If the environment kernel is missing or the user kernel
points to a python installation outside the environment option 1 (installation of jupyter in the
environment) will fail.

For occurrences and discussions of this problem and variants see [here](https://github.com/jupyter/notebook/issues/2898),
[here](https://github.com/jupyter/notebook/issues/4447),
[here](https://stackoverflow.com/questions/39604271/conda-environments-not-showing-up-in-jupyter-notebook/54985829#54985829)
and also [here](https://stackoverflow.com/questions/46551200/changing-jupyter-kernelspec-to-point-to-anaconda-python),
[here](https://github.com/jupyter/notebook/issues/397) and
[here](https://github.com/jupyter/jupyter/issues/245).

**Solution:** Use `jupyter kernelspec list` to list the location active kernel locations.
```bash
    $ conda activate my-conda-env
    $ jupyter kernelspec list
    Available kernels:
      python3 /opt/miniconda3/envs/my-conda-env/share/jupyter/kernels/python3
```
If the kernel in the environment is missing, you can try creating it manually using
`ipython kernel install --sys-prefix` in the activated environment, but it is probably better to
check your installation, because `conda install ipykernel` should have created the environment
(maybe try re-crate the environment and re-install all packages?).

If a user kernel specification is blocking the environment kernel specification, you can either
remove it or use a relative python path which will use `$PATH` to figure out which `python` to use.
So something like this, should be totally fine:
```bash
    $ cat ~/.local/share/jupyter/kernels/python3/kernel.json
    {
     "argv": [
      "python",
      "-m",
      "ipykernel_launcher",
      "-f",
      "{connection_file}"
     ],
     "display_name": "Python 3",
     "language": "python"
    }
```

## Correct conda environment not activated

**Symptoms:** ImportError for modules installed in the conda environment (but not installed system
wide) in Jupyter notebooks and Python terminals

**Explanation:** Each terminal has a set of environment variables, which are lost when the terminal
is closed. In order to use a conda environment certain environment variables need to be set, which
is done by activating it using `conda activate my-conda-env`. If you attempted to run Jupyter
notebook from inside the conda environment (option 1), but did not activate the conda environment
before running it, it might run the system's jupyter.

**Solution:** Activate conda environment before running Jupyter.
```bash
     conda activate my-conda-env
     jupyter notebook
```

## Broken kernel configuration

**Symptoms:** Strange things happening. Maybe similar symptoms as above, e.g. ImportError

**Explanation:** If you attempted to use option 2, i.e. running Jupyter from system and the Jupyter
kernel inside the conda environment by using an explicit configuration for the kernel, but it does
not behave as you expect, the configuration might be corrupted in [some way][1].

**Solution:** Check configuration in `~/.local/share/jupyter/kernels/my-kernel-name/kernel.json`
and fix mistakes manually or remove the entire directory and re-create it using the command
provided above for option 2. If you can't find the kernel configuration there run
`jupyter kernelspec list`.

## Python 2 vs 3


**Symptoms:** ImportError due to [wrong Python version of the Jupyter kernel][3] or \[other problems
with Python 2/3\]

**Explanation:** The kernel configuration can have all sorts of confusing and misleading effects.
For example the default Python 3 kernel configuration will allow me to launch a Jupyter notebook
running on Python 2:
```bash
    conda create -n my-conda-env
    conda activate my-conda-env
    conda install python=2
    conda install jupyter
    jupyter notebook
```
The default Python 3 kernel:
```bash
    $ cat ~/.local/share/jupyter/kernels/python3/kernel.json
    {
     "argv": [
      "python",
      "-m",
      "ipykernel_launcher",
      "-f",
      "{connection_file}"
     ],
     "display_name": "Python 3",
     "language": "python"
    }
```
After creating a new Jupyter Notebook with the Python 3 kernel, Python 2 from the conda
environment will be used even if "Python 3" is displayed by Jupyter.

**Solution:** [Don't use Python 2 ;-)][2]

[1]: https://github.com/jupyter/notebook/issues/2359#issuecomment-291114172
[2]: https://pythonclock.org/
[3]: https://stackoverflow.com/questions/43437884/jupyter-notebook-import-error-no-module-named-matplotlib
[5]: https://jupyter-client.readthedocs.io/en/stable/kernels.html#kernel-specs
[6]: https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments
[7]: https://github.com/jupyter/notebook/issues/3311#issuecomment-363804324


**Reference**  
Information taken from [StackOverFlow (2019) - How to use Jupyter notebooks in a conda environment?](https://stackoverflow.com/a/58068850/31606496)

