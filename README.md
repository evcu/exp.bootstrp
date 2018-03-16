# exp.bootstrp
This repo is a bootstrap for experiments and includes helper functions scripts for pytorch training and slurm job scheduler.

## Quick Start
Look ssh_init post.

```
ssh prince
git clone git@github.com:evcu/exp.bootstrp.git
mv exp.bootstrp my_exp
cd my_exp
```

First debug the experiment on a interactive session
prince_slurm_bootstrap.sh loads the modules needed, update as needed.
Personally I am using python3 with pip --user packages. You can call it with install for the first time

```
srun -t2:30:00 --mem=5000 --gres=gpu:1 --pty /bin/bash

. ./prince_slurm_bootstrap.sh install
cd experiments/cifar10/
python main.py --epoch 1
```

After we are sure that our main script works, we can start create automated experiments with
`create_experiment_jobs.py` scripts

```
cd ../
python create_experiment_jobs.py --debug
```
if they all look nice then you can create the experiment folder. and submit the jobs
```
python create_experiment_jobs.py
bash /scratch/ue225/my_project/exps/cifar10/cifarLR_03.26/submit_all.sh
```
which would output something like this


Let say you wanna define a new experiment. You would do by creating a new folder `experiments/new_folder/` and a `experiments/new_folder/main.py`script that is intended to be run. The main.py script should accept
`--log_folder` and `--conf_file` flags at minimum. Then you can change `exp_name` at `experiments/default_conf.yaml` to `new_folder` and create new experiments.

## Features
- with conf.yaml file seamless argParser generation. Write the conf read and overwrite with cli args.
- Customizable `eval-prefixes` which enables defining programatic eval-able arguments.
  i.e. the string '+range(5)' would be evaluated and read as the list.
- configuration copy to the experiment folder such that you can always change experiments default_args after submission
- `ClassificationTrainer`/`ClassificationTester` which wraps the main training/testing
functionalities and provides hooks for loggers.
- tensorboardX utils and examples.
- Multiple experiment definitions through yaml lists.
