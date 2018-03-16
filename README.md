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
tensorboard tensorboard --logdir=tb_logs/ > ./tensorboard.log 2>&1 &
python main.py --epoch 1
```

And then you can open a tunnel from your terminal and look at the tensorboard at http://localhost:6006/
```
ssh -L 6006:localhost:6006 prince
```
