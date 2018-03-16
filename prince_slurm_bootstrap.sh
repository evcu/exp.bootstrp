module load python3/intel/3.6.3
module load cuda/9.0.176

if [[ $1 == "install" ]]; then
  pip3 install --user http://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
  pip3 install --upgrade --user torchvision tensorflow tensorboardX
  pip3 install --user git+https://github.com/evcu/pytorchpruner
fi

cat<<EOF
1.Run the tensorboard first on the gpu instance and optionally background it
  tensorboard tensorboard --logdir=tb_logs/ > ./tensorboard.log 2>&1 &
  kill %1
2.Then from local(if you already haven't openned the tunnel)
  ssh -L 6006:localhost:6006 prince
3.go to the browser
  http://localhost:6006/
EOF
