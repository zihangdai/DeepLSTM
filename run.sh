source ~/.bashrc

ROOT_DIR=$(pwd)

export LD_LIBRARY_PATH=$ROOT_DIR/lib:$ROOT_DIR/lib/openblas/lib:$ROOT_DIR/lib/glog/lib:$LD_LIBRARY_PATH

echo $LD_LIBRARY_PATH

#./lstmRNN
mpirun -n 2 ./RNNTranslator
