#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

rm -rf build dist *.egg-info
python build_ops.py build
python build_ops.py install

LIBC10_DIR="$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib"
export LD_LIBRARY_PATH=$LIBC10_DIR:$LD_LIBRARY_PATH

SHELL_CONFIG="$HOME/.bashrc"
if [ -f "$HOME/.zshrc" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
fi

if ! grep -q "LD_LIBRARY_PATH=$LIBC10_DIR" "$SHELL_CONFIG"; then
    echo -e "\nexport LD_LIBRARY_PATH=$LIBC10_DIR:\$LD_LIBRARY_PATH" >> $SHELL_CONFIG
fi

source $SHELL_CONFIG
echo "Current LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
python -c "import cuda_block_ops; print('cuda_block_ops imported successfully')"
