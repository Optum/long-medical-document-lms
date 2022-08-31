#!/usr/bin/env bash
conda env remove --name=text-blocks
conda create --name=text-blocks python=3.8 -y
source activate text-blocks
pip install -r $BASE_PATH/requirements.txt
python $BASE_PATH/gen_blocks_msp.py
python $BASE_PATH/gen_blocks_soc.py