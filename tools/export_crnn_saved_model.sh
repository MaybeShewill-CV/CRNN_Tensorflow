#!/usr/bin/env bash

python tools/export_saved_model.py -s model/crnn_syn90k_saved_model -i model/crnn_syn90k/shadownet.ckpt \
-c data/char_dict/char_dict_en.json -o data/char_dict/ord_map_en.json