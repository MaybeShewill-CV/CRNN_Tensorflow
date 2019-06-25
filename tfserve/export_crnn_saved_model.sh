 #!/usr/bin/env bash
 # author: github.com/eldon

set -eux

PYTHONPATH=$(pwd) python tfserve/export_saved_model.py \
    --export_dir model/crnn_syn90k_saved_model \
    --ckpt_path model/crnn_syn90k/shadownet.ckpt \
    --char_dict_path data/char_dict/char_dict_en.json \
    --ord_map_dict_path data/char_dict/ord_map_en.json

rm -rf /tmp/crnn/1
mkdir -p /tmp/crnn/1
mv -f model/crnn_syn90k_saved_model/* /tmp/crnn/1
