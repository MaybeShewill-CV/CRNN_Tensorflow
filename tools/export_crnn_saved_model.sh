 #!/usr/bin/env bash

PYTHONPATH=$(pwd) python tools/export_saved_model.py --export_dir model/crnn_syn90k_saved_model --ckpt_path checkpoints/shadownet.ckpt \
--char_dict_path data/char_dict/char_dict_en.json --ord_map_dict_path data/char_dict/ord_map_en.json
