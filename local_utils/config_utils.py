import importlib
import os
import sys
from easydict import EasyDict

def load_config(config_file: str = None) -> EasyDict:
    if config_file:
        # Remove extension in case the user gave it
        config_file = os.path.splitext(config_file)[0]
        path = os.path.abspath(os.path.dirname(config_file))
        module = os.path.basename(config_file)
    else:
        path = "."
        module = "global_configuration.config"

    try:
        sys.path.insert(0, path)
        print("Importing configuration {:s} from {:s}".format(module, path))
        config = importlib.import_module(module)
        sys.path.remove(path)
        return config
    except (ImportError, SyntaxError, NameError) as e:
        print("Configuration file not found or invalid: %s" % str(e))
        exit(1)
