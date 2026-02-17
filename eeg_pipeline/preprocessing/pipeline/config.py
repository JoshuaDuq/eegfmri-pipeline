import importlib.util


###################################################################
# Configuration Management
###################################################################

def update_config(config, new_values, outfile=None):
    with open(config, "r") as file:
        lines = file.readlines()

    for key, value in new_values.items():
        found = False
        for i, line in enumerate(lines):
            if line.replace(" ", "").startswith(key + "=") and "no update" not in line:
                if isinstance(value, str):
                    lines[i] = f"{key} = '{value}'\n"
                else:
                    lines[i] = f"{key} = {value}\n"
                found = True
                break
        if not found:
            lines.append("\n")
            lines.append(f"{key} = {value}\n")

    if outfile is not None:
        with open(outfile, "w") as file:
            file.writelines(lines)
    else:
        with open(config, "w") as file:
            file.writelines(lines)


def get_specific_config(config_file, prefix):
    spec = importlib.util.spec_from_file_location(
        name="custom_config", location=config_file
    )
    custom_cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_cfg)
    
    config = {}
    for key in dir(custom_cfg):
        if prefix + "_" in key:
            val = getattr(custom_cfg, key)
            config[key.replace(prefix + "_", "")] = val
    
    return config


def get_config_keyval(config_file, key, return_false_if_not_found=True):
    spec = importlib.util.spec_from_file_location(
        name="custom_config", location=config_file
    )
    custom_cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_cfg)
    
    for k in dir(custom_cfg):
        if key in k:
            val = getattr(custom_cfg, k)
            return val
    
    if return_false_if_not_found:
        return False
    else:
        raise ValueError(f"Key {key} not found in {config_file}")

