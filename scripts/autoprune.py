import glob
import time
import shutil
import os
from tqdm import tqdm


from modules import shared

from constants import (AUTOPRUNE_FAILED_PATH, COMPONENT_SAVE_PATH, 
                       VAE_SAVE_PATH, MODEL_EXT, MODEL_SAVE_PATH)
from toolkit import fix_model, inspect_model, load, prune_model, save


def autoprune_move(in_file, out_folder):
    """
    Move a given file to a specified output folder. If a file with the same name 
    exists in the output folder, append a number to the file name to avoid 
    overwriting. The folders are created if they do not exist.

    Args:
    in_file (str): Path to the file to be moved.
    out_folder (str): Path to the destination folder.
    """
    name = in_file.rsplit(os.path.sep, 1)[1]
    name, ext = name.rsplit(".", 1)

    out_file = os.path.join(out_folder, f"{name}.{ext}")
    i = 1
    while os.path.exists(out_file):
        out_file = os.path.join(out_folder, f"{name}({i}).{ext}")
        i += 1

    os.makedirs(out_folder, exist_ok=True)
    shutil.move(in_file, out_file)


def autoprune_delete(in_file):
    """
    Try to delete a specified file. If the deletion fails, retries up to 100 
    times with a short delay between attempts. If the file still cannot be 
    deleted, outputs a message.

    Args:
    in_file (str): Path to the file to be deleted.
    """
    # Check if the file exists before trying to delete it.
    if not os.path.exists(in_file):
        print("\tFILE DOES NOT EXIST")
        return

    file_in_use = False  # Flag to keep track of PermissionError

    for _ in range(100):
        try:
            os.remove(in_file)
            if file_in_use:
                file_in_use = False
            break
        except PermissionError:
            # The file is in use. Set the flag and wait for a while before retrying.
            file_in_use = True
            time.sleep(0.05)
            continue
        except Exception as err:
            print(f"\tERROR: {err}")
            break
    else:
        print("\tCOULD NOT REMOVE")

    # If the file was in use during the attempts, log it now
    if file_in_use:
        print("\tFILE WAS IN USE")


def autoprune_get_models(dir):
    """
    Find all model files in the given directory. Model file extensions are 
    defined in the constant MODEL_EXT. Returns the absolute paths to these files.

    Args:
    dir (str): Directory to search for model files.

    Returns:
    list: List of absolute paths to the model files found.
    """
    ext = ["*" + e for e in MODEL_EXT]
    files = []
    for e in ext:
        for file in glob.glob(dir + os.sep + e):
            files += [os.path.abspath(file)]
    return files


def autoprune(in_folder):
    """
    Process the model files in the given input folder. Each file is loaded, 
    inspected, and pruned. If there are issues with the model (not intact, 
    unknown architecture, broken architecture), the file is moved to a failure 
    path. After pruning, if the size of the model remains the same, the model is 
    not saved and the original file is not deleted. Otherwise, the pruned model 
    is saved in a folder depending on the model architecture, and the original 
    file is deleted.

    Args:
    in_folder (str): Path to the folder containing the model files to be processed.
    """
    time.sleep(5)
    models = autoprune_get_models(in_folder)
    for in_file in tqdm(models, desc='Pruning models'):
        name = in_file.rsplit(os.path.sep, 1)[1]
        print("PRUNING", name)
        try:
            model, _ = load(in_file)
            original_size = os.path.getsize(in_file)
        except Exception:
            print("\tFAILED: MODEL NOT INTACT")
            continue
        fix_model(model, fix_clip=shared.opts.model_toolkit_fix_clip)
        found = inspect_model(model)

        if not found:
            print("\tFAILED: UNKNOWN ARCHITECTURE")
            print("\tMOVING TO Failed")
            autoprune_move(in_file, AUTOPRUNE_FAILED_PATH)
            continue

        arch = list(found.keys())[0]
        print("\tDETECTED AS", arch)
        if "BROKEN" in arch:
            print("\tFAILED: BROKEN ARCHITECTURE")
            print("\tMOVING TO Failed")
            autoprune_move(in_file, AUTOPRUNE_FAILED_PATH)
            continue

        prune_model(model, found, False, False)

        ext = "safetensors"
        if arch.startswith("SD"):
            out_folder = MODEL_SAVE_PATH
        elif arch.startswith("VAE"):
            out_folder = VAE_SAVE_PATH
            ext = "pt"
        else:
            out_folder = COMPONENT_SAVE_PATH

        name = name.rsplit(".", 1)[0]
        out_file = os.path.join(out_folder, f"{name}.{ext}")

        save(model, {}, out_file)
        pruned_size = os.path.getsize(out_file)

        if pruned_size == original_size:
            print("\tNO CHANGE")
            os.remove(out_file)  # Remove the saved pruned model
        else:
            print("\tMOVING TO", out_file)
            autoprune_delete(in_file)
            print("\tDONE")
