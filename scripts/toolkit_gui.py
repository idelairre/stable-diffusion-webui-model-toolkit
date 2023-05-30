import glob
import os
import sys
import copy

import gradio as gr
import torch

from modules import shared, script_callbacks

from constants import (ARCHITECTURES, AUTOPRUNE_PATH, COMPONENTS, COMPONENT_EXT, COMPONENT_SAVE_PATH,
                       IDENTIFICATION, LOAD_PATHS, METADATA, MODEL_EXT, MODEL_SAVE_PATH, VAE_SAVE_PATH)
from toolkit import (build_fake_model, contains_component, find_components, delete_class, extract_component,
                     fix_model, fix_ema, get_allowed_keys, inspect_model, load, load_components, prune_model,
                     save, tensor_size, tensor_shape, replace_component, resolve_class)

os.makedirs(AUTOPRUNE_PATH, exist_ok=True)
os.makedirs(COMPONENT_SAVE_PATH, exist_ok=True)


class ToolkitModel():
    """
    ToolkitModel represents a model in the toolkit. 

    This class is used to encapsulate and manage properties related to the model, 
    including its filename, the model itself, and its metadata. It also keeps track 
    of whether the model is partially loaded, if it needs to be fixed, as well as 
    the status of its architecture, such as whether it's broken or has been renamed.

    The class further records the state of components in the architecture, such as 
    its potential components, the type of the architecture, the classes, and the 
    components in it.

    Also, it tracks the model's states, like the UNet, VAE, and CLIP modules.

    Lastly, the class also handles information about the total number of parameters 
    in the model, as well as parameters that are wasted, parameters considered junk, 
    parameters in the exponential moving average (EMA), and keys for junk and EMA.

    Attributes:
    - filename (str): File name of the model.
    - model (dict): The model itself.
    - metadata (dict): Metadata related to the model.
    - partial (bool): Flag indicating if the model is partially loaded.
    - fix_clip (bool): Flag indicating if the model's CLIP needs to be fixed.
    - broken (list): List of broken components.
    - renamed (list): List of renamed components.
    - a_found (dict): Found architecture.
    - a_rejected (dict): Rejected architecture.
    - a_resolved (dict): Resolved architecture.
    - a_potential (list): Potential architectures.
    - a_type (str): Type of the architecture.
    - a_classes (dict): Classes in the architecture.
    - a_components (list): Components in the architecture.
    - m_str (str): Model string (default is "----/----/----").
    - m_unet (any): UNet module in the model.
    - m_vae (any): VAE module in the model.
    - m_clip (any): CLIP module in the model.
    - z_total (int): Total number of parameters.
    - z_waste (int): Wasted parameters.
    - z_junk (int): Junk parameters.
    - z_ema (int): EMA parameters.
    - k_junk (list): Keys of junk parameters.
    - k_ema (list): Keys of EMA parameters.
    """

    def __init__(self):
        self.filename = ""
        self.model = {}
        self.metadata = {}

        self.partial = False

        self.fix_clip = False
        self.broken = []
        self.renamed = []

        self.a_found = {}
        self.a_rejected = {}
        self.a_resolved = {}
        self.a_potential = []
        self.a_type = ""
        self.a_classes = {}
        self.a_components = []

        self.m_str = "----/----/----"
        self.m_unet = None
        self.m_vae = None
        self.m_clip = None

        self.z_total = 0
        self.z_waste = 0
        self.z_junk = 0
        self.z_ema = 0

        self.k_junk = []
        self.k_ema = []


def do_analysis(model):
    """
    Perform analysis on a given model. The analysis includes model inspection, 
    architecture resolution, metric computation, and size computation of different 
    components of the model. The analysis results are stored in a ToolkitModel 
    object and returned.

    Args:
    model (obj): The model object to be analyzed. This should be a pre-loaded model
                 object that provides access to model parameters.

    Returns:
    ToolkitModel: Object containing the analysis results.
    """
    tm = ToolkitModel()
    tm.model = model

    tm.a_found, tm.a_rejected = inspect_model(model, all=True)
    tm.a_resolved = resolve_arch(tm.a_found)

    if not tm.a_resolved:
        tm.m_str = "----/----/----"
        return tm

    tm.a_potential = list(tm.a_found.keys())
    tm.a_type = next(iter(tm.a_resolved))
    tm.a_classes = tm.a_resolved[tm.a_type]
    tm.a_components = [tm.a_classes[c][0] for c in tm.a_classes]

    tm.m_str, m = compute_metric(model, tm.a_resolved)
    tm.m_unet, tm.m_vae, tm.m_clip = m

    allowed = get_allowed_keys(tm.a_resolved)

    for k in model.keys():
        kk = (k, tensor_shape(k, model[k]))
        z = tensor_size(model[k])
        tm.z_total += z

        if kk in allowed:
            if z and model[k].dtype == torch.float32:
                tm.z_waste += z/2
            if z and model[k].dtype == torch.float64:
                tm.z_waste += z - (z/4)
        else:
            if k.startswith(EMA_PREFIX):
                tm.z_ema += z
                tm.k_ema += [k]
            else:
                tm.z_junk += z
                tm.k_junk += [k]

    return tm


def get_size(bytes):
    """
    Convert a given size in bytes to a human-readable string. Sizes are 
    displayed in appropriate units: Bytes, KB, MB, or GB, with two decimal 
    places of precision for KB, MB, and GB.

    Args:
    bytes (int): Size in bytes.

    Returns:
    str: String representing the size in a human-readable format.
    """
    KB = 1024
    MB = KB * 1024
    GB = MB * 1024

    if bytes < KB:
        return f"{bytes} Bytes"
    if bytes < MB:
        return f"{bytes/KB:.2f} KB"
    if bytes < 1024*1024*1024:
        return f"{bytes/MB:.2f} MB"
    return f"{bytes/GB:.2f} GB"


def do_basic_report(details: ToolkitModel, dont_half, keep_ema):
    """
    Generates a basic report about the provided model.

    Args:
        details (ToolkitModel): An object that contains various statistics and details about the model.
        dont_half (bool): If true, prevents the function from reporting about precision waste in the model.
        keep_ema (bool): If true, the function will include Exponential Moving Average (EMA) data in the report.

    Returns:
        str: A string report containing various details about the model. The report includes total model size, 
            identified model type, model components, presence of junk data, EMA data, precision waste, potential issues 
            with the CLIP component, and information about the VAE and CLIP used. It also provides information on potential 
            changes required for the model.
    """
    d = details

    report = f"### Report ({d.m_str})\n-----\n"

    if not d.a_found:
        report += "Model type could not be identified.\n\n"
        return report

    out = [f"Model is **{get_size(d.z_total)}**."]

    if len(d.a_potential) > 1:
        out += [
            f"Multiple model types identified: **{', '.join(d.a_potential)}**."]
        out += [f"Model type **{d.a_type}** will be used."]
    else:
        out += [f"Model type identified as **{d.a_type}**."]

    if d.a_components:
        out += [f"Model components are: **{', '.join(d.a_components)}**."]
    else:
        out += ["**Model has no components**."]

    report += " ".join(out) + "\n\n"
    out = []

    k_junk = d.k_junk
    z_junk = d.z_junk
    if not keep_ema:
        k_junk += d.k_ema
        z_junk += d.z_ema

    if k_junk:
        if z_junk > 16777216:
            out += [f"**Contains {get_size(z_junk)} of junk data!**"]
        else:
            out += [f"**Contains {len(k_junk)} junk keys!**"]
    else:
        out += ["Contains no junk data."]

    if keep_ema:
        if d.k_ema:
            if d.z_ema > 16777216:
                out += [f"**Contains {get_size(d.z_ema)} of EMA data!**"]
            else:
                out += [f"**Contains {len(d.k_ema)} EMA keys!**"]
        else:
            out += ["Contains no EMA data."]

    if d.z_waste > 0:
        out += [f"Wastes **{get_size(d.z_waste)}** on precision."]

    if d.renamed:
        out += [f"**CLIP was mislablled, {len(d.renamed)} keys renamed.**"]
    if d.broken:
        if d.fix_clip:
            out += [
                f"**CLIP had incorrect positions, fixed:** {', '.join([str(i) for i in d.broken])}."]
        else:
            out += [
                f"**CLIP has incorrect positions, missing:** {', '.join([str(i) for i in d.broken])}."]
    if "CLIP-v2-WD" in d.a_components:
        out += ["**CLIP is missing its final layer.**"]

    report += " ".join(out) + "\n\n"
    out = []

    if d.m_vae != None:
        for k, i in IDENTIFICATION["VAE"].items():
            if abs(d.m_vae-i) <= 1:
                out += [f"Uses the **{k}** VAE."]

    if d.m_clip != None:
        for k, i in IDENTIFICATION["CLIP-v1"].items():
            if abs(d.m_clip-i) <= 1:
                out += [f"Uses the **{k}** CLIP."]

        for k, i in IDENTIFICATION["CLIP-v2"].items():
            if abs(d.m_clip-i) <= 1:
                out += [f"Uses the **{k}** CLIP."]

    report += " ".join(out) + "\n\n"

    removed = d.z_junk
    if not keep_ema:
        removed += d.z_ema
    if not dont_half:
        removed += d.z_waste

    pruned = (d.z_waste and not dont_half) or d.k_ema or d.k_junk

    changes = len(d.renamed)
    if d.fix_clip:
        changes += len(d.broken)
    changed = changes > 0

    if pruned:
        report += f"Model will be pruned to **{get_size(d.z_total-removed)}**. "
    if changed:
        report += f"Model will be fixed (**{changes}** changes)."
    if not changed and not pruned:
        report += "**Model is unaltered, nothing to be done.**"

    return report


def do_adv_report(details: ToolkitModel, abbreviate=True):
    """
    Generates an advanced report about the provided model.

    Args:
        details (ToolkitModel): An object that contains various statistics and details about the model.
        abbreviate (bool): If True, the function will abbreviate the report for certain sections
                           if they contain too many elements.

    Returns:
        str: A string report that contains various details about the model. The report includes statistics 
            such as the total number of keys, the count of useless and unknown keys, the architecture(s) identified,
            and the rejected architecture(s) with reasons. It also lists unknown keys in the model.
            Sections with many elements can be abbreviated to maintain readability.
    """
    d = details

    model_keys = set((k, tensor_shape(k, d.model[k])) for k in d.model.keys())
    allowed_keys = get_allowed_keys(d.a_resolved)
    known_keys = get_allowed_keys(d.a_found)

    unknown_keys = model_keys.difference(known_keys)
    useless_keys = known_keys.difference(allowed_keys)

    model_size, useless_size, unknown_size = 0, 0, 0

    for k in d.model:
        kk = (k, tensor_shape(k, d.model[k]))
        z = tensor_size(d.model[k])
        model_size += z
        if kk in useless_keys:
            useless_size += z
        if kk in unknown_keys:
            unknown_size += z

    report = f"### Report ({d.m_str})\n-----\n"
    report += "#### Statistics\n"
    line = [f"Total keys: **{len(model_keys)} ({get_size(model_size)})**"]
    if useless_keys:
        line += [f"Useless keys: **{len(useless_keys)} ({get_size(useless_size)})**"]
    if unknown_keys:
        line += [f"Unknown keys: **{len(unknown_keys)} ({get_size(unknown_size)})**"]

    report += ", ".join(line) + ".\n"
    report += "#### Architecture\n"

    if d.a_found:
        archs = [d.a_type] + [a for a in d.a_potential if not a == d.a_type]
        for arch in archs:
            report += f"- **{arch}**\n"

            data = d.a_found[arch]
            if arch == d.a_type and abbreviate:
                data = d.a_resolved[arch]

            for clss in data:
                report += f"  - **{clss}**\n"
                for comp in data[clss]:
                    report += f"    - {comp}\n"
                if not data[clss]:
                    report += "    - NONE\n"
            if len(archs) > 1 and arch == d.a_type:
                report += "#### Additional\n"
    else:
        report += "- *NONE*\n"

    if d.a_rejected:
        report += "#### Rejected\n"
        for arch, rejs in d.a_rejected.items():
            for rej in rejs:
                report += f"- **{arch}**: {rej['reason']}\n"
                data = list(rej["data"])
                will_abbreviate = len(rej["data"]) > 5 and abbreviate
                if will_abbreviate:
                    data = data[:5]
                for k in data:
                    if type(k) == tuple:
                        report += f"  - {k[0]} {k[1]}\n"
                    else:
                        report += f"  - {k}\n"
                if will_abbreviate:
                    report += "  - ...\n"

    if unknown_keys:
        report += "#### Unknown\n"

        data = sorted(unknown_keys)
        will_abbreviate = len(data) > 5 and abbreviate
        if will_abbreviate:
            data = data[5:]
        for k, z in data:
            report += f" - {k} {z}\n"
        if will_abbreviate:
            report += " - ...\n"

    return report


source_list = []
name_list = []
file_list = []
loaded = None


def get_models(dir):
    """
    Retrieve all model files from a specified directory and its subdirectories. 
    The model file extensions are defined in the constant MODEL_EXT.

    Args:
    dir (str): The directory to search for model files.

    Returns:
    list: List of absolute paths to the model files found.
    """
    ext = ["**" + os.sep + "*" + e for e in MODEL_EXT]
    files = []
    for e in ext:
        for file in glob.glob(dir + os.sep + e, recursive=True):
            files += [os.path.abspath(file)]
    return files


def get_lists():
    """
    Prepare global lists of model files and their names from multiple 
    predefined paths (LOAD_PATHS). The lists are sorted alphabetically. 
    Also prepares a list of source names, which includes model names and 
    architecture names.
    """
    global source_list, file_list, name_list
    file_list = []
    for path in LOAD_PATHS:
        file_list += get_models(path)
    file_list = list(set(file_list))

    name_list = [p[p.rfind(os.sep)+1:] for p in file_list]
    name_list = [n if name_list.count(
        n) == 1 else file_list[i] for i, n in enumerate(name_list)]

    file_list = sorted(
        file_list, key=lambda x: name_list[file_list.index(x)].lower())
    name_list = sorted(name_list, key=lambda x: x.lower())

    source_list = name_list + [""]

    for a in ARCHITECTURES:
        if a.startswith("SD"):
            source_list += ["NEW " + a]


def find_source(source):
    """
    Find the file path of a model given its name. The name must be present 
    in the global list of source names. If the file does not exist, returns None.

    Args:
    source (str): The name of the model.

    Returns:
    str or None: The absolute path to the model file, or None if the file does not exist.
    """
    if not source:
        return None
    index = source_list.index(source)
    if index >= len(file_list):
        return None
    if not os.path.exists(file_list[index]):
        return None
    return file_list[index]


def get_name(tm: ToolkitModel, arch):
    """
    Create a filename for a model based on its original name and analysis 
    results. The new name includes model metrics and may include a special 
    extension depending on the model architecture.

    Args:
    tm (ToolkitModel): A ToolkitModel object containing the analysis results 
                       for the model.
    arch (str): The architecture of the model.

    Returns:
    str: The generated filename.
    """
    name = "model"
    if tm.filename:
        name = os.path.basename(tm.filename).rsplit(".", 1)[0]

    if tm.m_str:
        if "SD" in arch:
            name += "-" + tm.m_str.replace("/", "-")
        else:
            m = tm.m_str.split("/")
            if "UNET" in arch:
                name += "-" + m[0]
            elif "VAE" in arch:
                name += "-" + m[1]
            elif "CLIP" in arch:
                name += "-" + m[2]

    if arch in COMPONENT_EXT:
        name += COMPONENT_EXT[arch]
    else:
        name += ".safetensors"
    return name


def do_load(source, precision):
    """
    Loads the model given the source and precision.

    Args:
        source (str): The path or identifier to the model file to be loaded.
        precision (str): The precision setting for the model.

    Returns:
        list: A list of updates for the front-end.
    """
    global loaded

    basic_report, adv_report, save_name, error = "", "", "", ""

    dont_half = "FP32" in precision
    keep_ema = False

    loaded = None

    if source.startswith("NEW "):
        loaded = ToolkitModel()
        model_type = source[4:]
        for clss in ARCHITECTURES[model_type]["classes"]:
            loaded.a_classes[clss] = []
        loaded.a_type = model_type
        loaded.a_resolved = {model_type: loaded.a_classes}
        loaded.a_found = loaded.a_resolved
        loaded.a_potential = loaded.a_resolved

        loaded.partial = True
    else:
        filename = find_source(source)
        if not filename:
            error = f"Cannot find {source}!"
        else:
            model, _ = load(filename)
            renamed, broken = fix_model(
                model, fix_clip=shared.opts.model_toolkit_fix_clip)
            loaded = do_analysis(model)
            loaded.renamed = renamed
            loaded.broken = broken
            loaded.fix_clip = shared.opts.model_toolkit_fix_clip
            loaded.filename = filename
    if loaded:
        basic_report = do_basic_report(loaded, dont_half, keep_ema)
        adv_report = do_adv_report(loaded)
        save_name = get_name(loaded, loaded.a_type)

    if error:
        error = f"### ERROR: {error}\n----"

    reports = [gr.update(value=basic_report), gr.update(value=adv_report)]
    sources = [gr.update(), gr.update()]
    drops = [gr.update(choices=[], value="") for _ in range(3)]
    rows = [gr.update(visible=not loaded), gr.update(visible=not not loaded)]
    names = [gr.update(value=save_name), gr.update()]
    error = [gr.update(value=error), gr.update(visible=not not error)]

    if loaded and loaded.a_found:
        drop_arch_list = sorted(loaded.a_potential)
        drops = [gr.update(choices=drop_arch_list, value=loaded.a_type)]

        drop_class_list = sorted(loaded.a_found[loaded.a_type].keys())
        drops += [gr.update(choices=drop_class_list, value=drop_class_list[0])]

        drop_comp_list = ["auto"] + \
            sorted(loaded.a_found[loaded.a_type][drop_class_list[0]])
        drops += [gr.update(choices=drop_comp_list, value="auto")]

    updates = reports + sources + drops + rows + names + error
    return updates


def do_select(drop_arch, drop_class, drop_comp):
    """
    Selects specific architecture, class and component of the model.

    Args:
        drop_arch (str): The architecture of the model to be selected.
        drop_class (str): The class of the model to be selected.
        drop_comp (str): The component of the model to be selected.

    Returns:
        list: A list of updates for the front-end.
    """
    global loaded
    if not loaded:
        return [gr.update(choices=[], value="") for _ in range(3)] + [gr.update(value="")]

    arch_list = sorted(loaded.a_potential)
    if not drop_arch in arch_list:
        drop_arch = loaded.a_type

    arch = loaded.a_found[drop_arch]

    class_list = sorted(arch.keys())
    if not drop_class in class_list:
        drop_class = class_list[0]

    comps = arch[drop_class]

    comp_list = ["auto"] + sorted(comps)
    if not drop_comp in comp_list:
        drop_comp = comp_list[0]

    export_name = get_name(loaded, drop_class)

    updates = [
        gr.update(choices=arch_list, value=drop_arch),
        gr.update(choices=class_list, value=drop_class),
        gr.update(choices=comp_list, value=drop_comp),
        gr.update(value=export_name)
    ]

    return updates


def do_clear():
    """
    Clears the currently loaded model.

    Returns:
        list: A list of updates for the front-end.
    """
    global loaded
    loaded = None

    reports = [gr.update(value=""), gr.update(value="")]
    sources = [gr.update(), gr.update()]
    drops = [gr.update(choices=[], value="") for _ in range(3)]
    rows = [gr.update(visible=True), gr.update(visible=False)]
    names = [gr.update(value=""), gr.update()]
    error = [gr.update(value=""), gr.update(visible=False)]

    updates = reports + sources + drops + rows + names + error
    return updates


def do_refresh():
    """
    Refreshes the source list and name list of the models.

    Returns:
        list: A list of updates for the front-end.
    """
    get_lists()
    return [gr.update(choices=source_list, value=source_list[0]), gr.update(choices=name_list, value=name_list[0])]


def do_report(precision):
    """
    Generates a basic and advanced report for the loaded model based on precision.

    Args:
        precision (str): The precision setting for the model.

    Returns:
        list: A list of updates for the front-end.
    """
    dont_half = "FP32" in precision
    keep_ema = False

    basic_report = do_basic_report(loaded, dont_half, keep_ema)
    adv_report = do_adv_report(loaded)
    save_name = get_name(loaded, loaded.a_type)

    values = [basic_report, adv_report, save_name]

    return [gr.update(value=v) for v in values]


def do_save(save_name, precision):
    """
    Saves the currently loaded model with the given save name and precision.

    Args:
        save_name (str): The name to save the model as.
        precision (str): The precision setting for the model.

    Returns:
        list: A list of updates for the front-end.
    """
    dont_half = "FP32" in precision
    keep_ema = False

    folder = COMPONENT_SAVE_PATH
    if "SD" in loaded.a_type:
        folder = MODEL_SAVE_PATH
    elif "VAE" in loaded.a_type:
        folder = VAE_SAVE_PATH

    filename = os.path.join(folder, save_name)
    model = copy.deepcopy(loaded.model)

    prune_model(model, loaded.a_resolved, keep_ema, dont_half)

    error = ""

    if model:
        save(model, METADATA, filename)
    else:
        error = "### ERROR: Model is empty!\n----"
    del model

    reports = [gr.update(), gr.update()]
    sources = [gr.update(), gr.update()]
    drops = [gr.update() for _ in range(3)]
    rows = [gr.update(), gr.update()]
    names = [gr.update(), gr.update()]
    error = [gr.update(value=error), gr.update(visible=not not error)]

    updates = reports + sources + drops + rows + names + error
    return updates


def do_export(drop_arch, drop_class, drop_comp, export_name, precision):
    """
    Exports a specific architecture, class and component of the model with given export name and precision.

    Args:
        drop_arch (str): The architecture of the model to be exported.
        drop_class (str): The class of the model to be exported.
        drop_comp (str): The component of the model to be exported.
        export_name (str): The name to export the model as.
        precision (str): The precision setting for the model.

    Returns:
        list: A list of updates for the front-end.
    """
    dont_half = "FP32" in precision
    error = ""

    if not loaded or not loaded.model:
        error = "### ERROR: Model is empty!\n----"
    else:
        comp = drop_comp

        if comp == "auto":
            comp = resolve_class(loaded.a_found[drop_arch][drop_class])
            if comp:
                comp = comp[0]

        if comp:
            if not contains_component(loaded.model, comp):
                error = f"### ERROR: Model doesnt contain a {comp}!\n----"

            model = build_fake_model(loaded.model)
            prefixed = ARCHITECTURES[drop_arch]["prefixed"]
            prefix = COMPONENTS[comp]["prefix"]

            extract_component(model, comp, prefixed)

            for k in model:
                kk = prefix + k if prefixed else k
                model[k] = loaded.model[kk]
                if not dont_half and type(model[k]) == torch.Tensor and model[k].dtype == torch.float32:
                    model[k] = model[k].half()

            if "EMA" in comp:
                fix_ema(model)

            folder = COMPONENT_SAVE_PATH
            if "VAE" in comp:
                folder = VAE_SAVE_PATH

            filename = os.path.join(folder, export_name)
            save(model, {}, filename)
        else:
            error = f"### ERROR: Model doesnt contain a {drop_class}!\n----"

    updates = [gr.update() for _ in range(4)] + \
        [gr.update(value=error), gr.update(visible=not not error)]
    return updates


def do_import(drop_arch, drop_class, drop_comp, import_drop, precision):
    """
    Imports a specific architecture, class and component of the model from a specified source.

    Args:
        drop_arch (str): The architecture of the model to be imported.
        drop_class (str): The class of the model to be imported.
        drop_comp (str): The component of the model to be imported.
        import_drop (str): The source of the model to be imported.
        precision (str): The precision setting for the model.

    Returns:
        list: A list of updates for the front-end.
    """
    global loaded
    error = ""

    if not loaded or not import_drop:
        error = "### ERROR: No model is loaded!\n----"

    if not error:
        filename = find_source(import_drop)
        model, _ = load(filename)
        fix_model(model, fix_clip=shared.opts.model_toolkit_fix_clip)
        found, _ = inspect_model(model, all=True)
        if not found or not model:
            error = "### ERROR: Imported model could not be identified!\n----"

    choosen = ""
    if not error:
        # find all the components in the class
        possible = find_components(found, drop_class)
        if not possible:
            error = f"### ERROR: Imported model does not contain a {drop_class}!\n----"
        else:
            # figure which to choose
            if drop_comp == "auto":
                # pick the best component
                choosen = resolve_class(possible)[0]
            else:
                # user specified
                if not drop_comp in possible:
                    error = f"### ERROR: Imported model does not contain a {drop_comp}!\n----"
                else:
                    choosen = drop_comp

    reports = [gr.update(), gr.update()]
    names = [gr.update(), gr.update()]

    if not error:
        # delete the other conflicting components
        delete_class(loaded.model, drop_arch, drop_class)

        extract_component(model, choosen)

        replace_component(loaded.model, drop_arch, model, choosen)

        # update analysis
        filename = loaded.filename

        old = loaded
        loaded = do_analysis(loaded.model)
        loaded.filename = filename
        loaded.broken = old.broken
        loaded.renamed = old.renamed
        loaded.fix_clip = old.fix_clip
        if set(loaded.a_resolved.keys()) != set(old.a_resolved.keys()):
            loaded.a_found = old.a_found
            loaded.a_resolved = old.a_resolved
            loaded.a_type = old.a_type
            loaded.a_potential = old.a_potential
        old = None

        # update reports and names
        result = do_report(precision)
        reports = [result[0], result[1]]
        names[0] = result[2]
        names[1] = get_name(loaded, drop_class)

    sources = [gr.update(), gr.update()]
    drops = [gr.update() for _ in range(3)]
    rows = [gr.update(), gr.update()]
    error = [gr.update(value=error), gr.update(visible=not not error)]

    updates = reports + sources + drops + rows + names + error
    return updates


def on_ui_tabs():
    """
    This method constructs the UI tabs for the model toolkit.

    This function defines the user interface of the model toolkit, providing tabs for 
    different functionalities, such as Basic and Advanced functionalities. It includes 
    controls for loading, saving, clearing and refreshing models. It also enables 
    importing and exporting components, as well as selecting and working with 
    different architectural components.

    It dynamically generates the UI elements, including dropdowns, buttons, and textboxes, 
    and binds them with respective functions for handling user interactions.
    """
    get_lists()
    css = """
        .float-text { float: left; } .float-text-p { float: left; line-height: 2.5rem; } #mediumbutton { max-width: 32rem; } #smalldropdown { max-width: 2rem; } #smallbutton { max-width: 2rem; }
        #toolbutton { max-width: 8em; } #toolsettings > div > div { padding: 0; } #toolsettings { gap: 0.4em; } #toolsettings > div { border: none; background: none; gap: 0.5em; }
        #reportmd { padding: 1rem; } .dark #reportmd thead { color: #daddd8 } .gr-prose hr { margin-bottom: 0.5rem } #reportmd ul { margin-top: 0rem; margin-bottom: 0rem; } #reportmd li { margin-top: 0rem; margin-bottom: 0rem; }
        .dark .gr-compact { margin-left: unset }
        #errormd { min-height: 0rem; text-align: center; } #errormd h3 { color: #ba0000; }
    """
    with gr.Blocks(css=css, analytics_enabled=False, variant="compact") as checkpoint_toolkit:
        gr.HTML(value=f"<style>{css}</style>")
        with gr.Row() as load_row:
            source_dropdown = gr.Dropdown(
                label="Source", choices=source_list, value=source_list[0], interactive=True)
            load_button = gr.Button(value='Load', variant="primary")
            load_refresh_button = gr.Button(
                elem_id="smallbutton", value="Refresh")
        with gr.Row(visible=False) as save_row:
            save_name = gr.Textbox(label="Name", interactive=True)
            prec_dropdown = gr.Dropdown(elem_id="smalldropdown", label="Precision", choices=[
                                        "FP16", "FP32"], value="FP16", interactive=True)
            save_button = gr.Button(value='Save', variant="primary")
            clear_button = gr.Button(elem_id="smallbutton", value="Clear")
            save_refresh_button = gr.Button(
                elem_id="smallbutton", value="Refresh")
        with gr.Row(visible=False) as error_row:
            error_md = gr.Markdown(elem_id="errormd", value="")
        with gr.Tab("Basic"):
            with gr.Column(variant="compact"):
                basic_report_md = gr.Markdown(elem_id="reportmd", value="")
        with gr.Tab("Advanced"):
            with gr.Column(variant="panel"):
                gr.HTML(value='<h1 class="gr-button-lg float-text">Component</h1><p class="float-text-p"><i>Select a component class or specific component.</i></p>')
                with gr.Row():
                    arch_dropdown = gr.Dropdown(
                        label="Architecture", choices=[], interactive=True)
                    class_dropdown = gr.Dropdown(
                        label="Class", choices=[], interactive=True)
                    comp_dropdown = gr.Dropdown(
                        label="Component", choices=[], interactive=True)
                gr.HTML(
                    value='<h1 class="gr-button-lg float-text">Action</h1><p class="float-text-p"><i>Replace or save the selected component.</i></p>')
                with gr.Row():
                    import_dropdown = gr.Dropdown(
                        label="File", choices=name_list, value=name_list[0], interactive=True)
                    import_button = gr.Button(
                        elem_id="smallbutton", value='Import')
                    export_name = gr.Textbox(label="Name", interactive=True)
                    export_button = gr.Button(
                        elem_id="smallbutton", value='Export')
            with gr.Row(variant="compact"):
                adv_report_md = gr.Markdown(elem_id="reportmd", value="")

        reports = [basic_report_md, adv_report_md]
        sources = [source_dropdown, import_dropdown]
        drops = [arch_dropdown, class_dropdown, comp_dropdown]
        rows = [load_row, save_row]
        error = [error_md, error_row]
        names = [save_name, export_name]

        everything = reports + sources + drops + rows + names + error

        load_button.click(fn=do_load, inputs=[
                          source_dropdown, prec_dropdown], outputs=everything)
        clear_button.click(fn=do_clear, inputs=[], outputs=everything)
        load_refresh_button.click(fn=do_refresh, inputs=[], outputs=[
                                  source_dropdown, import_dropdown])
        save_refresh_button.click(fn=do_refresh, inputs=[], outputs=[
                                  source_dropdown, import_dropdown])
        prec_dropdown.change(fn=do_report, inputs=[
                             prec_dropdown], outputs=reports)
        arch_dropdown.change(fn=do_select, inputs=drops,
                             outputs=drops + [export_name])
        class_dropdown.change(fn=do_select, inputs=drops,
                              outputs=drops + [export_name])
        comp_dropdown.change(fn=do_select, inputs=drops,
                             outputs=drops + [export_name])
        save_button.click(fn=do_save, inputs=[
                          save_name, prec_dropdown], outputs=everything)

        export_button.click(fn=do_export, inputs=drops +
                            [export_name, prec_dropdown], outputs=drops + [export_name] + error)
        import_button.click(fn=do_import, inputs=drops +
                            [import_dropdown, prec_dropdown], outputs=everything)

    return (checkpoint_toolkit, "Toolkit", "checkpoint_toolkit"),


def on_ui_settings():
    """
    This method configures the UI settings for the model toolkit.

    This function sets up the options available in the settings of the model toolkit. 
    Currently, it allows the user to toggle whether to fix broken CLIP position IDs and 
    whether to enable Autopruning.
    """
    section = ('model-toolkit', "Model Toolkit")
    shared.opts.add_option("model_toolkit_fix_clip", shared.OptionInfo(
        False, "Fix broken CLIP position IDs", section=section))
    shared.opts.add_option("model_toolkit_autoprune", shared.OptionInfo(
        False, "Enable Autopruning", section=section))


script_callbacks.on_ui_settings(on_ui_settings)

script_callbacks.on_ui_tabs(on_ui_tabs)

load_components(os.path.join(sys.path[0], "components"))