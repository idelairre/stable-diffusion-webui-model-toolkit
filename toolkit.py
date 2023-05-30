import torch
import safetensors
import safetensors.torch
import os
import copy

from constants import ARCHITECTURES, COMPONENTS, COMPONENT_CLASS, EMA_PREFIX


def tensor_size(t):
    """
    Get the size of the tensor in bytes.
    
    Args:
    t (torch.Tensor): Tensor for which to compute size.
    
    Returns:
    int: Size of tensor in bytes. Returns 0 if input is not a tensor.
    """
    if type(t) == torch.Tensor:
        return t.nelement() * t.element_size()
    return 0


def tensor_shape(key, data):
    """
    Determine the shape of tensor data. If the key is in a predefined list of keys, the shape may be altered.

    Args:
    key (str): Identifier for the tensor.
    data: Tensor data.

    Returns:
    tuple: The tensor shape.
    """
    if hasattr(data, 'shape'):
        shape = tuple(data.shape)
        for c in ["LoRA-v1-UNET", "LoRA-v1-CLIP"]:
            if key in COMPONENTS[c]['shapes']:
                lora_shape = COMPONENTS[c]['shapes'][key]
                if len(shape) == len(lora_shape):
                    shape = tuple(a if b != -1 else b for a,
                                  b in zip(shape, lora_shape))
        return shape
    return tuple()


def load_components(path):
    """
    Load component definitions from a specified path.

    Args:
    path (str): Path to the directory containing component definition files.
    """
    for c in COMPONENTS:
        file = os.path.join(path, COMPONENTS[c]["source"])
        if not os.path.exists(file):
            print(f"CANNOT FIND {c} KEYS")
        with open(file, 'r') as f:
            COMPONENTS[c]["keys"] = set()
            for l in f:
                l = l.rstrip().split(" ")
                k, z = l[0], l[1]
                z = z[1:-1].split(",")
                if not z[0]:
                    z = tuple()
                else:
                    z = tuple(int(i) for i in z)
                COMPONENTS[c]["keys"].add((k, z))
                if "shapes" in COMPONENTS[c]:
                    COMPONENTS[c]["shapes"][k] = z


def get_prefixed_keys(component):
    """
    Retrieve a set of keys from a component, with each key being prefixed.

    Args:
    component (str): Identifier of the component from which to retrieve keys.
    
    Returns:
    set: A set of keys, each being prefixed.
    """
    prefix = COMPONENTS[component]["prefix"]
    allowed = COMPONENTS[component]["keys"]
    return set([(prefix + k, z) for k, z in allowed])


def get_keys_size(model, keys):
    """
    Computes the total size in bytes of the specified keys in a model.

    Args:
    model (dict): A dictionary representing the model.
    keys (list): List of keys to calculate their total size.

    Returns:
    int: Total size of the specified keys in the model.
    """
    z = 0
    for k in keys:
        if k in model:
            z += tensor_size(model[k])
    return z


class FakeTensor():
    """
    A mock tensor class, providing a simplified stand-in for PyTorch tensor objects.
    """
    def __init__(self, shape):
        self.shape = shape


def build_fake_model(model):
    """
    Constructs a fake model using the input model's keys and the shapes of its tensors.

    Args:
    model (dict): A dictionary representing the model.

    Returns:
    dict: A dictionary representing the fake model, containing the same keys as the input model and FakeTensor objects as values.
    """
    fake_model = {}
    for k in model:
        fake_model[k] = FakeTensor(tensor_shape(k, model[k]))
    return fake_model


def inspect_model(model, all=False):
    """
    Inspects a model's architecture, identifying its components and any issues.

    Args:
    model (dict): The model to be inspected.
    all (bool, optional): If True, the function returns both found and rejected components.

    Returns:
    dict: Information about the model's architecture.
    """
    
    # find all arch's and components in the model
    # also reasons for failing to find them

    keys = set([(k, tensor_shape(k, model[k])) for k in model])

    rejected = {}

    components = []  # comp -> prefixed
    classes = {}  # class -> [comp]
    for comp in COMPONENTS:
        required_keys_unprefixed = COMPONENTS[comp]["keys"]
        required_keys_prefixed = get_prefixed_keys(comp)
        missing_unprefixed = required_keys_unprefixed.difference(keys)
        missing_prefixed = required_keys_prefixed.difference(keys)

        if not missing_unprefixed:
            components += [(comp, False)]
        if not missing_prefixed:
            components += [(comp, True)]

        if missing_prefixed and missing_unprefixed:
            if missing_prefixed != required_keys_prefixed:
                rejected[comp] = rejected.get(
                    comp, []) + [{"reason": f"Missing required keys ({len(missing_prefixed)} of {len(required_keys_prefixed)})", "data": list(missing_prefixed)}]

            if missing_unprefixed != required_keys_unprefixed:
                rejected[comp] = rejected.get(
                    comp, []) + [{"reason": f"Missing required keys ({len(missing_unprefixed)} of {len(required_keys_unprefixed)})", "data": list(missing_unprefixed)}]
        else:
            clss = COMPONENT_CLASS[comp]
            classes[clss] = [comp] + classes.get(clss, [])

    found = {}  # arch -> {class -> [comp]}
    for arch in ARCHITECTURES:
        needs_prefix = ARCHITECTURES[arch]["prefixed"]
        required_classes = set(ARCHITECTURES[arch]["classes"])
        required_keys = set(ARCHITECTURES[arch]["required"])

        if not required_keys.issubset(keys):
            missing = required_keys.difference(keys)
            if missing != required_keys:
                rejected[arch] = rejected.get(
                    arch, []) + [{"reason": f"Missing required keys ({len(missing)} of {len(required_keys)})", "data": list(missing)}]
            continue

        found_classes = {}
        for clss in required_classes:
            if clss in classes:
                for comp in classes[clss]:

                    # or ((comp, not needs_prefix) in components and not needs_prefix):
                    if (comp, needs_prefix) in components:
                        found_classes[clss] = found_classes.get(clss, [])
                        found_classes[clss] += [comp]
                    # else:
                    #    rejected[arch] = rejected.get(arch, []) + [{"reason": "Class has incorrect prefix", "data": [clss]}]

        found_class_names = set(found_classes.keys())
        if not required_classes.issubset(found_class_names):
            if found_class_names:
                missing = list(required_classes.difference(found_class_names))
                rejected[arch] = rejected.get(
                    arch, []) + [{"reason": "Missing required classes", "data": missing}]
            continue

        found[arch] = found_classes

    # if we found a real architecture then dont show the broken ones
    if any([a.startswith("SD-") for a in found]):
        for a in list(found.keys()):
            if a.endswith("-BROKEN"):
                del found[a]

    for arch in list(found.keys()):
        if "LoRA" in arch:
            for clss in found[arch]:
                if len(found[arch][clss]) == 2:
                    found[arch][clss] = [found[arch][clss]
                                         [0].replace("-v1-", "-v1A-")]

    if "LoRA-v1" in found:
        del found["LoRA-v1-UNET"]
        del found["LoRA-v1-CLIP"]

    if all:
        return found, rejected
    else:
        return resolve_arch(found)


def resolve_class(components):
    """
    Resolves a list of components down to a single component, if necessary.

    Args:
    components (list): A list of components to be resolved.

    Returns:
    list: A list containing either the single resolved component, or the original list of components.
    """
    components = list(components)

    if not components or len(components) == 1:
        return components

    # prefer SD components vs busted ass components
    sd_components = [c for c in components if "SD" in c]
    if len(sd_components) == 1:
        return [sd_components[0]]

    # otherwise component with the most keys is probably the best
    components = sorted(components, key=lambda c: len(
        COMPONENTS[c]["keys"]), reverse=True)

    return [components[0]]


def resolve_arch(arch):
    """
    Resolves potentially many overlapping architectures to a single one.

    Args:
    arch (dict): Dictionary of architectures to be resolved.

    Returns:
    dict: Dictionary with a single architecture after resolution.
    """
    arch = copy.deepcopy(arch)
    # resolve potentially many overlapping arch's to a single one

    if not arch:
        return {}

    # select arch with most keys
    arch_sizes = {}
    for a in arch:
        arch_sizes[a] = len(ARCHITECTURES[a]["required"])
        for clss in arch[a]:
            arch[a][clss] = resolve_class(arch[a][clss])
            if arch[a][clss]:
                arch_sizes[a] += len(COMPONENTS[arch[a][clss][0]]["keys"])
    for normal in ["SD-v1", "SD-v2"]:
        if normal in arch_sizes:
            choosen = normal
            break
    else:
        choosen = max(arch_sizes, key=arch_sizes.get)
    return {choosen: arch[choosen]}


def find_components(arch, component_class):
    """
    Find components of a specific class within a model architecture.

    Args:
    arch (dict): The architecture of the model.
    component_class (str): The class of components to find.

    Returns:
    set: The set of components of the specified class in the architecture.
    """
    components = set()
    for a in arch:
        if component_class in arch[a]:
            components.update(arch[a][component_class])
    return components


def contains_component(model, component, prefixed=None):
    """
    Checks whether a model contains a specific component.

    Args:
    model (dict): The model to be checked.
    component (str): The component to look for in the model.
    prefixed (bool, optional): Specifies whether to look for the component name as a prefix.

    Returns:
    bool: True if the model contains the component, False otherwise.
    """
    model_keys = set([(k, tensor_shape(k, model[k])) for k in model])

    allowed = False
    if prefixed == None:  # prefixed or unprefixed
        allowed = get_prefixed_keys(component).issubset(model_keys)
        allowed = allowed or COMPONENTS[component]["keys"].issubset(model_keys)
    elif prefixed == True:
        allowed = get_prefixed_keys(component).issubset(model_keys)
    elif prefixed == False:
        allowed = COMPONENTS[component]["keys"].issubset(model_keys)

    return allowed


def get_allowed_keys(arch, allowed_classes=None):
    """
    Determines the allowed keys for a given architecture.
    
    Args:
        arch (dict): A dictionary representing the architecture to be analyzed.
        allowed_classes (list, optional): A list of classes that are permitted. If None, all classes are allowed.
        
    Returns:
        set: A set of keys that are allowed within the architecture.
    """
    
    # get all allowed keys
    allowed = set()
    for a in arch:
        if allowed_classes == None:
            allowed.update(ARCHITECTURES[a]["required"])
            allowed.update(ARCHITECTURES[a]["optional"])
        prefixed = ARCHITECTURES[a]["prefixed"]
        for clss in arch[a]:
            if allowed_classes == None or clss in allowed_classes:
                for comp in arch[a][clss]:
                    comp_keys = COMPONENTS[comp]["keys"]
                    if prefixed:
                        comp_keys = get_prefixed_keys(comp)
                    allowed.update(comp_keys)
    return allowed


def fix_model(model, fix_clip=False):
    """
    Fixes certain keys in the provided model dictionary.
    
    Args:
        model (dict): A dictionary representing the model.
        fix_clip (bool, optional): If True, fix the ids for the clip model. Default is False.
        
    Returns:
        tuple: A tuple containing lists of renamed keys and broken keys in the model.
    """
    
    # fix NAI nonsense
    
    nai_keys = {
        'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
        'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
        'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.'
    }
    renamed = []
    for k in list(model.keys()):
        for r in nai_keys:
            if type(k) == str and k.startswith(r):
                kk = k.replace(r, nai_keys[r])
                renamed += [(k, kk)]
                model[kk] = model[k]
                del model[k]
                break

    # fix merging nonsense
    i = "cond_stage_model.transformer.text_model.embeddings.position_ids"
    broken = []
    if i in model:
        correct = torch.Tensor([list(range(77))]).to(torch.int64)
        current = model[i].to(torch.int64)

        broken = correct.ne(current)
        broken = [i for i in range(77) if broken[0][i]]

        if fix_clip:
            # actually fix the ids
            model[i] = correct
        else:
            # ensure fp16 looks the same as fp32
            model[i] = current

    return renamed, broken


def fix_ema(model):
    """
    Converts a UNET-v1-EMA model into a UNET-v1-SD model, but only when in component form (unprefixed).
    
    Args:
        model (dict): A dictionary representing the model.
        
    Returns:
        None.
    """
    
    # turns UNET-v1-EMA into UNET-v1-SD
    # but only when in component form (unprefixed)

    # example keys
    # EMA = model_ema.diffusion_modeloutput_blocks91transformer_blocks0norm3weight
    # SD  = model.diffusion_model.output_blocks9.1.transformer_blocks.0.norm3.weight

    normal = COMPONENTS["UNET-v1-SD"]["keys"]
    for k, _ in normal:
        kk = k.replace(".", "")
        if kk in model:
            model[k] = model[kk]
            del model[kk]


def compute_metric(model, arch=None):
    """
    Computes a metric score for a given model based on the architecture.
    
    Args:
        model (dict): A dictionary representing the model.
        arch (dict, optional): A dictionary representing the architecture. If None, the architecture will be inspected from the model.
        
    Returns:
        tuple: A tuple containing the metric as a string and a tuple of individual scores for unet, vae and clip respectively.
    """
    
    def tensor_metric(t):
        t = t.to(torch.float16).to(torch.float32)
        return torch.sum(torch.sigmoid(t)-0.5)

    if arch == None:
        arch = inspect_model(model)

    unet_keys = get_allowed_keys(
        arch, ["UNET-v1", "UNET-v1-Pix2Pix", "UNET-v2", "UNET-v2-Depth"])
    vae_keys = get_allowed_keys(arch, ["VAE-v1"])
    clip_keys = get_allowed_keys(arch, ["CLIP-v1", "CLIP-v2"])

    unet, vae, clip = 0, 0, 0

    is_clip_v1 = "CLIP-v1" in next(iter(arch.values()))

    for k in model:
        kk = (k, tensor_shape(k, model[k]))

        if kk in unet_keys:
            unet += tensor_metric(model[k])

        if kk in vae_keys:
            if "encoder." in k or "decoder." in k:
                vae += tensor_metric(model[k])

        if kk in clip_keys:
            if "mlp." in k and not ".23." in k:
                clip += tensor_metric(model[k])

    b_unet, b_vae, b_clip = -6131.5400, 17870.7051, - \
        2097.8596 if is_clip_v1 else -8757.5630
    k_unet, k_vae, k_clip = 10000, 10000, 1000000 if is_clip_v1 else 10000

    r = 10000

    n_unet = int(abs(unet/b_unet - 1) * k_unet)
    n_vae = int(abs(vae/b_vae - 1) * k_vae)
    n_clip = int(abs(clip/b_clip - 1) * k_clip)

    while n_unet >= r:
        n_unet -= r//2

    while n_vae >= r:
        n_vae -= r//2

    while n_clip >= r:
        n_clip -= r//2

    s_unet = f"{n_unet:04}" if unet != 0 else "----"
    s_vae = f"{n_vae:04}" if vae != 0 else "----"
    s_clip = f"{n_clip:04}" if clip != 0 else "----"

    n_unet = None if unet == 0 else n_unet
    n_vae = None if vae == 0 else n_vae
    n_clip = None if clip == 0 else n_clip

    return s_unet+"/"+s_vae+"/"+s_clip, (n_unet, n_vae, n_clip)


def load(file):
    """
    Loads a model and its metadata from a file.
    
    Args:
        file (str): The file path from which the model will be loaded.
        
    Returns:
        tuple: A tuple containing a dictionary of the model and a dictionary of the metadata.
    """
    
    model = {}
    metadata = {}

    if file.endswith(".safetensors") or file.endswith(".st"):
        model = safetensors.torch.load_file(file, device="cpu")
    else:
        model = torch.load(file, map_location="cpu")
        if not model:
            return {}, {}
        if 'state_dict' in model:
            for k in model:
                if k != 'state_dict':
                    metadata[k] = model[k]
            model = model['state_dict']

    return model, metadata


def save(model, metadata, file):
    """
    Saves a model and its metadata to a file.
    
    Args:
        model (dict): A dictionary representing the model to be saved.
        metadata (dict): A dictionary containing the metadata of the model.
        file (str): The file path where the model will be saved.
        
    Returns:
        None.
    """
    
    if file.endswith(".safetensors"):
        safetensors.torch.save_file(model, file)
        return
    else:
        out = metadata
        out['state_dict'] = model
        torch.save(out, file)


def prune_model(model, arch, keep_ema, dont_half):
    """
    Prunes the model based on the allowed keys in the given architecture.
    
    Args:
        model (dict): A dictionary representing the model to be pruned.
        arch (dict): A dictionary representing the architecture of the model.
        keep_ema (bool): If True, keeps the keys that start with EMA_PREFIX in the model.
        dont_half (bool): If True, ensures that the model tensors are of float32 type. If False, converts the tensors to float16.
        
    Returns:
        None.
    """
    
    allowed = get_allowed_keys(arch)
    for k in list(model.keys()):
        kk = (k, tensor_shape(k, model[k]))
        keep = False
        if kk in allowed:
            keep = True
        if k.startswith(EMA_PREFIX) and keep_ema:
            keep = True
        if not keep:
            del model[k]
            continue
        if type(model[k]) == torch.Tensor:
            if dont_half and model[k].dtype in {torch.float16, torch.float64, torch.bfloat16}:
                model[k] = model[k].to(torch.float32)
            if not dont_half and model[k].dtype in {torch.float32, torch.float64, torch.bfloat16}:
                model[k] = model[k].to(torch.float16)


def extract_component(model, component, prefixed=None):
    """
    Extracts a specific component from the model.

    Args:
        model (dict): A dictionary representing the model.
        component (str): The component to extract from the model.
        prefixed (bool, optional): Whether to consider only the keys that start with the component prefix. 
            If None, both prefixed and non-prefixed keys are considered.

    Returns:
        None. The model is modified in-place.
    """
    
    prefix = COMPONENTS[component]["prefix"]
    allowed = set()
    if prefixed != True:
        allowed = allowed.union(COMPONENTS[component]["keys"])
    if prefixed != False:
        allowed = allowed.union(get_prefixed_keys(component))

    for k in list(model.keys()):
        z = tensor_shape(k, model[k])
        if (k, z) in allowed:
            if k.startswith(prefix):
                kk = k.replace(prefix, "")
                if kk != k:
                    model[kk] = model[k]
                    del model[k]
        else:
            del model[k]


def replace_component(target, target_arch, source, source_component):
    """
    Replaces a component of the target model with a component from the source model.

    Args:
        target (dict): A dictionary representing the target model.
        target_arch (str): The architecture of the target model.
        source (dict): A dictionary representing the source model.
        source_component (str): The component from the source model to be copied to the target.

    Raises:
        ValueError: If the source_component is not in the target_arch architecture.

    Returns:
        None. The target model is modified in-place.
    """
    
    if not COMPONENT_CLASS[source_component] in ARCHITECTURES[target_arch]["classes"]:
        raise ValueError(f"{target_arch} cannot contain {source_component}!")

    # get component for class
    prefix = COMPONENTS[source_component]["prefix"]
    component_keys = COMPONENTS[source_component]["keys"]

    # find out if we should prefix the component
    is_prefixed = ARCHITECTURES[target_arch]["prefixed"]

    for k in list(source.keys()):
        src_z = tensor_shape(k, source[k])
        src_k = k[len(prefix):] if k.startswith(prefix) else k
        dst_k = prefix + k if is_prefixed else k
        if (src_k, src_z) in component_keys:
            target[dst_k] = source[k]


def delete_class(model, model_arch, component_class):
    """
    Deletes a specific class of components from the model.

    Args:
        model (dict): A dictionary representing the model.
        model_arch (str): The architecture of the model.
        component_class (str): The class of components to delete from the model.

    Returns:
        None. The model is modified in-place.
    """
    
    keys = set([(k, tensor_shape(k, model[k])) for k in model])
    prefixed = ARCHITECTURES[model_arch]["prefixed"]

    for name, component in COMPONENTS.items():
        if COMPONENT_CLASS[name] != component_class:
            continue
        component_keys = component["keys"] if not prefixed else get_prefixed_keys(
            name)
        for k in component_keys:
            if k in keys:
                del model[k[0]]
                keys.remove(k)


def log(model, file):
    """
    Logs the keys and shapes of the tensors in the model to a file.

    Args:
        model (dict): A dictionary representing the model.
        file (str): The file path where the log will be saved.

    Returns:
        None.
    """
    
    keys = []
    for k in model:
        size = str(list(model[k].shape))
        keys += [f"{k},{size}"]
    keys.sort()
    out = "\n".join(keys)
    with open(file, "w") as f:
        f.write(out)


if __name__ == '__main__':
    load_components("components")

    for l in ["instruct-pix2pix-00-22000.safetensors"]:
        a, _ = load(l)
        for k in sorted(list(a.keys())):
            print(k, tensor_shape(k, a[k]))
