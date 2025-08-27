from src.core.llama import *
from src.core.model import *


def create_batch_fingerprint(batch):
    def prefix_suffix_only(array, prefix=3, suffix=3):
        prefix_part = array[:prefix]
        suffix_part = array[-suffix:]
        result = prefix_part + suffix_part
        return result

    first_row = prefix_suffix_only(batch[0]).numpy().tolist()
    middle_row = prefix_suffix_only(batch[len(batch) // 2]).numpy().tolist()
    last_row = prefix_suffix_only(batch[-1]).numpy().tolist()

    return first_row + middle_row + last_row

def get_classes_from_globals(names):
    # print("global names========================================")#dev
    # print(names)#dev
    return [globals().get(name) for name in names]


def wrap_model_fsdp(model, fsdp_config):
    classes_to_wrap = get_classes_from_globals(fsdp_config.modules_to_wrap)
    print(f"Wrapping model with classes: {classes_to_wrap}")
    igonore_mixed_precision_classes = get_classes_from_globals(
        fsdp_config.mixed_precision.ignored_classes
    )
    print(f"Ignoring mixed precision for classes: {igonore_mixed_precision_classes}")
    mixed_precision_dtype = getattr(
        sys.modules["torch"], fsdp_config.mixed_precision.dtype
    )
    print(f"Using mixed precision dtype: {mixed_precision_dtype}")

    wrapped_model = FSDP(
        model,
        device_id=int(os.environ["RANK"]),
        mixed_precision=MixedPrecision(
            param_dtype=mixed_precision_dtype,
            cast_forward_inputs=True,
            _module_classes_to_ignore=igonore_mixed_precision_classes,
        ),
        auto_wrap_policy=ModuleWrapPolicy(classes_to_wrap),
    )
    return wrapped_model

def wrap_model_distributed(model, distributed_config):
    if distributed_config is not None:
        if torch.cuda.is_available():
            model = wrap_model_fsdp(model, distributed_config.fsdp)
        else:
            logger.info("FSDP is not supported with CPU. Running DDP instead")
            model = DDP(model)
    return model