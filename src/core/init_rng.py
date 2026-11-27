"""Model initialization that no config switch can perturb.

Every random draw at model construction used to come from the one global stream,
in build order. So whatever changed how much one part drew -- the KV shift's two
coefficient draws, a layer's attention type, a gate -- moved the stream under
everything built after it: with the shift on, a block's feed-forward, every later
block and the head started from other weights than with it off. Two rules now make
each part's init a function of the run's seed and the part's name, and of nothing
else (documents/notes/archive/review_issues.md, N6):

1. `component_init_rng(name)`: a component -- the token embedding, each block's
   attention and feed-forward, the output head -- is built from its own seed,
   derived from the run's seed (`torch.initial_seed()`, which main.py sets with
   torch.manual_seed just before the model is built) and the component's name.
   The global stream is restored afterwards. So adding, removing or changing one
   component leaves every other one exactly as it was, and the global stream
   after the model is built does not depend on the model.
2. `private_generator(tag)` / `private_rand(..., tag=)`: an optional parameter
   inside a component (the KV shift's coefficients, the ST gate) draws from a
   generator of its own, seeded from the component stream's current state and a
   tag, without advancing the stream. Turning it on or off leaves the component's
   other parameters as they were.

main.py builds the model on the CPU, so the CPU stream carries this. Under a CUDA
default device (the GPU tests build that way) the default CUDA generator is seeded
and restored per component too.
"""

import hashlib
from contextlib import contextmanager

import torch


def _derived_seed(base: bytes, tag: str) -> int:
    digest = hashlib.blake2b(base + b"\x1f" + tag.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _default_cuda_index():
    """The index of the default device if it is a CUDA device, else None."""
    device = torch.empty(0).device
    if device.type != "cuda":
        return None
    return device.index if device.index is not None else torch.cuda.current_device()


@contextmanager
def component_init_rng(name: str):
    """Build one component from its own seed, then restore the global stream."""
    seed = _derived_seed(str(torch.initial_seed()).encode(), name)
    cuda = _default_cuda_index()
    devices = [] if cuda is None else [cuda]
    with torch.random.fork_rng(devices=devices):
        torch.default_generator.manual_seed(seed)
        if cuda is not None:
            torch.cuda.default_generators[cuda].manual_seed(seed)
        yield


def private_generator(tag: str) -> torch.Generator:
    """A CPU generator for one optional parameter: seeded from the global CPU
    stream's current state and `tag`, without advancing the stream. Within a
    component that state is fixed by the component's seed and its always-present
    draws, so the generator is too."""
    state = torch.random.get_rng_state().numpy().tobytes()
    return torch.Generator().manual_seed(_derived_seed(state, tag))


def private_rand(*shape, tag: str, dtype=torch.float32) -> torch.Tensor:
    """`torch.rand(*shape)` from private_generator(tag), on the default device:
    drawn on the CPU, where the generator lives, then moved."""
    draw = torch.rand(
        *shape, generator=private_generator(tag), dtype=dtype, device="cpu"
    )
    return draw.to(torch.empty(0).device)
