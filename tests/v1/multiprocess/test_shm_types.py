# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.v1.multiprocess.shm_types import ShmSlotDescriptor


def test_shm_slot_descriptor_round_trip() -> None:
    """Round-trip SHM slot descriptors through the shared dict schema."""
    descriptor = ShmSlotDescriptor(
        offset=128,
        length=256,
        shape=[2, 4, 8],
        dtype="bfloat16",
    )

    payload = descriptor.to_dict()

    assert payload == {
        "offset": 128,
        "length": 256,
        "shape": [2, 4, 8],
        "dtype": "bfloat16",
    }
    assert ShmSlotDescriptor.from_dict(payload) == descriptor
