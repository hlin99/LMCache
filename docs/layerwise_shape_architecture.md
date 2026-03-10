# Layerwise Shape Architecture - Visual Explanation

## Architecture Comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NON-LAYERWISE (Batch Mode)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Single Memory Object for ALL layers                               │
│  ┌───────────────────────────────────────────────────────────┐    │
│  │  Shape: [2, num_layers, num_tokens, hidden_dim]          │    │
│  │         └┬┘ └────┬────┘ └────┬─────┘ └──────┬──────┘    │    │
│  │          │       │            │               │            │    │
│  │      K/V dims  Layer      Token          Feature          │    │
│  │                dimension   dimension      dimension        │    │
│  │                                                            │    │
│  │  Example: [2, 32, 1000, 4096]                            │    │
│  │           4D - includes layer dimension                   │    │
│  └───────────────────────────────────────────────────────────┘    │
│                                                                     │
│  Memory Format: KV_2LTD (4D)                                       │
│  Storage: All layers in one blob                                   │
│  Pros: Simple structure                                            │
│  Cons: Large memory footprint, no streaming                        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      LAYERWISE Mode                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Multiple Memory Objects - ONE per layer                           │
│                                                                     │
│  Layer 0:                                                          │
│  ┌────────────────────────────────────────────────┐               │
│  │ Shape: [num_tokens, 2, hidden_dim]            │               │
│  │         └────┬─────┘ └┬┘ └──────┬──────┘     │               │
│  │         Token dim  K/V   Feature dim          │               │
│  │ Example: [1000, 2, 4096]                      │               │
│  │          3D - NO layer dimension               │               │
│  │ Key: {layer_id: 0, ...}                       │               │
│  └────────────────────────────────────────────────┘               │
│                                                                     │
│  Layer 1:                                                          │
│  ┌────────────────────────────────────────────────┐               │
│  │ Shape: [num_tokens, 2, hidden_dim]            │               │
│  │ Example: [1000, 2, 4096]                      │               │
│  │ Key: {layer_id: 1, ...}                       │               │
│  └────────────────────────────────────────────────┘               │
│                                                                     │
│  Layer 2...31: (similar)                                           │
│                                                                     │
│  Memory Format: KV_T2D (3D)                                        │
│  Storage: Each layer separate                                      │
│  Pros: Memory efficient, supports streaming                        │
│  Cons: More memory objects to manage                               │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow with Shape Transformation

```
┌──────────────────────────────────────────────────────────────────┐
│  1. GPU Memory (vLLM Paged Format)                              │
│     Per-layer tensors: [2, num_blocks, block_size, ...]         │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│  2. GPU Connector Extract (VLLMPagedMemLayerwiseGPUConnector)   │
│     - Extract layer by layer                                     │
│     - Reshape to token-major format                              │
│     - get_shape() returns: [num_tokens, 2, hidden_dim]  ← 3D!   │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│  3. Memory Allocation (GPUMemoryAllocator)                       │
│     - Allocate buffer with 3D shape                              │
│     - Create MemoryObj with metadata:                            │
│       * shapes = [torch.Size([num_tokens, 2, hidden_dim])]       │
│       * fmt = MemoryFormat.KV_T2D                                │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│  4. Remote Storage Put (RemoteBackend)                           │
│     - Serialize memory_obj for remote storage                    │
│     - memory_obj.get_shapes() returns 3D shapes                  │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│  5. Protocol Serialization (RemoteMetadata)                      │
│                                                                  │
│     OLD CODE (FAILS):                                            │
│     ┌────────────────────────────────────────────────────┐      │
│     │ for shape in shapes:                               │      │
│     │   assert len(shape) == 4  ← FAILS for 3D shape! ❌│      │
│     └────────────────────────────────────────────────────┘      │
│                                                                  │
│     NEW CODE (WORKS):                                            │
│     ┌────────────────────────────────────────────────────┐      │
│     │ for shape in shapes:                               │      │
│     │   padded = _pad_shape_to_4d(shape)  ✓              │      │
│     │   # [num_tokens, 2, hidden_dim]                    │      │
│     │   # becomes [1, num_tokens, 2, hidden_dim]         │      │
│     │   params.extend([s0, s1, s2, s3])                  │      │
│     └────────────────────────────────────────────────────┘      │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│  6. Network Transmission                                         │
│     - Binary struct: [length, fmt, dtype, s0, s1, s2, s3]       │
│     - Always 4D in wire format                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Shape Padding Examples

```
Input Shape (3D)          Padding Process              Output Shape (4D)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[100, 2, 4096]      →   prepend [1]           →   [1, 100, 2, 4096]
                        ▲
                        └─ Added dimension


[50, 8192]          →   prepend [1, 1]        →   [1, 1, 50, 8192]
                        ▲   ▲
                        └───┴─ Two added dimensions


[1000]              →   prepend [1, 1, 1]     →   [1, 1, 1, 1000]
                        ▲   ▲   ▲
                        └───┴───┴─ Three added dimensions


[2, 32, 100, 4096]  →   no padding            →   [2, 32, 100, 4096]
                        (already 4D)
```

## Why This Design?

### Layerwise Benefits

```
┌──────────────────────────────────────────────────────────────┐
│  Scenario: 32-layer model, 10K tokens                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Non-Layerwise:                                              │
│  ╔════════════════════════════════════════╗                 │
│  ║ [2, 32, 10000, 4096]                  ║                 │
│  ║ Memory: ~10 GB (all at once)          ║                 │
│  ╚════════════════════════════════════════╝                 │
│    ▲                                                         │
│    └─ Must allocate entire buffer upfront                   │
│                                                              │
│  Layerwise:                                                  │
│  ┌────────────────┐                                         │
│  │[10000, 2, 4096]│  Layer 0 (~320 MB)                      │
│  └────────────────┘                                         │
│  ┌────────────────┐                                         │
│  │[10000, 2, 4096]│  Layer 1 (~320 MB)                      │
│  └────────────────┘                                         │
│  ...                                                         │
│    ▲                                                         │
│    └─ Can load/unload layers on demand                      │
│                                                              │
│  Benefits:                                                   │
│  ✓ Lower peak memory usage                                  │
│  ✓ Supports streaming/pipelining                            │
│  ✓ Better cache locality per layer                          │
│  ✓ Flexible layer-wise operations                           │
└──────────────────────────────────────────────────────────────┘
```

## Code References

```python
# File: lmcache/v1/gpu_connector/gpu_connectors.py

class VLLMPagedMemLayerwiseGPUConnector:
    def get_shape(self, num_tokens: int) -> torch.Size:
        if self.use_mla:
            return torch.Size([num_tokens, self.hidden_dim_size])  # 2D
        else:
            return torch.Size([num_tokens, 2, self.hidden_dim_size])  # 3D
            #                  └────┬────┘ └┬┘ └────────┬──────────┘
            #                  tokens    K/V    features
            #
            #  Note: NO layer dimension - layer info in key!

# File: lmcache/v1/protocol.py

def _pad_shape_to_4d(shape: torch.Size) -> torch.Size:
    """Automatically pad lower-dimensional shapes to 4D for serialization."""
    if len(shape) < 4:
        padding = [1] * (4 - len(shape))
        return torch.Size(padding + list(shape))
    return shape
    #
    #  This bridge function allows 3D layerwise shapes
    #  to work with 4D protocol requirements!
```
