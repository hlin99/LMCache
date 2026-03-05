# Understanding `align_bytes` in LMCache

## Question (Chinese)
backend.memory_allocator.gpu_allocator.align_bytes这个到底是什么？

## Answer / 答案

### 简介 (Chinese Introduction)

`align_bytes` 是 LMCache 中分页内存分配器的核心概念，它表示**一个完整的 KV 缓存页面的大小（以字节为单位）**。

### What is `align_bytes`?

`align_bytes` is a fundamental concept in LMCache's paged memory allocator. It represents **the size in bytes of one complete KV cache page/block**.

## Technical Details

### Where is it used?

```
backend.memory_allocator.gpu_allocator.align_bytes
```

- `backend.memory_allocator` is an instance of `PagedCpuGpuMemoryAllocator`
- `gpu_allocator` is an instance of `PagedTensorMemoryAllocator` (for GPU memory)
- `align_bytes` is the page size attribute

### How is it calculated?

`align_bytes` is calculated using the `get_size_bytes()` function:

```python
self.align_bytes = get_size_bytes(shapes, dtypes)
```

Where:
- `shapes`: List of KV cache tensor shapes
- `dtypes`: List of corresponding data types

The calculation is:
```python
align_bytes = sum(shape.numel() * dtype.itemsize for shape, dtype in zip(shapes, dtypes))
```

### Example Calculation

For a typical LLM with the following KV cache configuration:
- Shape: `torch.Size([2, 32, 16, 1024])`
  - 2 layers (K and V)
  - 32 attention heads
  - 16 tokens per block
  - 1024 hidden dimension per head
- Dtype: `torch.bfloat16` (2 bytes per element)

Calculation:
```
align_bytes = 2 × 32 × 16 × 1024 × 2 bytes
            = 2,097,152 bytes
            = 2 MB per page
```

### Why is it called "align_bytes"?

The name "align_bytes" comes from the concept of **memory alignment**. In the paged allocator:

1. The entire memory buffer is divided into fixed-size pages
2. Each page size is exactly `align_bytes`
3. All allocations are **aligned** to this page boundary
4. You cannot allocate partial pages - only complete pages

This alignment ensures:
- Efficient memory management
- Compatible with hardware transfer mechanisms (NIXL, RDMA)
- Predictable memory layout for distributed caching

### How is it used in the allocator?

```python
# Split buffer into pages of size align_bytes
self.paged_buffers = torch.split(self.buffer, self.align_bytes, dim=0)

# Each allocation consumes one or more complete pages
# Buffer size must be a multiple of align_bytes (or adjusted downward)
if self.buffer_size % self.align_bytes != 0:
    num_blocks = self.buffer_size // self.align_bytes
    adjusted_size = num_blocks * self.align_bytes
    self.buffer = self.buffer[:adjusted_size]
```

## Relationship to Configuration

When you set `pd_buffer_size` in the LMCache configuration:
- This specifies the total GPU/CPU memory buffer size
- The buffer is divided into `pd_buffer_size // align_bytes` pages
- Each page can store KV cache for one sequence block

### Example
```python
pd_buffer_size = 4,317,511,681 bytes  # User-configured
align_bytes = 8,994,816 bytes          # Calculated from KV shape/dtype

num_pages = 4,317,511,681 // 8,994,816 = 479 pages (with remainder)
actual_buffer_size = 479 × 8,994,816 = 4,308,476,864 bytes
```

The system automatically adjusts the buffer size downward to fit complete pages.

## Summary / 总结

### English
`align_bytes` is the **page size** in LMCache's paged memory allocator. It equals the total bytes needed to store one complete KV cache block, calculated from the model's KV cache shape and data type. The memory buffer is divided into pages of this size, and allocations happen at page granularity.

### 中文
`align_bytes` 是 LMCache 分页内存分配器中的**页面大小**。它等于存储一个完整的 KV 缓存块所需的总字节数，根据模型的 KV 缓存形状和数据类型计算得出。内存缓冲区被划分为这种大小的页面，分配以页面粒度进行。

## Related Files

- `lmcache/v1/memory_management.py`: PagedTensorMemoryAllocator implementation
- `lmcache/integration/vllm/utils.py`: get_size_bytes() function
- `lmcache/v1/storage_backend/pd_backend.py`: Usage in PD backend
