# torch

## torch.clamp

```python
#将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。
# (n1, n2, 2)
intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0) 
torch.clamp(input, min, max, out=None) → Tensor
```



## torch.cuda.is_available

```python
#cuda是否可用；
torch.cuda.is_available()

# 返回gpu数量；
torch.cuda.device_count()

# 返回gpu名字，设备索引默认从0开始；
torch.cuda.get_device_name(0)

# 返回当前设备索引
torch.cuda.current_device()
```