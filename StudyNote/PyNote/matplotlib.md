# cn

```python
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
```



# plt.scatter

```python
import matplotlib.pyplot as plt
plt.scatter(x,y,s=20, c='b', marker = '0', cmap = None, norm = None, vmin = None, alpha = None, linewidths = None, verts = None, hold = None, **kwargs)

plt.figure(figsize=(10,10))
plt.scatter(np_df[0,0], np_df[0,1], color = 'r',marker ='o')
plt.scatter(np_df[1:,0], np_df[1:,1],marker ='o')
for i in range(np_df.shape[0]):
    t_x, t_y = float(np_df[i,0]) + 0.000001, float(np_df[i, 1]) + 0.0003
    plt.annotate(str(i), xy = (np_df[i,0], np_df[i, 1]), xytext = (t_x, t_y))

#plt.colorbar()
plt.grid(True)
plt.xlabel('经度')
plt.ylabel('纬度')
plt.savefig('sensor_sit.png')
plt.show()
```

