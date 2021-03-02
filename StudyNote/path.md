```C
C:\Program Files (x86)\Common Files\MVS\Runtime\Win32_i86;C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common;C:\Program Files (x86)\MVS\Applications\Win64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin;D:\Microsoft VS Code\bin;D:\anaconda3\Library\bin;

```



```c
#C/C++ -> 附加包含目录
$(MVCAM_COMMON_RUNENV)\Includes;$(MELSEC)\INCLUDE;$(OPENCV342)\include\opencv2;$(OPENCV342)\include\opencv;$(OPENCV342)\include;D:\anaconda3\envs\torch-15\include;D:\anaconda3\envs\torch-15\Lib\site-packages\numpy\core\include
    

#链接器->附加库目录
$(MVCAM_COMMON_RUNENV)\Libraries\win64;$(MELSEC)\Lib\x64;$(OPENCV342)\lib;D:\anaconda3\envs\torch-15\libs;D:\anaconda3\envs\torch-15\Lib\site-packages\numpy\core\lib

#附加依赖项
MvCameraControl.lib;MdFunc32.lib;opencv_world342.lib;python37.lib;npymath.lib    
```

