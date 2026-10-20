# 用户问题

将指定 CTA 和线程打印文件名、行号的代码写成宏。

# 完整回答

在头文件中添加：

```cpp
#ifndef LVZ_DBG_PRINT_LINE
#define LVZ_DBG_PRINT_LINE()                                              \
  do {                                                                    \
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&            \
        threadIdx.x == 0) {                                               \
      printf("[LVZ_DBG] %s:%d\n", __FILE__, __LINE__);                      \
    }                                                                     \
  } while (0)
#endif
```

在设备代码中调用：

```cpp
LVZ_DBG_PRINT_LINE();
```

`__LINE__` 输出的是**宏调用位置的行号**。每次执行到该调用点时，仅 CTA `(0,0,0)` 的线程 0 打印；若调用点位于该线程不会进入的 warp 分支中，则不会输出。

# 验证与未完成事项

提供可复制宏，未修改目标源码，未编译或 GPU 运行验证。本地保存；跨电脑同步未完成。

