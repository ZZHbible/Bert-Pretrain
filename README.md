## Bert情感二分类 单机多卡训练方法

#### 使用如下命令运行

``` shell
!python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 bert.py 
```

上述命令行参数nproc_per_node表示每个节点需要创建多少个进程(使用几个GPU就创建几个)；nnodes表示使用几个节点，因为我们是做单机多核训练，所以设为1。



## 感谢

https://github.com/jia-zhuang/pytorch-multi-gpu-training