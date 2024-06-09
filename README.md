## conda

1. 创建虚拟环境

```
conda create -n xxx python=3.xx
```

1. 进入虚拟环境

```
conda activate xxx
```

1. 退出虚拟环境

```
conda deactivate
```

1. 退出虚拟环境
```
# 查找package信息
conda search numpy
 
# 安装package
conda install -n xxx numpy
# 如果不用-n指定环境名称，则被安装在当前激活环境
# 也可以通过-c指定通过某个channel安装
 
# 更新package
conda update -n xxx numpy
 
# 删除package
conda remove -n xxx numpy
```

## git 

```
git push origin xxx
```