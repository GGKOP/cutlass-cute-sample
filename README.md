# cute-practice
- 主要是进行cute相关语法练习和一些矩阵加速到flash attention V2
## 环境要求
- 最新版本的pycorch和cuda 
- cute和cutlass全都放在了3rd文件夹下 
## 启动
### 步骤1: 更新库
```bash
./update.sh
```
### 步骤2: 编译和运行
```bash
make + name.cu
./name
```