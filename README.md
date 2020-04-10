## Platform: Ubuntu16.04
## Install OpenVINO: [OpenVINO开发必备知识点——环境搭建](https://www.toutiao.com/i6813281706506715651/)
## InsightFace OpenVINO Model:

- 链接:https://pan.baidu.com/s/1mtsMXN8889nBRaNRo9VlKQ  密码:h77z
- 解压并保存至model目录下

## Usage:

```
source /opt/intel/openvino/setupvars.sh
cd insightface-openvino
mkdir build
cd build
cmake ..
make
cd ..
./bin/demo model/model-0000.xml CPU image/test.jpg
```
