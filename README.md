[TOC]

## 韵律训练工具

### 核心模块  
```
../auto-nn/  
|-- auto_train_prosody.py   # 训练、测试主函数  
|-- text_eval.py            # 计算性能评估的指标  
|-- models  
|   |-- basic_model.py      # model基类，定义一些通用函数和属性  
|   |-- prosody_common.py   # 模型训练、预测接口  
|-- data_reader             # 读取数据，为模型训练提供数据  
|   |-- text_reader.py      # 读取txt文件，用于生成训练前端模型的数据  
|   |-- data_feeder.py      # batch生成器，为graph提供输入数据  
|-- common                  # 通用的工具性模块  
|-- scripts                 # 常用的工具性脚本  
|   |-- calc_normalize_params.py    # 计算特征数据的归一化数值  
|   |-- crf_dict_2_tf_dict.py       # crf特征映射词典转化为此训练工具所用的词典格式  
|   |-- split_fea_tar.py            # 拆分文本数据集的特征文本和label文本  
|   |-- get_report_from_log.py      #从训练log中抽取性能指标，转写为csv格式  
|-- unittest                # 功能模块单元测试  
```

### 支持的模型
  
    CBHGModel           --- 继承BasicModel， 支持CBHG模型  
    DilatedCNNModel     --- 继承BasicModel， 支持id_cnn_block  
    MultiSpeakerModel   --- 默认继承DilatedCNNModel, 支持多说话人tag数据(@bianyanyao);可继承于其他模型  

### 数据格式
```  
data_fea:  

特征1 特征2 ... 特征N       （第一句）  
特征1 特征2 ... 特征N  
...  
特征1 特征2 ... 特征N  
                          （空一行表示断句）  
特征1 特征2 ... 特征N           （第二句）  
特征1 特征2 ... 特征N  
...  
data_tar:  

标签1 标签2 ... 标签M1         （第一句）  
标签1 标签2 ... 标签M2         （第二句）  
...  
```
### mapping_dict  
每个特征都需要一个映射词典，每个task的标签都需要一个映射词典, 格式如下：  
```
N  
default_value  
1   symbol_1  
2   symbol_2  
...  
N   symbol_N  
```
注：  
在模型支持multitask的情况下，一个data_fea file可以有多个对应的data_tar file  

---
## 一、 快速开始 
### 1、运行  


### 2、结果查看  
**输出目录：**
```  
${save_tag}/  
|-- *.json                          # 训练配置文件  
|-- model/                          # 存放每一轮的模型  
|-- prediction/                     # 存放开发集/测试集每一轮的预测输出  
|-- report/                         # 存放开发集/测试集每一轮的评估结果  
|   |-- train_report_<tag>.csv      # 每轮训练模型的评估指标  
|   |--  
|-- tb_logs/                        # 存放tensorboard可视化log  
```
## 二、配置文件  
**参考：**  
```
./conf/prosody_basic.json   # 韵律训练,支持通用模型  
./conf/prosody_cbhg.json    # cbhg模型  
./conf/prosody_dilated_cnn.json     # dilated cnn 模型  
./conf/prosody_id_cnn_block.json    # id_cnn_block 
./conf/prosody_multi_speaker_id_cnn_block.json  # 多发音人模型
```
## 三、 自定义模块  
**1、自定义loss函数**  
**2、自定义激活函数**  
**3、自定义模型结构**  
    继承basic_models.py中的BasicModel类即可  

## 四、注意事项  
tensorflow版本 >= 1.0, 默认 v1.3  