# 数据集外部交接说明

## 为什么图片不在 GitHub 中

完整项目数据和环境超过 7 GB，其中训练/验证/测试图片约 3.8 GB。Git 仓库只保存可审计、可版本化的标签、配置、清单和代码；RGB 图片通过加密网盘、移动硬盘或学校服务器单独交接。

## 正式数据版本

- 数据集名称：`grading_four_class_v3`
- 图像总数：1,211
- 实例总数：5,371
- 数据划分：train 966、val 116、test 129
- YOLO 配置：`data/grading_four_class_v3/data.yaml`
- 类别定义：`data/grading_four_class_v3/classes.txt`

## 接收方目录结构

数据包解压后应形成：

```text
data/grading_four_class_v3/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── labelme_json/
├── audit/
├── classes.txt
└── data.yaml
```

仓库已经提供除 `images/` 外的内容。复制图片时不要覆盖或重新生成现有标签。

## 数据包制作要求

交接方应为图片数据包记录以下信息：

```text
文件名：wolfberry_grading_four_class_v3_images.zip
版本：v3
制作日期：YYYY-MM-DD
文件大小：待填写
SHA256：待填写
下载位置：待填写
```

建议将上述信息补充到本文件后再提交。接收方下载后应先验证 SHA256，再执行：

```powershell
python .\scripts\verify_four_class_dataset.py
```

## 使用边界

- 图像只用于本项目研究、模型训练和论文复现，不应在未经数据所有者同意的情况下公开传播。
- `test` 划分是冻结测试集；开发期间只使用 train/val。
- 当前尺寸无毫米标定，不能将像素长度直接解释为实际毫米。
