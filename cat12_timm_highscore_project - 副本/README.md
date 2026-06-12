
# Cat12 timm 开源高分模型版

这版不是手写 CNN，而是直接调用开源 `timm` 预训练模型。

推荐模型：

1. `hf_hub:timm/tf_efficientnetv2_s.in21k_ft_in1k`
2. `hf_hub:timm/convnextv2_tiny.fcmae_ft_in22k_in1k_384`
3. `hf_hub:timm/swin_base_patch4_window12_384.ms_in22k_ft_in1k`，这个更大，4060 可能 batch_size 要调小

## 数据目录

把下面三个文件放到 `data/`：

- cat_12_train.zip
- cat_12_test.zip
- train_list.txt

## 安装

建议 Python 3.10/3.11。

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm scikit-learn pillow tqdm numpy
```

## 推荐冲分命令

### EfficientNetV2-S，推荐先跑这个

```bash
python train_timm_kfold.py --data_dir data --work_dir work --model hf_hub:timm/tf_efficientnetv2_s.in21k_ft_in1k --img_size 384 --batch_size 8 --folds 5 --epochs 18 --freeze_epochs 2
```

如果显存够：

```bash
python train_timm_kfold.py --data_dir data --work_dir work --model hf_hub:timm/tf_efficientnetv2_s.in21k_ft_in1k --img_size 384 --batch_size 12 --folds 5 --epochs 20 --freeze_epochs 2
```

### ConvNeXtV2 Tiny，第二个可以试

```bash
python train_timm_kfold.py --data_dir data --work_dir work --model hf_hub:timm/convnextv2_tiny.fcmae_ft_in22k_in1k_384 --img_size 384 --batch_size 8 --folds 5 --epochs 18 --freeze_epochs 2
```

## 输出

最终提交：

```text
work/result.csv
```

格式已经固定为：

```text
图片名.jpg,类别编号
```

没有表头，没有 `cat_12_test/` 前缀。
