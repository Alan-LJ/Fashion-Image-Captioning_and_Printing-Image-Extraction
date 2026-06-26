# itr_pe 本地部署与测试说明

`itr_pe` 用于服装印花图像提取，流程分两步：

1. `Inference.py`：使用本地 IS-Net 权重生成印花区域灰度 mask。
2. `apply_mask.py`：把 mask 应用到原图，输出带透明通道的 RGBA 印花抠图 PNG。

推荐直接运行 `extract_print.py`，它会一次性生成并保留两类结果：

- 灰度 mask：`itr_pe/output_detect/*.png`
- RGBA 印花抠图：`itr_pe/masked/*.png`

RGBA 抠图默认会填充灰度 mask 中被前景包围的内部孔洞，避免黑色印花细节被当作透明区域抠掉。

## 目录结构

```text
itr_pe/
  Inference.py
  apply_mask.py
  extract_print.py
  requirements-lj.txt
  models/
  weights/
  input/           # 放待处理原图
  output_detect/   # 保存灰度 mask
  masked/          # 保存 RGBA 印花抠图
```

## 1. 激活虚拟环境

```bash
cd /home/xd/lj
source .fic/bin/activate
```

确认当前 Python 来自 `.fic`：

```bash
which python
```

期望输出：

```text
/home/xd/lj/.fic/bin/python
```

## 2. 安装依赖

当前脚本依赖 `torch`、`torchvision`、`numpy`、`Pillow`、`tqdm`。其中 `torch/torchvision` 需要和服务器 CUDA 匹配，通常已经单独安装。

安装本项目额外依赖：

```bash
python -m pip install --no-cache-dir -r itr_pe/requirements-lj.txt
```

检查依赖：

```bash
python - <<'PY'
import torch, torchvision, numpy, PIL, tqdm
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda:", torch.cuda.is_available())
PY
```

## 3. 准备测试图片

把待测试图片复制到 `itr_pe/input/`：

```bash
cp /home/xd/lj/0bbb948a0876edce2827a164c6783fbe.jpg itr_pe/input/
```

如需重新测试，可以只清空最终抠图；灰度 mask 默认会保留在 `output_detect/`：

```bash
rm -f itr_pe/masked/*.png
```

## 4. 一键生成 mask 和抠图

GPU 推理：

```bash
python itr_pe/extract_print.py --device cuda --input-size 1024
```

如果当前环境没有 GPU，可用 CPU 测试：

```bash
python itr_pe/extract_print.py --device cpu --input-size 512
```

如果确实需要保留 mask 内部孔洞，可以追加：

```bash
--keep-holes
```

默认输出目录：

```text
itr_pe/output_detect/   # 保留灰度 mask
itr_pe/masked/          # 保留 RGBA 印花抠图
```

## 5. 分步运行

如果需要分步调试，可以先生成灰度 mask，再生成 RGBA 印花抠图：

```bash
python itr_pe/Inference.py --device cuda --input-size 1024
python itr_pe/apply_mask.py
```

## 6. 自定义输入输出目录

```bash
python itr_pe/extract_print.py \
  --input-dir /path/to/images \
  --mask-dir /path/to/masks \
  --cutout-dir /path/to/cutouts \
  --device cuda
```

也可以分步自定义：

```bash
python itr_pe/Inference.py \
  --input-dir /path/to/images \
  --output-dir /path/to/masks \
  --device cuda

python itr_pe/apply_mask.py \
  --input-dir /path/to/images \
  --mask-dir /path/to/masks \
  --output-dir /path/to/cutouts
```

## 7. 平台整合接口

后续整合 demo 时，可以直接从 Python 导入函数：

```python
from pathlib import Path
from itr_pe.Inference import run_inference
from itr_pe.apply_mask import apply_masks

mask_paths = run_inference(
    input_dir=Path("/path/to/images"),
    output_dir=Path("/path/to/masks"),
    device_name="cuda",
)

cutout_paths = apply_masks(
    input_dir=Path("/path/to/images"),
    mask_dir=Path("/path/to/masks"),
    output_dir=Path("/path/to/cutouts"),
)
```
