# Qwen2-VL 图生文推理说明

本目录集中保存图生文推理相关文件：

- `local_infer_seal.py`：单图图生文推理脚本。
- `requirements.txt`：除 PyTorch 外的推理依赖。
- `fixed_reward_seed2026_20260515_184503/`：基于 Qwen2-VL-2B 训练得到的 SEAL LoRA adapter。

## 1. 激活环境

从项目根目录执行：

```bash
cd /home/xd/lj
source .fic/bin/activate
```

确认 PyTorch 和 CUDA：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

## 2. 安装依赖

PyTorch 需要按服务器 CUDA 情况单独安装。其它依赖使用本目录的 `requirements.txt`：

```bash
python -m pip install -r image_caption_infer/requirements.txt
```

如果缺少 `torchvision`，按当前环境的 PyTorch CUDA 版本安装。例如当前环境为 `torch==2.12.1+cu130`：

```bash
python -m pip install --no-cache-dir torchvision==0.27.1+cu130 --index-url https://download.pytorch.org/whl/cu130
```

## 3. 运行推理

从项目根目录运行：

```bash
python image_caption_infer/local_infer_seal.py \
  --image /home/xd/lj/0bbb948a0876edce2827a164c6783fbe.jpg \
  --device cuda \
  --max-new-tokens 128
```

也可以进入推理目录运行：

```bash
cd /home/xd/lj/image_caption_infer
python local_infer_seal.py \
  --image /home/xd/lj/0bbb948a0876edce2827a164c6783fbe.jpg \
  --device cuda \
  --max-new-tokens 128
```

默认 adapter 路径为：

```text
/home/xd/lj/image_caption_infer/fixed_reward_seed2026_20260515_184503
```

## 4. 基座模型加载方式

如果不传 `--base-model`，脚本会按顺序查找：

1. adapter 目录下 `run_config.json` 的 `model_id`。
2. adapter 目录下 `adapter_config.json` 的 `base_model_name_or_path`。
3. 默认值 `Qwen/Qwen2-VL-2B-Instruct`。

如果本地已有 Qwen2-VL-2B-Instruct，建议显式指定并禁止联网：

```bash
python image_caption_infer/local_infer_seal.py \
  --base-model /home/xd/models/Qwen2-VL-2B-Instruct \
  --image /home/xd/lj/0bbb948a0876edce2827a164c6783fbe.jpg \
  --device cuda \
  --local-files-only \
  --max-new-tokens 128
```

如果需要通过 SSH 隧道访问 Hugging Face，只在当前 shell 临时设置：

```bash
export http_proxy=http://127.0.0.1:7899
export https_proxy=http://127.0.0.1:7899
```

用完可取消：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```

## 5. 常见问题

检查当前终端是否使用虚拟环境：

```bash
which python
```

期望输出类似：

```text
/home/xd/lj/.fic/bin/python
```

如果报 `ModuleNotFoundError: No module named 'torch'`，通常是没有激活 `.fic`。

如果报 `Qwen2VLVideoProcessor requires the Torchvision library`，说明缺少 `torchvision`。

如果 Hugging Face 下载超时，先确认代理或镜像可用，再重新执行推理命令。模型会写入 Hugging Face 缓存，后续通常不会重复下载。

查看当前目录所在磁盘空间：

```bash
df -h .
```
