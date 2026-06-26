# XDCV-1024

本项目是一个本地视觉推理 demo 平台，包含两个功能：

- 图生文：Qwen2-VL-2B + 本地 SEAL LoRA adapter。
- 印花图像提取：IS-Net 生成灰度 mask，并输出 RGBA 印花抠图。

## 公开仓库说明

本公开仓库中暂未上传模型推理权重。

后续将会提供 demo 入口。

## 效果展示

| 印花图像提取 | 图生文推理 |
| --- | --- |
| ![XDCV-1024 印花图像提取效果](demo1.png) | ![XDCV-1024 图生文推理效果](demo2.png) |

| 印花图像提取 | 图生文推理 |
| --- | --- |
| ![XDCV-1024 印花图像提取效果](demo4.png) | ![XDCV-1024 图生文推理效果](demo3.png) |

## 目录

```text
image_caption_infer/   # 图生文推理脚本、依赖说明
itr_pe/                # 印花图像提取脚本、依赖说明
xdcv_1024/             # 网页展示平台
demo/                  # 示例图片
```

## 启动网页平台

```bash
cd /home/xd/lj
source .fic/bin/activate
python xdcv_1024/server.py --host 0.0.0.1 --port 7860
```

后台服务开启状态下，校园网内设备可通过浏览器登录访问（账号密码、IP 及端口联系本人）：

```text
http://<校园网内网IP>:<端口>
```

## 模型推理权重

本公开仓库中暂未上传模型推理权重，包括：

- `image_caption_infer/fixed_reward_seed2026_20260515_184503/adapter_model.safetensors`
- `itr_pe/weights/*.pth`

如后续上传大模型/推理权重，需要使用 Git LFS。
