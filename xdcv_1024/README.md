# XDCV-1024 网页展示平台

`XDCV-1024` 是本地视觉推理 demo 平台，当前整合两个功能：

- 图生文：调用 `image_caption_infer/local_infer_seal.py`，使用 Qwen2-VL-2B + 本地 SEAL LoRA adapter。
- 印花图像提取：调用 `itr_pe`，输出灰度 mask 和 RGBA 印花抠图。

平台使用 Python 标准库实现 Web 服务，不需要额外安装 FastAPI/Flask。

## 启动

```bash
cd /home/xd/lj
source .fic/bin/activate
python xdcv_1024/server.py --host 127.0.0.1 --port 7860
```

浏览器打开：

```text
http://127.0.0.1:7860
```

如果通过 SSH 访问远程服务器，可以使用 VS Code 端口转发或 SSH 本地转发。

## 公网和手机访问

公网访问前建议启用登录保护：

```bash
export XDCV_AUTH_USER=xdcv
export XDCV_AUTH_PASSWORD='换成一个强密码'
```

绑定所有网卡：

```bash
python xdcv_1024/server.py --host 0.0.0.0 --port 7860
```

如果服务器有公网 IP，并且防火墙/安全组已放行 `7860`，手机浏览器访问：

```text
http://服务器公网IP:7860
```

如果服务器没有公网 IP，或在校园网/公司网/NAT 后面，需要使用反向隧道、VPN、内网穿透或部署到有公网 IP 的机器上，再把流量转发到本服务。

公网暴露时不要把服务裸奔到互联网上；至少设置 `XDCV_AUTH_USER` 和 `XDCV_AUTH_PASSWORD`，并优先通过 HTTPS 反向代理访问。

## 环境变量

通用设备：

```bash
export XDCV_DEVICE=cuda
```

图生文可选配置：

```bash
export XDCV_CAPTION_BASE_MODEL=/home/xd/models/Qwen2-VL-2B-Instruct
export XDCV_CAPTION_LOCAL_FILES_ONLY=1
export XDCV_CAPTION_DEVICE=cuda
```

印花提取可选配置：

```bash
export XDCV_PRINT_DEVICE=cuda
```

## 输出目录

每次网页推理会生成一个独立任务目录：

```text
xdcv_1024/runs/<job_id>/
  input/
  result/
```

图生文返回文本结果。印花提取返回：

```text
result/mask.png     # 灰度 mask
result/cutout.png   # RGBA 印花抠图
```

## 接口

健康检查：

```bash
curl http://127.0.0.1:7860/api/health
```

印花提取测试：

```bash
curl -s \
  -F mode=print \
  -F input_size=512 \
  -F threshold=127 \
  -F image=@/home/xd/lj/demo/demo14.jpg \
  http://127.0.0.1:7860/api/infer
```

图生文测试：

```bash
curl -s \
  -F mode=caption \
  -F max_new_tokens=128 \
  -F image=@/home/xd/lj/demo/demo14.jpg \
  http://127.0.0.1:7860/api/infer
```
