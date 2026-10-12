# 本地采集端与服务器滴定定位 API

该接口用于完成以下闭环：

```text
本地采集 RGB + aligned depth + 内参 + 拍照时 T_world_ee
  -> HTTP 上传到服务器
  -> 服务器排队运行 YOLO / affordance / SAM3D / 三维投影
  -> 返回胶头中心和烧杯口中心的世界坐标
  -> 本地自动下载 JSON / TXT / PNG / PLY 结果包
```

服务端使用单工作线程队列，一次只运行一个完整任务，避免多个 SAM3D 进程抢占同一张 GPU。

## 1. 通信架构

推荐使用 SSH 隧道，API 只监听服务器回环地址：

```text
本地采集程序
  -> http://127.0.0.1:8000
  -> SSH tunnel
  -> 服务器 127.0.0.1:8000
  -> 任务队列
  -> run_titration_two_object.sh
```

这种方式不需要在服务器公开 API 端口。HTTP 请求仍需要 Bearer token。

## 2. 服务器启动 API

首先确认手眼标定文件的方向已确定，且 JSON 中的状态为：

```json
{
  "status": "confirmed",
  "T_ee_cam": [[...], [...], [...], [...]]
}
```

定义：

```text
p_ee = T_ee_cam * p_opencv_camera
```

启动：

```bash
cd /home/dsj/code/language_affordance_grasp

export LAG_API_TOKEN='REPLACE_WITH_A_LONG_RANDOM_TOKEN'
export LAG_EYE_IN_HAND_CALIBRATION=/home/dsj/code/language_affordance_grasp/configs/T_ee_cam.json

./scripts/start_titration_api.sh
```

默认监听：

```text
127.0.0.1:8000
```

可选环境变量：

```bash
export LAG_API_HOST=127.0.0.1
export LAG_API_PORT=8000
export LAG_API_RUN_ROOT=/home/dsj/code/language_affordance_grasp/runs/api_jobs
```

不要使用多个 Uvicorn worker；模型任务需要在同一个服务进程内排队。

### 标定保护

如果标定 JSON 包含：

```json
"status": "direction_assumed_pending_confirmation"
```

API 会拒绝启动。确认方向后才可将它改成 `confirmed`。

## 3. 本地建立 SSH 隧道

在本地机器保持该命令运行：

```bash
ssh -N -L 8000:127.0.0.1:8000 USER@SERVER
```

建立后，本地程序通过以下地址访问服务器：

```text
http://127.0.0.1:8000
```

如果本地 8000 端口被占用：

```bash
ssh -N -L 18080:127.0.0.1:8000 USER@SERVER
```

然后客户端使用 `http://127.0.0.1:18080`。

## 4. 将客户端脚本拷贝到本地

本地执行：

```bash
scp USER@SERVER:/home/dsj/code/language_affordance_grasp/scripts/titration_client.py .
```

客户端需要 Python `requests`：

```bash
python -m pip install requests
```

## 5. 本地上传、等待并下载结果

本地采集程序每次需要提供：

- RGB 图像；
- 对齐到 RGB 的原始深度 PNG；
- 相机内参 JSON；
- 拍照瞬间的 `T_world_ee` JSON。

一条命令完成上传、轮询、坐标打印和结果下载：

```bash
export LAG_API_TOKEN='SAME_TOKEN_AS_SERVER'

python titration_client.py run \
  --server http://127.0.0.1:8000 \
  --rgb ./capture_001/rgb.png \
  --depth ./capture_001/depth.png \
  --camera ./capture_001/camera.json \
  --world-ee ./capture_001/T_world_ee.json \
  --depth-scale 0.001 \
  --capture-id capture_001 \
  --output-dir ./capture_001/server_result
```

如果需要指定服务器物理 GPU：

```bash
--gpu 1
```

不传 `--gpu` 时，服务器 bash 脚本会选择空闲显存最多的 GPU。

默认下载精简结果包。如需下载所有中间产物：

```bash
--scope all
```

## 6. 客户端收到的坐标

完成时客户端会打印：

```json
{
  "job_id": "...",
  "units": "meters",
  "frame": "world",
  "dropper_bulb_center_world": [0.1, 0.2, 0.3],
  "beaker_mouth_center_world": [0.4, 0.5, 0.6],
  "dropper_bulb_center_pixel_uv": [440.0, 90.0],
  "beaker_mouth_center_pixel_uv": [290.0, 95.0]
}
```

世界坐标单位是米。本地机械臂程序可以读取这两个字段：

```text
dropper_bulb_center_world
beaker_mouth_center_world
```

客户端同时会下载：

```text
titration_<job_id>_results.zip
```

默认结果包包含：

- `04_targets/titration_targets.json`；
- `04_targets/target_centers_world.txt`；
- `04_targets/target_centers_overlay.png`；
- `04_targets/titration_scene_world.ply`；
- `04_targets/targets_only_world.ply`；
- 两个物体的模型/深度点云；
- YOLO overlay；
- affordance overlay；
- SAM3D overlay；
- `pipeline.log`。

## 7. 独立查询和下载

查询任务：

```bash
python titration_client.py status \
  --server http://127.0.0.1:8000 \
  --token "$LAG_API_TOKEN" \
  JOB_ID
```

重新下载结果：

```bash
python titration_client.py download \
  --server http://127.0.0.1:8000 \
  --token "$LAG_API_TOKEN" \
  JOB_ID \
  --scope results \
  --output-dir ./server_result
```

## 8. HTTP API

| Method | Path | 用途 |
|---|---|---|
| `GET` | `/health` | 健康检查 |
| `POST` | `/v1/titration/jobs` | 上传数据并提交任务 |
| `GET` | `/v1/titration/jobs/{job_id}` | 查询队列/运行状态 |
| `GET` | `/v1/titration/jobs/{job_id}/result` | 获取两个世界坐标 |
| `GET` | `/v1/titration/jobs/{job_id}/bundle?scope=results` | 下载精简结果包 |
| `GET` | `/v1/titration/jobs/{job_id}/bundle?scope=all` | 下载全部中间产物 |

POST 使用 `multipart/form-data`：

| Field | Type | Required | 说明 |
|---|---|---|---|
| `rgb` | file | yes | RGB PNG/JPG |
| `depth` | file | yes | aligned depth PNG |
| `camera` | file | yes | 内参 JSON |
| `world_ee` | file | yes | 拍照时 `T_world_ee` JSON |
| `depth_scale` | float | no | 默认 `0.001` |
| `gpu` | int | no | 物理 GPU 索引 |
| `capture_id` | string | no | 本地采集编号 |

所有请求都必须携带：

```http
Authorization: Bearer <LAG_API_TOKEN>
```

## 9. 任务状态

```text
uploading
  -> queued
  -> running
  -> completed
```

失败状态：

```text
failed
interrupted
```

- `failed`：输入检查、YOLO、SAM3D 或最终导出失败；
- `interrupted`：API 服务在任务排队或运行期间被重启，需要本地重新提交。

## 10. 安全与真机要求

1. 必须先确认手眼标定方向是 `T_ee_cam`。
2. `T_world_ee` 必须与 RGB-D 拍摄时刻同步。
3. API token 不要写入 Git 仓库或采集数据目录。
4. 优先使用 SSH 隧道，不要将无 TLS 的 HTTP API 暴露到公网。
5. 本地程序收到坐标后，应先检查 overlay/PLY 和工作空间，不要立即执行机械臂运动。
6. 机械臂第一次使用新标定时必须低速、安全高度和 dry-run。

## 11. 相关文件

- 服务端 API：`src/language_affordance_grasp/titration_api.py`
- API 启动脚本：`scripts/start_titration_api.sh`
- 本地客户端：`scripts/titration_client.py`
- 完整视觉流水线：`scripts/run_titration_two_object.sh`
- 本地采集/文件传输说明：`README_LOCAL_CAPTURE_AND_TRANSFER.md`
