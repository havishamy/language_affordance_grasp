# 给本地 Codex 的任务：接入玻璃棒定位结果并完成抓取、竖直调整和烧杯搅拌

本文是服务器与本地机器人端之间的唯一联调契约。请按本文修改本地仓库
`/home/dsj/vla/affordance`，不要修改服务器返回字段，不要把服务器感知结果当作可直接执行的机器人位姿。

## 1. 已实现的服务器能力

服务器新增了一个不运行 SAM3D 的 CPU 流程：

```text
RGB-D + camera.json + T_world_ee.json
  -> YOLO class 12 玻璃棒分割（CPU）
  -> 掩码 PCA 得到二维主轴和两个端点
  -> 端点像素射线与桌面平面求交
  -> T_world_ee * T_ee_cam 转到 UR base/world
  -> 两个对称抓取候选
```

接口由原来的统一服务提供，仍然启动：

```bash
cd /home/dsj/code/language_affordance_grasp
export LAG_API_TOKEN='与本地一致的值'
export LAG_EYE_IN_HAND_CALIBRATION=/home/dsj/code/language_affordance_grasp/configs/T_ee_cam_affordance.json
./scripts/start_complex_demo_api.sh
```

新增路由：

```text
POST /v1/stirring/jobs
GET  /v1/stirring/jobs/{job_id}
GET  /v1/stirring/jobs/{job_id}/result
GET  /v1/stirring/jobs/{job_id}/bundle
```

现有 `/v1/titration/jobs` 和 `/v1/complex-demo/jobs` 不变。

## 2. 使用前提

该版本只适用于：

1. 玻璃棒完整出现在图像中；
2. 玻璃棒平放在桌面上；
3. 深度图已经对齐到 RGB；
4. 上传的是采集时刻的 `T_world_ee = T_base_tool0`；
5. 服务器和本地使用同一份手眼标定；
6. 烧杯在前序倾倒和滴定过程中没有移动，继续使用前序 complex-demo 返回的烧杯口位置。

端点 `A/B` 只是确定性编号，不代表“上端/下端”或“抓取端/搅拌端”。本地选择哪一端抓取，另一端就作为插入烧杯的自由端。

## 3. POST 输入格式

请求：

```http
POST http://127.0.0.1:8000/v1/stirring/jobs
Authorization: Bearer <LAG_API_TOKEN>
Content-Type: multipart/form-data
```

文件字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `rgb` | PNG, uint8, H×W×3 | 当前观察位彩色图 |
| `depth` | PNG, uint16, H×W | 对齐到 RGB 的深度图 |
| `camera` | JSON | 彩色相机内参 |
| `world_ee` | JSON | 当前采集时刻 `T_world_ee` |

普通字段：

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `capture_id` | 必填 | 1～128位安全ID |
| `depth_scale` | `0.001` | 每个深度单位对应的米数 |
| `rod_radius_m` | `0.0025` | 玻璃棒实测半径；实验前修改为真实值 |
| `grasp_inset_m` | `0.025` | 抓取点从端部向内偏移的距离 |
| `min_length_m` | `0.08` | 可接受玻璃棒最小长度 |
| `max_length_m` | `0.40` | 可接受玻璃棒最大长度 |

`camera.json` 至少包含：

```json
{
  "width": 640,
  "height": 480,
  "fx": 607.2,
  "fy": 605.7,
  "cx": 318.6,
  "cy": 246.9,
  "depth_scale_m_per_unit": 0.001
}
```

`T_world_ee.json`：

```json
{
  "T_world_ee": [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ]
}
```

## 4. 提交、轮询和下载示例

本地复用现有 Bearer Token、超时和重试策略。最小 Python 请求逻辑：

```python
from contextlib import ExitStack
from pathlib import Path
import time
import requests


def submit_stirring(server, token, capture_dir, output_dir):
    capture_dir = Path(capture_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"}

    with ExitStack() as stack:
        files = {
            "rgb": ("rgb.png", stack.enter_context(open(capture_dir / "rgb.png", "rb")), "image/png"),
            "depth": ("depth.png", stack.enter_context(open(capture_dir / "depth.png", "rb")), "image/png"),
            "camera": ("camera.json", stack.enter_context(open(capture_dir / "camera.json", "rb")), "application/json"),
            "world_ee": (
                "T_world_ee.json",
                stack.enter_context(open(capture_dir / "T_world_ee.json", "rb")),
                "application/json",
            ),
        }
        data = {
            "capture_id": capture_dir.name,
            "depth_scale": "0.0010000000474974513",
            "rod_radius_m": "0.0025",
            "grasp_inset_m": "0.025",
            "min_length_m": "0.08",
            "max_length_m": "0.40",
        }
        response = requests.post(
            f"{server}/v1/stirring/jobs",
            headers=headers,
            files=files,
            data=data,
            timeout=60,
        )
        response.raise_for_status()
        job_id = response.json()["job_id"]

    while True:
        response = requests.get(
            f"{server}/v1/stirring/jobs/{job_id}", headers=headers, timeout=30
        )
        response.raise_for_status()
        status = response.json()
        if status["state"] == "completed":
            break
        if status["state"] in {"failed", "interrupted"}:
            raise RuntimeError(f"Stirring job failed: {status.get('error')}")
        time.sleep(1.0)

    response = requests.get(
        f"{server}/v1/stirring/jobs/{job_id}/result", headers=headers, timeout=30
    )
    response.raise_for_status()
    result = response.json()
    (output_dir / "result.json").write_text(
        __import__("json").dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
```

POST成功返回：

```json
{
  "job_id": "uuid",
  "state": "queued"
}
```

状态值：

```text
uploading -> queued -> running -> completed
                         |-> failed
```

## 5. result JSON 格式

核心格式如下，数组中的数字均为示意：

```json
{
  "schema_version": "stirring-geometry-v1",
  "job_id": "uuid",
  "capture_id": "capture-id",
  "frame": "world",
  "units": "meters",
  "assumptions": {
    "glass_rod_lies_on_table": true,
    "endpoint_a_b_have_no_task_semantics": true,
    "centerline_is_offset_from_table_by_rod_radius_m": 0.0025
  },
  "world_up": [0.0, 0.0, 1.0],
  "world_down": [0.0, 0.0, -1.0],
  "glass_rod": {
    "class_id": 12,
    "class_name": "glass rod",
    "segmentation_confidence": 0.92,
    "mask_artifact": "00_segmentation/glass_rod/mask.png",
    "mask_probability_artifact": "00_segmentation/glass_rod/mask_prob.png",
    "segmentation_overlay_artifact": "00_segmentation/glass_rod/overlay.png",
    "geometry_overlay_artifact": "glass_rod/geometry_overlay.png",
    "endpoint_a_pixel_uv": [168.0, 221.0],
    "endpoint_b_pixel_uv": [237.0, 436.0],
    "endpoint_a_world": [0.683, -0.089, 0.020],
    "endpoint_b_world": [0.488, -0.066, 0.021],
    "center_world": [0.585, -0.077, 0.020],
    "axis_world_unoriented": [-0.993, 0.117, 0.004],
    "length_m": 0.196,
    "grasp_candidates": [
      {
        "id": "grasp_near_a",
        "grasp_point_world": [0.658, -0.086, 0.020],
        "grasp_point_pixel_uv": [177.0, 248.0],
        "opposite_tip_id": "endpoint_b",
        "opposite_tip_world": [0.488, -0.066, 0.021],
        "rod_direction_grasp_to_tip_world": [-0.993, 0.117, 0.004]
      },
      {
        "id": "grasp_near_b",
        "grasp_point_world": [0.513, -0.069, 0.021],
        "grasp_point_pixel_uv": [228.0, 409.0],
        "opposite_tip_id": "endpoint_a",
        "opposite_tip_world": [0.683, -0.089, 0.020],
        "rod_direction_grasp_to_tip_world": [0.993, -0.117, -0.004]
      }
    ],
    "geometry_source": "mask_pca_plus_table_plane_intersection",
    "quality": {
      "valid": true,
      "mask_area_px": 2849,
      "mask_elongation": 18.3,
      "mask_length_px": 226.0,
      "mask_width_sigma_px": 3.6,
      "table_depth_point_count": 10149,
      "table_plane_inlier_count": 9833,
      "table_plane_inlier_fraction": 0.969,
      "table_plane_rmse_m": 0.0012,
      "endpoint_uncertainty_m": 0.0027
    }
  },
  "calibration": {
    "details": {},
    "T_world_opencv_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
  }
}
```

## 6. 本地必须做的结果校验

在生成任何机械臂动作前，至少检查：

```python
assert result["schema_version"] == "stirring-geometry-v1"
assert result["frame"] == "world"
assert result["units"] == "meters"
assert result["glass_rod"]["quality"]["valid"] is True
assert len(result["glass_rod"]["grasp_candidates"]) == 2
```

本地配置建议增加安全阈值：

```yaml
stirring_perception:
  min_segmentation_confidence: 0.50
  min_mask_elongation: 6.0
  min_table_plane_inlier_fraction: 0.75
  max_table_plane_rmse_m: 0.004
  max_endpoint_uncertainty_m: 0.010
  expected_rod_length_m: 0.196
  rod_length_tolerance_m: 0.020
```

任何一项失败都应停止，不允许使用旧结果继续执行。

## 7. 本地状态机需要增加的阶段

在原流程：

```text
开塞 -> 倾倒 -> 滴定
```

后增加：

```text
放回滴管
  -> 移动到最终观察位
  -> 采集并提交 /v1/stirring/jobs
  -> 人工确认 geometry_overlay.png（首轮实验必须）
  -> 从两个抓取候选中做 IK/碰撞筛选
  -> 抓取玻璃棒
  -> 提升到安全高度
  -> 将“抓取点到自由端”的方向对齐 world_down
  -> 自由端移动到烧杯口上方
  -> 竖直插入
  -> 水平圆周搅拌
  -> 竖直退出
  -> 放回玻璃棒或进入安全结束位
```

## 8. 抓取候选选择

服务器不选择端点。本地按以下顺序选择：

1. 为两个 `grasp_point_world` 分别生成上方预抓取姿态；
2. 夹爪从桌面法向上方接近；
3. 夹爪闭合轴与玻璃棒轴线垂直；
4. 分别进行 IK、关节限位、桌面和周边物体碰撞检查；
5. 选择安全裕量更大的候选；
6. 若两个候选都不可达，停止并重新摆放，不得使用端点中心硬凑。

选中候选后：

```python
p_grasp = candidate["grasp_point_world"]
p_tip = candidate["opposite_tip_world"]
u_grasp_to_tip = candidate["rod_direction_grasp_to_tip_world"]
```

玻璃棒抓取后，本地必须根据自己的 TCP/夹爪坐标约定保存 `T_tool_rod`。服务器不知道夹爪手指方向，因此不会返回六自由度抓取姿态。

## 9. 竖直向下调整

目标是让：

```text
R_world_rod * u_grasp_to_tip = world_down = [0, 0, -1]
```

本地计算把 `u_grasp_to_tip` 对齐到 `world_down` 的旋转，并结合实际 `T_tool_rod` 得到 TCP 姿态。

姿态调整必须在安全提升高度完成，不要贴近桌面或烧杯旋转长玻璃棒。还要检查旋转扫掠体是否碰到广口瓶、滴瓶、相机或机器人自身。

## 10. 烧杯口和搅拌轨迹

本接口不重新解算烧杯。复用前序 complex-demo 结果：

```python
mouth_points = previous_result["objects"]["beaker"]["affordance_regions"]["mouth"]["points_world"]
beaker_mouth_center = median(mouth_points, axis=0)
```

本地配置必须实测：

```yaml
stirring_motion:
  pregrasp_height_m: ...
  lift_height_m: ...
  preinsert_clearance_m: ...
  target_depth_below_mouth_m: ...
  stir_radius_m: ...
  stir_cycles: ...
  stir_speed_mps: ...
  beaker_safe_inner_radius_m: ...
  bottom_clearance_m: ...
  rod_grasp_force: ...
```

禁止直接复制示例数字。插入深度和安全内半径必须根据真实烧杯尺寸、液面和玻璃棒长度测量。

自由端预插入点：

```text
p_preinsert_tip = beaker_mouth_center + world_up * preinsert_clearance_m
```

自由端搅拌平面中心：

```text
p_stir_center = beaker_mouth_center - world_up * target_depth_below_mouth_m
```

圆周轨迹：

```text
p_tip(theta) = p_stir_center
             + world_x * stir_radius_m * cos(theta)
             + world_y * stir_radius_m * sin(theta)
```

执行中保持玻璃棒方向为 `world_down`。

必须满足：

```text
stir_radius_m + endpoint_uncertainty_m + rod_radius_m
    < beaker_safe_inner_radius_m
```

## 11. 运动安全要求

首轮只允许分阶段确认：

1. 只观察并下载结果；
2. 人工核对分割和端点叠加图；
3. 只运动到预抓取点，不闭合夹爪；
4. 低速抓取并仅提升少量距离；
5. 在远离器皿的安全高度验证竖直调整；
6. 只运动到杯口上方，不插入；
7. 使用空烧杯验证浅插入；
8. 使用最小圆周半径低速运行；
9. 最后才接入完整长程实验。

任何服务器任务失败、质量指标越界、IK失败、夹爪宽度异常或人工确认未通过，都必须阻止动作。

## 12. 服务器不会返回的内容

本地不得假设服务器提供以下信息：

- 可直接执行的六自由度抓取姿态；
- 哪个端点必须抓取；
- 烧杯液面；
- 烧杯底部位置；
- 安全插入深度；
- 搅拌半径、速度和圈数；
- 机器人IK、碰撞检查或UR路点；
- 抓取是否成功。

这些必须由本地配置、机器人模型、夹爪反馈和安全状态机负责。

## 13. 调试结果包

下载：

```text
GET /v1/stirring/jobs/{job_id}/bundle
```

ZIP包含：

```text
result.json
job.json
pipeline.log
00_segmentation/glass_rod/detections.json
00_segmentation/glass_rod/mask.png
00_segmentation/glass_rod/mask_prob.png
00_segmentation/glass_rod/overlay.png
glass_rod/geometry_overlay.png
```

本地失败日志应同时记录 `capture_id`、`job_id`、所选候选、所有运动参数和服务器完整结果。

## 14. 已完成的服务器验证

服务器使用真实采集 `20261012_202830_286562` 完成了CPU离线和ASGI接口测试：

```text
玻璃棒检测置信度：0.923
估计长度：0.1959 m
掩码细长度：18.34
桌面平面内点率：0.9689
桌面平面RMSE：0.00118 m
端点估计不确定度：0.00268 m
POST状态：202 queued
最终状态：completed
result状态：200, schema stirring-geometry-v1
结果ZIP：已生成并验证可解压
```

这些数值只是该采集的测试结果，不能作为所有实验的固定值。
