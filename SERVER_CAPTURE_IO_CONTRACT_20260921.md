# 服务器输入输出契约：采集 `20260921_171530_302852`

> 本文件用于服务器端实现和联调。本次只上传已经采集好的数据、等待服务器解算并把结果下载回本地；不控制机械臂，不重新拍照。

## 1. 本次任务范围

采集目录：

```text
/home/dsj/vla/affordance/captures/20260921_171530_302852
```

RGB 中包含：

- 一个带白色瓶塞的广口瓶；
- 一个胶头滴瓶；
- 一个烧杯。

本次画面中没有玻璃棒，因此服务器只处理：

```text
wide_mouth_bottle
beaker
dropper
```

不要要求 `glass_rod`，也不要在结果中返回伪造的玻璃棒数据。

## 2. 服务器能力边界

本次只要求服务器输出它已有能力可以得到的信息：

1. 每个目标的三维点云/模型；
2. 每个模型的尺度；
3. 每个模型的位姿；
4. 已支持的 affordance 区域三维点集：
   - 广口瓶瓶塞区域；
   - 广口瓶瓶身区域；
   - 胶头滴瓶的胶头区域；
   - 复用旧滴定 pipeline 的烧杯口（pour）区域。

本次不要求服务器输出：

- 瓶口中心；
- 瓶身轴或拔塞轴；
- 烧杯口法向或半径；
- 液面；
- 滴管尖端；
- 抓取姿态；
- 倾倒量；
- 真实滴数；
- 滴定终点；
- 机械臂路径。

服务器不得使用零点、包围盒中心或未经算法支持的猜测补齐这些信息。

## 3. HTTP 接口

建议使用独立端点，不修改旧 `/v1/titration/jobs`：

```text
POST /v1/complex-demo/jobs
GET  /v1/complex-demo/jobs/{job_id}
GET  /v1/complex-demo/jobs/{job_id}/result
GET  /v1/complex-demo/jobs/{job_id}/bundle?scope=results
GET  /v1/complex-demo/jobs/{job_id}/bundle?scope=all
```

认证方式：

```http
Authorization: Bearer <LAG_API_TOKEN>
```

新旧任务应使用服务器现有的同一个串行 GPU 队列，避免多个三维重建任务抢显存。

## 4. POST 输入

请求类型：

```http
Content-Type: multipart/form-data
```

### 4.1 文件字段

| multipart 字段 | 本地文件 | 类型 | 必需 |
|---|---|---|---:|
| `rgb` | `rgb.png` | 640×480 彩色 PNG | 是 |
| `depth` | `depth.png` | 640×480 `uint16` PNG | 是 |
| `camera` | `camera.json` | JSON | 是 |
| `world_ee` | `T_world_ee.json` | JSON | 是 |

完整路径：

```text
/home/dsj/vla/affordance/captures/20260921_171530_302852/rgb.png
/home/dsj/vla/affordance/captures/20260921_171530_302852/depth.png
/home/dsj/vla/affordance/captures/20260921_171530_302852/camera.json
/home/dsj/vla/affordance/captures/20260921_171530_302852/T_world_ee.json
```

### 4.2 普通表单字段

| 字段 | 值 | 编码 |
|---|---|---|
| `depth_scale` | `0.0010000000474974513` | 浮点数字符串 |
| `capture_id` | `20260921_171530_302852` | 字符串 |
| `requested_objects` | `['wide_mouth_bottle','beaker','dropper']` | JSON 字符串，实际使用双引号 |
| `requested_affordance_regions` | 见下方 | JSON 字符串 |
| `gpu` | 可省略或物理 GPU 编号 | 整数 |

`requested_objects` 实际值：

```json
["wide_mouth_bottle", "beaker", "dropper"]
```

`requested_affordance_regions` 实际值：

```json
{
  "wide_mouth_bottle": ["stopper", "body_grasp"],
  "beaker": ["mouth"],
  "dropper": ["bulb"]
}
```

烧杯要求完整三维模型、尺度、位姿，以及旧 pipeline 已支持的烧杯口 affordance 区域。服务器只导出该区域点集，不需要额外预测法向、半径或液面。

## 5. 输入文件内容

### 5.1 `rgb.png`

```text
shape: 480 × 640 × 3
dtype: uint8
```

服务器按现有图像加载方式读取即可。如果使用 OpenCV，读取结果为 BGR；不要再额外交换通道，除非模型明确要求 RGB。

### 5.2 `depth.png`

```text
shape: 480 × 640
dtype: uint16
raw range in this capture: 0..758
```

深度已经对齐到彩色图。米制深度：

```text
depth_m = depth_uint16 * 0.0010000000474974513
```

`0` 表示无效深度，不能作为真实三维点。

### 5.3 `camera.json`

本次实际内容：

```json
{
  "width": 640,
  "height": 480,
  "fx": 607.2290649414062,
  "fy": 605.6921997070312,
  "cx": 318.6340026855469,
  "cy": 246.89401245117188,
  "distortion_model": "inverse_brown_conrady",
  "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
  "depth_scale_m_per_unit": 0.0010000000474974513
}
```

服务器需要检查内参尺寸与 RGB/depth 一致。

### 5.4 `T_world_ee.json`

本次实际内容：

```json
{
  "T_world_ee": [
    [-0.44469121439265535, -0.5472008750806913, 0.7090986716620693, 0.281365777745128],
    [-0.8912511292809765, 0.19166425594153247, -0.4110185367471434, 0.007560793029075481],
    [0.08900083368918255, -0.814761324134439, -0.5729248085895093, 0.3222485944574013],
    [0.0, 0.0, 0.0, 1.0]
  ]
}
```

坐标含义：

```text
world = UR base
ee = UR 控制器全零 TCP 的 tool0
T_world_ee = T_base_tool0
```

本地采集时实际 TCP offset 为：

```text
[0, 0, 0.14, 0, 0, 0]
```

本地在保存 `T_world_ee` 前已经去除了这个 140 mm TCP offset。服务器必须直接使用 `T_world_ee`，不要再次去除 TCP offset，也不要根据 `capture_info.actual_tcp_pose` 重建它。

## 6. 坐标变换

眼在手上标定定义为：

```text
p_tool0 = T_tool0_camera * p_opencv_camera
```

最终 world/base 坐标：

```text
p_base = T_base_tool0 * T_tool0_camera * p_opencv_camera
```

即：

```text
p_world = T_world_ee * T_ee_cam * p_opencv_camera
```

如果服务器内部 SAM3D 坐标与 OpenCV optical frame 存在：

```text
T_opencv_from_sam3d = diag(-1, -1, 1, 1)
```

只应用一次，不能在输出 world 点云后再次翻转。

模型点云与 affordance 区域点必须使用相同的 world/base 米制坐标定义。

## 7. POST 返回

成功接收并排队：

```json
{
  "job_id": "安全的字母数字下划线或连字符",
  "state": "queued"
}
```

状态流转：

```text
uploading -> queued -> running -> completed
                         |-> failed
```

查询结果中应保留：

```json
{
  "metadata": {
    "capture_id": "20260921_171530_302852",
    "depth_scale": 0.0010000000474974513,
    "requested_objects": ["wide_mouth_bottle", "beaker", "dropper"]
  }
}
```

## 8. 必须返回的 result JSON

`GET /v1/complex-demo/jobs/{job_id}/result` 直接返回以下结构，不要再套一层 `result`：

```json
{
  "job_id": "abc123",
  "capture_id": "20260921_171530_302852",
  "frame": "world",
  "units": "meters",
  "model_transform_convention": "p_world = T_world_object @ [scale * p_object, 1]",
  "requested_objects": [
    "wide_mouth_bottle",
    "beaker",
    "dropper"
  ],
  "objects": {
    "wide_mouth_bottle": {
      "model_id": "服务器实际稳定模型ID",
      "T_world_object": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      "scale": [1.0, 1.0, 1.0],
      "model_points_world_artifact": "models/wide_mouth_bottle_world.ply",
      "affordance_regions": {
        "stopper": {
          "points_world": [[0.0,0.0,0.0], [0.0,0.0,0.0]]
        },
        "body_grasp": {
          "points_world": [[0.0,0.0,0.0], [0.0,0.0,0.0]]
        }
      }
    },
    "beaker": {
      "model_id": "服务器实际稳定模型ID",
      "T_world_object": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      "scale": [1.0, 1.0, 1.0],
      "model_points_world_artifact": "models/beaker_world.ply",
      "affordance_regions": {
        "mouth": {
          "points_world": [[0.0,0.0,0.0], [0.0,0.0,0.0]]
        }
      }
    },
    "dropper": {
      "model_id": "服务器实际稳定模型ID",
      "T_world_object": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      "scale": [1.0, 1.0, 1.0],
      "model_points_world_artifact": "models/dropper_world.ply",
      "affordance_regions": {
        "bulb": {
          "points_world": [[0.0,0.0,0.0], [0.0,0.0,0.0]]
        }
      }
    }
  },
  "calibration": {
    "id": "标定文件SHA256或版本ID",
    "ee_frame": "tool0",
    "world_frame": "base"
  }
}
```

示例中的单位矩阵和零点仅表示字段格式，真实结果绝不能返回占位值。

## 9. 输出字段解释

### 9.1 `model_id`

标识服务器使用的 canonical 模型定义。同一 ID 的模型原点、轴、尺度定义必须跨运行稳定。如果 canonical 坐标每次随机变化，服务器必须明确说明，不能假装稳定。

### 9.2 `T_world_object` 和 `scale`

服务器必须使用并验证：

```text
p_scaled_object = scale elementwise-multiply p_object
p_world = R_world_object * p_scaled_object + t_world_object
```

如果服务器内部约定不同，应在导出层转换到这个统一表示。

### 9.3 `model_points_world_artifact`

必须是下载 ZIP 内的安全相对路径，例如：

```text
models/beaker_world.ply
```

不要返回服务器绝对路径。

### 9.4 `points_world`

这是服务器已有 affordance 区域的三维点样本，不是服务器推导的中心。

要求：

- shape 为 `[N,3]`；
- 米制 world/base 坐标；
- 全部有限；
- 不包含无效深度形成的零点；
- 建议均匀下采样 256～4096 点；
- 不要只返回单一中心；
- 不要伪造 confidence。

本地之后会用区域点逐轴中位数计算代表点。

## 10. 结果 ZIP

`scope=results` 至少包含：

```text
result.json
pipeline.log
models/wide_mouth_bottle_world.ply
models/beaker_world.ply
models/dropper_world.ply
affordance_regions/wide_mouth_bottle/stopper_world.ply
affordance_regions/wide_mouth_bottle/body_grasp_world.ply
affordance_regions/beaker/mouth_world.ply
affordance_regions/dropper/bulb_world.ply
overlays/segmentation_overlay.png
overlays/affordance_overlay.png
```

JSON 中声明的每个 artifact 必须真实存在于 ZIP。打包失败时任务不能标记 completed。

## 11. 失败返回

以下情况任务进入 failed：

- 任一请求对象重建失败；
- 模型尺度或位姿无效；
- model ID/canonical 定义不明确；
- 瓶塞、瓶身、烧杯口或胶头 affordance 区域为空；
- affordance 无法投影出有效 world 点；
- 输出出现 NaN/Inf；
- 标定链不完整；
- artifact 写入或打包失败。

错误至少说明：

```json
{
  "stage": "affordance_region_export",
  "object": "wide_mouth_bottle",
  "region": "stopper",
  "code": "NO_VALID_WORLD_POINTS",
  "message": "..."
}
```

不要返回部分伪造结果后标记 completed。

## 12. 本次服务器不做的事情

服务器端不要新增或声称存在：

```text
liquid_surface_detector
dropper_tip_detector
object_axis_estimator
grasp_pose_planner
pour_amount_estimator
drop_counter
titration_endpoint_detector
```

烧杯口不是新增算法：必须复用旧实验中已有的 `Pour affordance for beaker mouth`、三维 mesh 投影和 world 坐标导出逻辑。

本次任务完成标准只是：可靠返回三个请求对象的模型、尺度/位姿和四个已有 affordance 区域（瓶塞、瓶身、烧杯口、胶头）。

## 13. 联调交付物

服务器端完成后，请向本地提供：

1. 本次任务 `job_id`；
2. `job.json`；
3. `result.json`；
4. `scope=results` ZIP；
5. pipeline log；
6. `model_id`、canonical 坐标、scale、pose 的准确定义；
7. canonical 原点与轴是否跨运行稳定的明确结论；
8. 新旧 API 测试结果。

本次联调到“结果文件成功下载并可人工检查”为止，不执行任何机械臂动作。
