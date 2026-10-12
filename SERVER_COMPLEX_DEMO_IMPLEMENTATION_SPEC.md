# 服务器 Codex 实施任务：按现有真实能力新增复杂实验数据接口

> 请把本文件完整交给服务器端 Codex。本规格严格限制服务器职责，不要求服务器输出当前算法不能得到的信息。

## 0. 已确认的能力边界

服务器当前只能提供：

1. 目标物体的三维点云/表面模型；
2. 重建模型的尺度和位姿；
3. 已支持的部分 affordance 区域：广口瓶瓶盖/橡胶塞、广口瓶瓶身、旧实验烧杯口（pour）、胶头滴瓶胶头、玻璃棒两个端部。

服务器当前不能直接提供：

- 瓶口中心、拔塞轴或瓶身轴；
- 烧杯口法向、半径或液面；
- 滴管尖端；
- 玻璃棒哪端是上端；
- 六自由度抓取姿态；
- 倾倒量、真实滴数或滴定终点。

不得把本地几何推导结果描述成服务器预测，也不得为了满足接口而返回虚构的中心、轴、液面或置信度。

## 1. 新接口的唯一职责

新接口只把服务器已经产生的数据交给本地：

```text
RGB-D + 内参 + T_world_ee + 手眼标定
  -> 现有目标分割/三维重建
  -> 模型尺度和位姿
  -> 现有 affordance 区域投影
  -> world 模型点云和 affordance 区域 world 点集
  -> HTTP result + ZIP
```

本地负责从区域点集求鲁棒中心，并结合模型位姿、尺度和人工标定的 canonical 几何先验生成机械臂动作。服务器不生成 UR 路点、不控制机器人。

## 2. 旧接口必须保持不变

以下旧 API 继续原样工作：

```text
GET  /health
POST /v1/titration/jobs
GET  /v1/titration/jobs/{job_id}
GET  /v1/titration/jobs/{job_id}/result
GET  /v1/titration/jobs/{job_id}/bundle
```

要求：

1. 不改变旧结果字段、任务目录或 ZIP；
2. 新旧任务使用同一个串行 GPU 队列；
3. 不启动第二套并发 SAM3D worker；
4. 复用现有 Bearer Token、状态、重启恢复、标定和打包逻辑；
5. 可在内部增加 `job_kind=titration|complex_demo`，但不能要求旧客户端传它。

## 3. 新 HTTP 路由

新增：

| Method | Path |
|---|---|
| `POST` | `/v1/complex-demo/jobs` |
| `GET` | `/v1/complex-demo/jobs/{job_id}` |
| `GET` | `/v1/complex-demo/jobs/{job_id}/result` |
| `GET` | `/v1/complex-demo/jobs/{job_id}/bundle?scope=results` |
| `GET` | `/v1/complex-demo/jobs/{job_id}/bundle?scope=all` |

POST multipart 与旧接口一致：`rgb`、`depth`、`camera`、`world_ee`，表单字段为 `depth_scale`、`capture_id`、可选 `gpu`。继续校验图像尺寸、uint16 depth、内参、正深度尺度、刚体矩阵、文件大小和安全 capture ID。

状态保持：

```text
uploading -> queued -> running -> completed
                         |-> failed
queued/running + API restart -> interrupted
```

只有模型、尺度/位姿、必需 affordance 区域和 ZIP 全部导出后才可标记 completed。

## 4. 精确结果 JSON

结果只描述重建对象和 affordance 区域，不包含服务器无法提供的高级语义：

```json
{
  "job_id": "abc123",
  "frame": "world",
  "units": "meters",
  "model_transform_convention": "p_world = T_world_object @ [scale * p_object, 1]",
  "objects": {
    "wide_mouth_bottle": {
      "model_id": "wide_mouth_bottle_v1",
      "T_world_object": [[1,0,0,0.46],[0,1,0,0.18],[0,0,1,0.13],[0,0,0,1]],
      "scale": [1.0, 1.0, 1.0],
      "model_points_world_artifact": "models/wide_mouth_bottle_world.ply",
      "affordance_regions": {
        "stopper": {"points_world": [[0.46,0.18,0.23],[0.461,0.18,0.23]]},
        "body_grasp": {"points_world": [[0.46,0.18,0.13],[0.461,0.18,0.13]]}
      }
    },
    "beaker": {
      "model_id": "beaker_v1",
      "T_world_object": [[1,0,0,0.62],[0,1,0,-0.10],[0,0,1,0.12],[0,0,0,1]],
      "scale": [1.0, 1.0, 1.0],
      "model_points_world_artifact": "models/beaker_world.ply",
      "affordance_regions": {
        "mouth": {"points_world": [[0.62,-0.10,0.12],[0.621,-0.10,0.12]]}
      }
    },
    "dropper": {
      "model_id": "dropper_v1",
      "T_world_object": [[1,0,0,0.55],[0,1,0,0.20],[0,0,1,0.08],[0,0,0,1]],
      "scale": [1.0, 1.0, 1.0],
      "model_points_world_artifact": "models/dropper_world.ply",
      "affordance_regions": {
        "bulb": {"points_world": [[0.55,0.20,0.18],[0.551,0.20,0.18]]}
      }
    },
    "glass_rod": {
      "model_id": "glass_rod_v1",
      "T_world_object": [[1,0,0,0.34],[0,1,0,-0.20],[0,0,1,0.07],[0,0,0,1]],
      "scale": [1.0, 1.0, 1.0],
      "model_points_world_artifact": "models/glass_rod_world.ply",
      "affordance_regions": {
        "end_a": {"points_world": [[0.34,-0.20,0.20]]},
        "end_b": {"points_world": [[0.34,-0.20,0.07]]}
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

示例点数是简写，真实区域必须包含足够多的三维点。

### 4.1 必需对象与区域

```text
wide_mouth_bottle:
  模型 + pose + scale
  stopper 区域
  body_grasp 区域

beaker:
  模型 + pose + scale
  mouth 区域（复用旧实验的 pour/beaker-mouth pipeline）
  不要求法向、半径或液面

dropper:
  模型 + pose + scale
  bulb 区域
  不要求 tip

glass_rod:
  模型 + pose + scale
  end_a 区域
  end_b 区域
  不要求判断端部语义
```

某个必需模型或区域失败时任务进入 failed，不能用零点、包围盒中心或猜测补齐。

### 4.2 字段约束

- `frame=world`，`units=meters`；
- `T_world_object` 是有限的 4×4 刚体矩阵；
- `scale` 是正标量或三个正数，推荐统一为三轴尺度；
- `model_id` 必须稳定并对应本地 canonical 几何先验；
- artifact 是 ZIP 内安全相对路径，不能是 `/home/...` 绝对路径；
- `points_world` 是已有 affordance 区域三维点的直接导出，不是新预测；
- 区域建议均匀采样 256～4096 点，避免 JSON 过大；
- 不得伪造 confidence；已有原始分数可作为额外字段保留。

## 5. 尺度、位姿与 model ID

客户端使用唯一公式：

```text
p_scaled_object = scale elementwise-multiply p_object
p_world = R_world_object * p_scaled_object + t_world_object
```

如果当前服务器内部尺度顺序不同，应在导出层转换到该表示，不能仅修改字符串声明。

同一 `model_id` 的 canonical 原点、轴方向、单位和归一化必须跨运行稳定。更换模型或 canonical 定义必须更换 model ID。如果当前算法每次 canonical 朝向随机，本地锚点方案不能安全工作，服务器必须明确报告，不能假装稳定。

## 6. Affordance 区域导出

服务器直接导出已有投影区域：

```text
affordance_regions/<object>/<region>_world.npz
affordance_regions/<object>/<region>_world.ply
```

NPZ 推荐字段：

```text
points_world: float32 [N,3]
scores:       float32 [N]  # 只有算法原本有时才输出
```

result JSON 中放均匀下采样点。不要只返回服务端计算的单一中心；本地用逐轴中位数从实际区域计算中心。

可以保留投影点数、阈值、原始分数、mask 和 overlay，这些是已有结果；不要包装成不存在的高级语义。

## 7. 坐标链

继续使用当前眼在手上标定：

```text
p_tool0 = T_tool0_camera * p_opencv_camera
p_base  = T_base_tool0(capture) * p_tool0
```

本地 `server.ee_frame: flange` 实际表示 UR 全零 TCP `tool0`。本地已去掉 140 mm TCP offset，所以上传的是 `T_base_tool0`。

要求：

1. 继续从 `LAG_EYE_IN_HAND_CALIBRATION` 加载标定；
2. 使用对齐到 RGB 的 depth 和彩色相机内参；
3. SAM3D/OpenCV 的 `diag(-1,-1,1,1)` 只应用一次；
4. 模型 PLY 和区域点使用同一 world/base 变换；
5. 记录标定版本；
6. 不要求服务器估计世界竖直、桌面法向或对象语义轴。

## 8. 本地如何使用有限输出（不是服务器任务）

本地保存对象特定先验：模型 ID、canonical 瓶轴、瓶塞到瓶口偏移、烧杯 world 轴/实测安全内半径、滴管 canonical 尖端锚点等。

本地推导：

```text
stopper_grasp = median(stopper.points_world)
bottle_body_grasp = median(body_grasp.points_world)
bottle_axis = transform(server pose/scale, local canonical bottle axis)
bottle_mouth = stopper_grasp + bottle_axis * local calibrated signed offset

beaker_mouth = median(beaker mouth affordance points_world)
beaker_axis = locally configured world axis
beaker_radius = locally measured safe inner radius

dropper_bulb = median(bulb.points_world)
dropper_tip = transform(server pose/scale, local canonical tip anchor)

rod_end_a = median(end_a.points_world)
rod_end_b = median(end_b.points_world)
```

这些都是本地推导，不是服务器检测。

没有液面检测。搅拌目标是“服务器烧杯口 affordance 中心沿本地配置的 world 轴向下固定距离”，配置名为 `target_depth_below_mouth_m`。它是开环下探深度，必须人工保证不碰杯底。

## 9. 结果 ZIP

`scope=results` 至少包含：

```text
pipeline.log
result.json
models/wide_mouth_bottle_world.ply
models/beaker_world.ply
models/dropper_world.ply
models/glass_rod_world.ply
affordance_regions/wide_mouth_bottle/stopper_world.ply
affordance_regions/wide_mouth_bottle/body_grasp_world.ply
affordance_regions/beaker/mouth_world.ply
affordance_regions/dropper/bulb_world.ply
affordance_regions/glass_rod/end_a_world.ply
affordance_regions/glass_rod/end_b_world.ply
overlays/segmentation_overlay.png
overlays/affordance_overlay.png
```

`scope=all` 再包含已有流水线产生的 mask、heatmap、camera 点云、SAM3D 中间 mesh、对齐步骤和完整日志。

## 10. 推荐服务器改动

服务器 Codex 先阅读实际代码，再决定文件名。推荐：

1. 在现有 FastAPI/JobManager 增加 complex 路由和 job kind；
2. 新增复杂场景 runner，复用现有分割、重建、world 变换和打包；
3. 新增“模型/区域导出器”，不新增语义推理；
4. 导出 pose/scale/world PLY 和已有区域 world 点；
5. 新旧任务共用现有队列。

可能的组织：

```text
src/language_affordance_grasp/
  titration_api.py
  complex_demo_pipeline.py
  reconstruction_export.py
  affordance_region_export.py
  schemas/complex_demo.py
scripts/run_complex_demo.sh
```

不要新增或声称存在：

```text
liquid_surface_detector
dropper_tip_detector
grasp_pose_planner
pour_amount_estimator
drop_counter
```

烧杯口不得实现成新检测器；直接复用旧实验已有的 `Pour affordance for beaker mouth`、mesh 投影和 world 区域导出。

## 11. 失败策略

只对服务器真实负责的输出做失败判断：

- 必需对象重建失败；
- pose/scale 无效；
- model ID/canonical 定义不明确；
- 必需 affordance 区域为空或没有有效 world 点；
- world 点出现 NaN/Inf；
- artifact 未写入 ZIP；
- 标定链不完整。

错误应指出 stage/object/region，例如：

```json
{
  "stage": "affordance_region_export",
  "object": "glass_rod",
  "region": "end_b",
  "code": "NO_VALID_WORLD_POINTS"
}
```

本地缺少 canonical 锚点不是服务器失败；那属于本地配置。

## 12. 测试要求

### 回归

- 旧 titration API 测试全部通过；
- 旧 JSON/ZIP 不变；
- 新旧任务进入同一串行 GPU 队列。

### 新接口

覆盖：

1. 四个对象的 model ID、pose、scale、artifact；
2. scalar/三轴 scale；
3. 非刚体矩阵、负尺度、NaN/Inf 被拒绝；
4. affordance world 区域导出和下采样；
5. 空区域导致失败；
6. artifact 路径安全；
7. camera 到 world 变换只应用一次；
8. Token、failed、timeout、interrupted；
9. results/all ZIP 可打开。

必须有确定性数值测试：给定 canonical 点、scale 和 `T_world_object`，导出的 world 点应严格符合第 5 节公式。

### 真实数据验收

- 四个模型 PLY 能在同一 world 场景中对齐；
- 瓶塞、瓶身、胶头、玻璃棒两端区域位于正确物理区域；
- pose/scale 能复现 world 模型点；
- 多个观察位姿下，同一静态对象的 world 位置基本一致；
- 不评价服务器未提供的瓶口、液面、滴管尖端等语义精度。

## 13. 本地联调

```bash
ssh -N -L 8000:127.0.0.1:8000 USER@SERVER
export LAG_API_TOKEN='与服务器一致的值'

cd /home/dsj/vla/affordance
conda activate ur_data
python run_complex_demo.py
```

默认只采集、上传、下载和规划，不运动。也可离线检查：

```bash
python run_complex_demo.py --scene-json /path/to/result.json
```

服务器完成时应交付：

1. 修改文件列表；
2. 新旧 API 测试结果；
3. 一份真实或固定输入的 `result.json`；
4. 对应 results ZIP；
5. canonical、scale、pose 的精确定义；
6. model ID/canonical 朝向是否跨运行稳定的明确结论。

## 14. 完成定义

- 未声称服务器能检测瓶口、液面、尖端或语义轴；
- 只输出已有模型、尺度/位姿和已有 affordance 区域；
- result 符合第 4 节；
- 模型点和区域点均为 world/base 米制；
- 旧 API 无回归且共用队列；
- ZIP 包含所有引用 artifact；
- canonical/scale/pose 有测试和文档；
- 失败时不返回虚构几何量。
