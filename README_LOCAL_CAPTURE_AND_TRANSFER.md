# 双物体滴定实验：本地采集、上传、服务器处理与结果下载

这份文档用于本地采集端与服务器端的交接。实验场景固定为：

- 一个带红色胶头的滴瓶；
- 一个烧杯；
- 眼在手上 RGB-D 相机；
- 目标是定位红色胶头中心和烧杯口中心，输出机器人世界/基坐标系下的三维坐标。

服务器端不运行 GraspNet。处理链路为：

```text
RGB-D + 内参 + 手眼标定 + 拍照时末端位姿
  -> YOLO 分割滴瓶和烧杯
  -> mask + depth 反投影观测点云
  -> SAM3D 分别重建两个物体
  -> 桌面/深度/轮廓约束恢复尺度和位姿
  -> dispense affordance 定位红色胶头
  -> pour affordance 定位烧杯口
  -> affordance 投影到三维 mesh
  -> 转换到机器人世界/基坐标系
  -> TXT / JSON / PLY / PNG
```

## 1. 本地每次需要采集什么

推荐每次采集使用独立目录：

```text
titration_001/
├── rgb.png
├── depth.png
├── camera.json
├── T_world_ee.json
└── capture_info.json
```

静态手眼标定 `T_ee_cam.json` 可以单独保存，标定不变时不需要每次采集。

### 1.1 `rgb.png`

要求：

- PNG 或 JPG；
- 推荐保留相机原始分辨率，当前已验证样例为 `640 x 480`；
- 滴瓶、红色胶头和烧杯口不应被机械臂或夹爪遮挡；
- RGB 必须与 depth 像素对齐。

### 1.2 `depth.png`

要求：

- 推荐使用无损 `uint16 PNG`；
- 不要存成 8-bit 可视化图；
- 必须是已对齐到 RGB/彩色相机的深度；
- RGB、depth 和机器人位姿必须来自同一采集时刻。

如果深度原始值的单位是毫米，服务器运行时使用：

```bash
--depth-scale 0.001
```

如果深度数值已经是米，使用：

```bash
--depth-scale 1.0
```

### 1.3 `camera.json`

支持 RealSense 嵌套格式：

```json
{
  "depth_scale_m_per_unit": 0.001,
  "aligned_depth_intrinsics": {
    "width": 640,
    "height": 480,
    "fx": 607.92822265625,
    "fy": 607.904541015625,
    "ppx": 326.13800048828125,
    "ppy": 250.4611053466797
  }
}
```

也支持简单格式：

```json
{
  "fx": 607.92822265625,
  "fy": 607.904541015625,
  "cx": 326.13800048828125,
  "cy": 250.4611053466797
}
```

脚本会自动生成：

```text
run_inputs/intrinsics_canonical.json
```

### 1.4 `T_ee_cam.json`：静态手眼标定

服务器代码期望：

```text
p_ee = T_ee_cam * p_opencv_camera
```

即 `T_ee_cam` 将 OpenCV 相机坐标系中的点转换到机器人末端坐标系。

当前手眼标定候选值为：

```text
translation (m):
  x = -0.004562593505863851
  y = -0.11592850154128732
  z =  0.003180012183660791

quaternion (x, y, z, w):
  x = -0.1483498681916791
  y =  0.011916681754711056
  z =  0.010742907230660681
  w =  0.9888047831840728
```

换算的齐次矩阵为：

```text
T_ee_cam =
[[ 0.999485165  -0.024780952   0.020379126  -0.004562594]
 [ 0.017709600   0.955753813   0.293634158  -0.115928502]
 [-0.026753962  -0.293122079   0.955700619   0.003180012]
 [ 0             0             0             1          ]]
```

重要：当前仍需要根据标定工具的 `frame_id` 和 `child_frame_id` 确认输出方向。如果工具输出的是 `T_cam_ee`，必须求逆后再作为 `T_ee_cam`。

### 1.5 `T_world_ee.json`：拍照瞬间末端位姿

这个矩阵必须在拍摄 RGB-D 的同一时刻从机器人获取：

```text
p_world = T_world_ee * p_ee
```

支持 JSON：

```json
{
  "T_world_ee": [
    [1.0, 0.0, 0.0, 0.42],
    [0.0, 1.0, 0.0, -0.13],
    [0.0, 0.0, 1.0, 0.58],
    [0.0, 0.0, 0.0, 1.0]
  ]
}
```

也支持 `.npy` 或纯文本 4 x 4 矩阵。

如果机器人接口返回：

```text
[x, y, z, rx, ry, rz]
```

后三项通常是旋转向量，本地采集代码应先将它转换为 3 x 3 旋转矩阵，再保存为 4 x 4 齐次矩阵。不能把 6D pose 数组直接当作脚本输入。

### 1.6 `capture_info.json`：推荐但非强制

```json
{
  "capture_id": "titration_001",
  "rgb_timestamp": 1234567890.123,
  "depth_timestamp": 1234567890.123,
  "robot_pose_timestamp": 1234567890.120,
  "depth_scale_m_per_unit": 0.001,
  "rgb_frame": "camera_color_optical_frame",
  "world_frame": "base",
  "ee_frame": "tool0"
}
```

建议检查 RGB、depth 和机器人 pose 时间差，机械臂运动期间不要把不同时刻的数据组合起来。

## 2. 世界坐标计算链

SAM3D 使用的相机坐标约定与 OpenCV optical frame 的 x/y 方向相反：

```text
SAM3D:
  x = -(u-cx)/fx*z
  y = -(v-cy)/fy*z
  z = depth

OpenCV:
  x = +(u-cx)/fx*z
  y = +(v-cy)/fy*z
  z = depth
```

因此：

```text
T_opencv_from_sam3d = diag(-1, -1, 1, 1)
```

最终计算为：

```text
p_world =
    T_world_ee(capture)
  * T_ee_cam
  * T_opencv_from_sam3d
  * p_sam3d_camera
```

缺少 `T_world_ee` 时只能计算相机坐标或末端坐标，无法计算真实世界坐标。

## 3. 本地上传到服务器

在本地机器执行：

```bash
rsync -avP ./titration_001/ \
  USER@SERVER:/home/dsj/code/language_affordance_grasp/inputs/titration_001/
```

静态手眼标定只需上传一次：

```bash
rsync -avP ./T_ee_cam.json \
  USER@SERVER:/home/dsj/code/language_affordance_grasp/configs/T_ee_cam.json
```

将 `USER` 和 `SERVER` 替换为服务器的用户名和 IP/域名。

## 4. 服务器端运行

服务器入口脚本：

```text
/home/dsj/code/language_affordance_grasp/scripts/run_titration_two_object.sh
```

运行：

```bash
cd /home/dsj/code/language_affordance_grasp

./scripts/run_titration_two_object.sh \
  --rgb inputs/titration_001/rgb.png \
  --depth inputs/titration_001/depth.png \
  --intrinsics inputs/titration_001/camera.json \
  --calibration configs/T_ee_cam.json \
  --world-ee inputs/titration_001/T_world_ee.json \
  --depth-scale 0.001 \
  --output-dir runs/titration_001
```

不指定 `--gpu` 时，脚本会选择空闲显存最多的 GPU，并要求至少空闲 22 GB。

手动指定物理 GPU：

```bash
--gpu 1
```

如果把 `T_ee_cam` 和当次 `T_world_ee` 合并在一个 JSON 里，可以只传：

```bash
--calibration calibration_and_capture_pose.json
```

该 JSON 应为：

```json
{
  "T_ee_cam": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
  "T_world_ee": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
}
```

脚本在启动 YOLO/SAM3D 前会检查标定链是否完整。如果 JSON 只包含 `T_ee_cam` 但没有 `T_world_ee`，脚本会立即拒绝，避免误将手眼矩阵当作世界外参。

## 5. 服务器输出目录

```text
runs/titration_001/
├── pipeline.log
├── run_inputs/
│   └── intrinsics_canonical.json
│
├── 00_segmentation/
│   ├── dropper/
│   │   ├── mask.png
│   │   ├── mask_prob.png
│   │   ├── overlay.png
│   │   └── detections.json
│   ├── beaker/
│   │   └── ...
│   └── depth_clouds/
│       ├── dropper_depth_camera.ply
│       ├── beaker_depth_camera.ply
│       └── depth_clouds.json
│
├── 01_affordance/
│   ├── dropper/
│   │   ├── affordance_heatmap.npy
│   │   ├── affordance_heatmap.png
│   │   └── affordance_overlay.png
│   └── beaker/
│       └── ...
│
├── 02_sam3d/
│   ├── dropper/
│   │   ├── optimized_mesh_camera.ply
│   │   ├── pose_result.json
│   │   ├── overlay.png
│   │   └── optimization_trace.jsonl
│   └── beaker/
│       └── ...
│
├── 03_projection/
│   ├── dropper/
│   │   ├── affordance_region_camera.ply
│   │   └── affordance_region_camera.npz
│   └── beaker/
│       └── ...
│
└── 04_targets/
    ├── target_centers_world.txt
    ├── titration_targets.json
    ├── target_centers_overlay.png
    ├── titration_scene_camera.ply
    ├── titration_scene_world.ply
    ├── targets_only_camera.ply
    ├── targets_only_world.ply
    ├── dropper_model_surface_camera.ply
    ├── dropper_model_surface_world.ply
    ├── beaker_model_surface_camera.ply
    ├── beaker_model_surface_world.ply
    ├── dropper_depth_world.ply
    └── beaker_depth_world.ply
```

### 最重要的结果

`04_targets/target_centers_world.txt`：

```text
Coordinates are in meters.
dropper_bulb_center_world: X Y Z
beaker_mouth_center_world: X Y Z
dropper_bulb_marker_rgba: 255 0 255 255
beaker_mouth_marker_rgba: 255 215 0 255
```

`04_targets/titration_targets.json` 包含：

- SAM3D 相机坐标；
- OpenCV 相机坐标；
- 世界/机器人基坐标；
- `T_world_cam`；
- `T_world_ee * T_ee_cam` 组合信息；
- affordance 阈值、支持点数和区域尺寸；
- 所有 PLY/PNG/TXT 的路径。

`04_targets/target_centers_overlay.png` 把最终三维中心重新投影到 RGB，用于检查是否定位在红色胶头和烧杯口。

### PLY 颜色

```text
滴瓶模型：   青色     RGBA = 50, 180, 255, 255
烧杯模型：   绿色     RGBA = 70, 220, 120, 255
胶头中心球： 洋红色   RGBA = 255, 0, 255, 255
烧杯口中心球：黄色   RGBA = 255, 215, 0, 255
桌面：       灰色     RGBA = 150, 150, 150, 180
```

## 6. 从服务器下载到本地

下载整个运行目录：

```bash
rsync -avP \
  USER@SERVER:/home/dsj/code/language_affordance_grasp/runs/titration_001/ \
  ./titration_001_result/
```

只下载最终结果：

```bash
rsync -avP \
  USER@SERVER:/home/dsj/code/language_affordance_grasp/runs/titration_001/04_targets/ \
  ./titration_001_targets/
```

建议额外下载日志、YOLO/affordance 检查图和 SAM3D 对齐图：

```bash
mkdir -p ./titration_001_debug

rsync -avP \
  USER@SERVER:/home/dsj/code/language_affordance_grasp/runs/titration_001/pipeline.log \
  USER@SERVER:/home/dsj/code/language_affordance_grasp/runs/titration_001/00_segmentation/ \
  USER@SERVER:/home/dsj/code/language_affordance_grasp/runs/titration_001/01_affordance/ \
  USER@SERVER:/home/dsj/code/language_affordance_grasp/runs/titration_001/02_sam3d/ \
  ./titration_001_debug/
```

## 7. 本地 Codex 需要完成的采集任务

本地采集代码应保证：

1. 获取 RGB 帧。
2. 获取对齐到 RGB 的原始深度帧。
3. 保存彩色/对齐深度内参和 depth scale。
4. 在拍照的同一时刻获取机器人 `T_world_ee`。
5. 把机器人 6D pose 转换为合法的 4 x 4 齐次矩阵。
6. 记录 RGB、depth 和 robot pose 的时间戳与 frame name。
7. 检查 RGB 和 depth 尺寸一致。
8. 确保 depth 是 `uint16` 或其他无损原始精度，不要转为 8-bit。
9. 确保拍照时滴瓶胶头和烧杯口未被夹爪遮挡。
10. 把一次采集的所有文件放在同一目录后再上传。

## 8. 开始机械臂操作前的检查

不要仅依赖 `target_centers_world.txt` 就直接执行机械臂运动。至少检查：

1. `00_segmentation/*/overlay.png` 的两个 mask 是否正确。
2. `01_affordance/dropper/affordance_overlay.png` 是否指向红色胶头。
3. `01_affordance/beaker/affordance_overlay.png` 是否指向烧杯口。
4. `02_sam3d/*/overlay.png` 的三维模型轮廓是否合理。
5. `04_targets/target_centers_overlay.png` 的两个中心是否落在正确区域。
6. `04_targets/titration_scene_world.ply` 中两个物体和彩色中心球是否在正确位置。
7. 用一个世界坐标已知的标志点验证完整的 `T_world_ee * T_ee_cam` 变换链。
8. 机械臂首次实验使用安全高度、低速度和 dry-run。

当前服务器端已完成 RGB-D 样例 smoke test，但真机前仍必须确认手眼标定矩阵方向，并上传拍照瞬间的 `T_world_ee`。
