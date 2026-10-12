# Language Affordance Grasp

玻璃棒最终搅拌阶段使用独立的 CPU 几何定位接口（不运行 SAM3D）。服务器/本地输入输出契约和本地机械臂配合要求见
[`STIRRING_LOCAL_INTEGRATION_PROMPT.md`](STIRRING_LOCAL_INTEGRATION_PROMPT.md)。

本地采集、数据上传、服务器运行和结果下载的完整交接说明见 `README_LOCAL_CAPTURE_AND_TRANSFER.md`。HTTP 任务接口和本地客户端说明见 `README_LOCAL_SERVER_API.md`。

把三个已有项目串成一条离线可回放链路：

```text
RGB-D + optional YOLOv5 mask + language instruction
  -> YOLOv5-seg object mask, if `--mask` is omitted
  -> FastSAM affordance heatmap
  -> SAM3D camera-frame object/table reconstruction
  -> 2D affordance projected to 3D mesh surface
  -> GraspNet scene grasp candidates
  -> affordance/workspace/task/safety filtering
  -> selected camera/base/TCP grasp JSON for dry-run or robot execution
```

## 依赖策略

本项目尽量不改动原仓库，使用文件中间产物和命令行适配器：

- FastSAM：默认用 `fastsam-annot` conda env，遵守 `/home/dsj/code/FastSAM/AGENTS.md`。
- YOLOv5：默认在 `/home/dsj/code/yolo_3D` 下导出二值实例 mask。
- SAM3D：默认在 `/home/dsj/code/sam-3d-objects` 下运行 `recover_glass_pose_notable_graspnet.py`。
- GraspNet：默认在 `/home/dsj/code/graspnet-baseline` 下运行 `demo_me.py`。

如三个环境不同，在 `configs/default.yaml` 里填：

```yaml
envs:
  yolo: yolo-env
  fastsam: fastsam-annot
  sam3d: sam3d-env
  graspnet: graspnet-env
```

空字符串表示直接用当前 Python 环境运行。

## 运行完整链路

`fit_case_122` 可以用一个脚本完成 YOLO、affordance、SAM3D、投影、GraspNet、过滤和导出。不传 GPU 索引时会自动选择空闲显存最多的卡：

```bash
./scripts/run_fit_case_122_full.sh
```

手动指定物理 GPU 和输出目录：

```bash
./scripts/run_fit_case_122_full.sh 1 /home/dsj/code/language_affordance_grasp/runs/fit_case_122_manual
```

如果需要生成机器人基坐标/TCP 位姿，设置标定文件：

```bash
LAG_T_BASE_CAM=/path/to/T_base_cam.npy ./scripts/run_fit_case_122_full.sh 1
```

```bash
cd /home/dsj/code/language_affordance_grasp
PYTHONPATH=src python -m language_affordance_grasp run \
  --rgb /path/to/rgb.png \
  --depth /path/to/depth.png \
  --mask /path/to/mask.png \
  --table-mask /path/to/table_mask.png \
  --intrinsics-json /path/to/intrinsics.json \
  --instruction "where should i interact to open/hold the jar?" \
  --t-base-cam /path/to/T_base_cam.npy \
  --output-dir /home/dsj/code/language_affordance_grasp/runs/jar_open_001
```

如果不传 `--mask`，会先用 YOLOv5 自动分割目标物体并生成 `00_segmentation/mask.png`：

```bash
PYTHONPATH=src python -m language_affordance_grasp run \
  --rgb /path/to/rgb.png \
  --depth /path/to/depth.png \
  --table-mask /path/to/table_mask.png \
  --intrinsics-json /path/to/intrinsics.json \
  --yolo-class-name "Glass dropper bottle" \
  --instruction "Where should I interact with the Glass dropper bottle to dispense it?" \
  --output-dir /tmp/lag_dropper_single_object
```

单独测试 YOLOv5 mask 导出：

```bash
PYTHONPATH=src python -m language_affordance_grasp segment \
  --rgb /path/to/rgb.png \
  --yolo-class-name "beaker" \
  --output-dir /tmp/lag_beaker_seg
```

`segment` 会输出 `mask.png`、`mask_prob.png`、`overlay.png` 和 `detections.json`。`--segmentation-selection best` 取最高置信度实例，`union` 合并同类所有实例。

先做命令检查，不跑重模型：

```bash
PYTHONPATH=src python -m language_affordance_grasp run \
  --rgb /home/dsj/code/sam-3d-objects/fit_case_122_zhui/rgb.png \
  --depth /home/dsj/code/sam-3d-objects/fit_case_122_zhui/depth.png \
  --mask /home/dsj/code/sam-3d-objects/fit_case_122_zhui/mask.png \
  --table-mask /home/dsj/code/sam-3d-objects/fit_case_122_zhui/table_mask.png \
  --intrinsics-json /home/dsj/code/sam-3d-objects/fit_case_122_zhui/intrinsics.json \
  --instruction "where should i interact to open the jar lid?" \
  --output-dir /tmp/lag_dry_run \
  --dry-run
```

## 双物体滴定实验（不使用 GraspNet）

`run_titration_two_object.sh` 针对“一个滴瓶 + 一个烧杯”的单次实验，依次执行：

```text
YOLO 两个 mask
  -> 保留当前目标并屏蔽其他 YOLO 物体
  -> 分别运行目标条件 affordance
  -> mask + depth 反投影两个观测点云
  -> SAM3D 分别重建和恢复尺度位姿
  -> dispense / pour 两张 affordance heatmap
  -> affordance 投影回各自三维 mesh
  -> 胶头中心和烧杯口中心转到世界坐标
  -> JSON/TXT 坐标和带彩色标记球的 PLY
```

Affordance 默认使用 `mask_others`：对每个目标保留整图视野，将其他 YOLO 物体的 mask 膨胀 12 px 后 inpaint，再运行网络。该方法在两个实测场景中比向外扩展 35% 的 bbox 裁剪更稳定。每个目标会额外保存 `model_input.png`、`prepare_metadata.json` 和 `quality_metrics.json`；如果热图峰值不在目标的预期操作区域，流程会在 SAM3D 前终止。

运行：

```bash
./scripts/run_titration_two_object.sh \
  --rgb /path/to/rgb.png \
  --depth /path/to/depth.png \
  --intrinsics /path/to/intrinsics.json \
  --calibration /path/to/calibration.json
```

可以通过 `--gpu 1` 指定物理 GPU，通过 `--output-dir` 指定产物目录。如果没有单独的 table mask，SAM3D 会使用自带的桌面估计。
深度默认按毫米 PNG 处理（`--depth-scale 0.001`），如果原始深度已经是米，使用 `--depth-scale 1.0`。

眼在手上时，世界坐标还依赖拍照时刻的末端位姿。推荐把两个矩阵放入同一个 JSON：

```json
{
  "T_ee_cam": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
  "T_world_ee": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
}
```

也可以让 `--calibration` 直接指向 `T_world_cam`/`T_base_cam` 4x4 矩阵。如果两个矩阵分开保存，用 `--calibration T_ee_cam.npy --world-ee T_world_ee.npy`。示例结构见 `configs/titration_calibration.example.json`。

主要最终产物：

- `04_targets/target_centers_world.txt`
- `04_targets/target_centers_overlay.png`
- `04_targets/titration_targets.json`
- `04_targets/titration_scene_camera.ply`
- `04_targets/titration_scene_world.ply`
- `04_targets/targets_only_world.ply`
- `04_targets/dropper_model_surface_world.ply`
- `04_targets/beaker_model_surface_world.ply`
- `04_targets/dropper_depth_world.ply`
- `04_targets/beaker_depth_world.ply`

PLY 颜色约定：滴瓶模型为青色，烧杯模型为绿色，胶头中心球为洋红色，烧杯口中心球为黄色，桌面为灰色。

## 分阶段复用

只做 2D affordance 到 3D mesh 投影：

```bash
PYTHONPATH=src python -m language_affordance_grasp project \
  --heatmap /path/to/affordance_heatmap.npy \
  --mesh /path/to/optimized_mesh_camera.ply \
  --intrinsics-json /path/to/intrinsics.json \
  --output-dir /path/to/projection_out
```

只过滤已有 GraspNet 输出：

```bash
PYTHONPATH=src python -m language_affordance_grasp filter \
  --raw-grasps /path/to/grasp_candidates_raw.npy \
  --region-npz /path/to/affordance_region_camera.npz \
  --instruction "where should i interact to open the jar lid?" \
  --t-base-cam /path/to/T_base_cam.npy \
  --output-dir /path/to/filter_out
```

只导出选中抓取：

```bash
PYTHONPATH=src python -m language_affordance_grasp export \
  --filtered-json /path/to/grasp_candidates_filtered.json \
  --t-base-cam /path/to/T_base_cam.npy \
  --grasp-index 0 \
  --output-dir /path/to/robot_out
```

## 主要输出

一次完整运行会生成：

- `01_affordance/affordance_heatmap.npy`：FastSAM 语言条件 affordance。
- `00_segmentation/mask.png`：未传入 `--mask` 时由 YOLOv5 自动导出的目标二值 mask。
- `01_affordance/affordance_overlay.png`：2D affordance 可视化。
- `02_sam3d/pose_result.json`：SAM3D 位姿/尺度/mesh 汇总。
- `02_sam3d/optimized_mesh_camera.ply`：目标物体 camera-frame mesh。
- `02_sam3d/combined_table_object_camera_pointcloud.ply`：GraspNet 输入场景点云。
- `03_projection/affordance_surface_camera.ply`：带 affordance 颜色的可见表面点。
- `03_projection/affordance_region_camera.npz`：高响应 3D ROI 点、法向、score、像素。
- `04_graspnet/grasp_candidates_raw.npy/.txt`：GraspNet 原始候选。
- `04_graspnet/grasp_candidates_filtered.npy/.txt/.json`：按 affordance ROI 过滤和重排后的候选。
- `05_robot/selected_grasp.json`：相机系、基坐标系、TCP 预抓/抓取/撤退/抬升位姿。
- `summary.json`：本次运行的所有关键路径和结果汇总。

## 坐标系

SAM3D 当前脚本使用的相机约定是：

```text
x = - (u - cx) / fx * z
y = - (v - cy) / fy * z
z = depth
```

本项目的投影函数也使用同一约定：

```text
u = -x / z * fx + cx
v = -y / z * fy + cy
```

真机执行阶段使用：

```text
T_base_grasp = T_base_cam @ T_cam_grasp
```

## 真机建议

第一版只生成“任务相关区域内的夹爪抓取/接触位姿”。真正开广口瓶平盖通常还需要瓶身固定和旋盖动作 primitive。实机前建议按顺序验证：

1. 检查 `affordance_overlay.png` 是否落在 lid/body 正确区域。
2. 检查 `02_sam3d/overlay.png` 和 `alignment_best_step_0000.ply` 是否对齐。
3. 检查 `03_projection/affordance_region_camera.ply` 是否投到正确 3D 区域。
4. 检查 `04_graspnet/grasp_candidates_filtered.json` 中候选宽度、中心和接近角。
5. 先 dry-run 打印机器人位姿，再低速真机执行。



## 实验步骤（服务器）
```bash
cd /home/dsj/code/language_affordance_grasp

  export LAG_API_TOKEN='与本地客户端相同的Token'
  export LAG_EYE_IN_HAND_CALIBRATION=/home/dsj/code/language_affordance_grasp/configs/T_ee_cam_affordance.json

  ./scripts/start_complex_demo_api.sh
```
