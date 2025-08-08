---

## ⚠️ 注意事项 (Important Notes)

### 关于 `train_grpo_math_tune_ray.sh`

这个脚本文件是一个**与特定机器相关的配置文件** (machine-specific)。

其中包含的路径、GPU数量、节点信息等参数，需要每一位开发者根据自己所使用的服务器环境进行修改。为了避免在协作中互相覆盖个人的配置，请遵循以下约定：

* **不要提交你对 `train_grpo_math_tune_ray.sh` 文件的任何本地修改。**
* 如果你修改了这个文件以适应你自己的环境，请在执行 `git add .` 命令前，确保这个文件没有被添加到暂存区。如果不小心添加了，可以使用 `git restore --staged train_grpo_math_tune_ray.sh` 将其移出暂存区。

### 关于 `bfloat16`

V100上不支持bfloat16，有些地方n可能得手动调整。

---