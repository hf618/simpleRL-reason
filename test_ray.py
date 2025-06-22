import ray
import os
import subprocess
import torch

# 1. 定义一个远程任务，打印环境、CUDA可用性、并尝试编译某个需要CUDA的操作
@ray.remote(num_gpus=1)
def test_cuda_in_ray():
    print("PID:", os.getpid())
    
    # 打印 LD_LIBRARY_PATH，确认是否传入成功
    print("\n[Env] LD_LIBRARY_PATH =", os.getenv("LD_LIBRARY_PATH", "Not set"))

    # 检查 PyTorch 是否能看到 CUDA
    print("\n[PyTorch] CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("[PyTorch] Current device:", torch.cuda.current_device())
        print("[PyTorch] Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("⚠️ PyTorch cannot find CUDA")

    # 可选：运行一个简单的 CUDA 操作来验证
    try:
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = torch.matmul(x, y)
        print("\n[Matmul] Result on CUDA:", z)
    except Exception as e:
        print("🚨 CUDA op failed:", str(e))

    # 可选：运行 ldconfig 查询 libcuda.so
    try:
        result = subprocess.check_output(["ldconfig", "-p", "|", "grep", "libcuda"])
        print("\n[ldconfig] Found libcuda:\n", result.decode())
    except Exception as e:
        print("🚨 ldconfig failed to find libcuda:", str(e))


if __name__ == "__main__":
    # 设置 LD_LIBRARY_PATH（根据你的情况修改）
    ld_path = "/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.6/lib64:/usr/local/cuda-12.6/targets/x86_64-linux/lib"

    # 启动 Ray，带环境变量传递
    ray.init(
        runtime_env={
            "env_vars": {
                "LD_LIBRARY_PATH": ld_path
            }
        }
    )

    # 提交任务
    ray.get(test_cuda_in_ray.remote())