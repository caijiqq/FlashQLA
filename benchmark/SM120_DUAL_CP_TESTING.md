# FlashQLA 双 5090 测试手册

本手册用于 `feat/multi-gpu-cp-opt` 分支，在同一台 Linux 主机的两张 RTX 5090 上验证 SM120 卡内聚合和跨卡通信两项优化。正确性复用仓库现有 pytest；性能使用现有 demo。不需要新增测试代码。

建议依次执行环境准备、正确性回归、AutoCP 性能测试。先保存日志，再判断收益。文中的命令在仓库根目录、同一个 Bash 终端内执行。

**1. 本次测试的四种模式**

| 模式 | 卡内反向聚合 | 跨卡通信 |
| --- | --- | --- |
| `baseline` | 原有实现 | 前向、反向均 AllGather 状态与 M |
| `aggregate` | WS/TMA 聚合内核 | 原有通信 |
| `comm` | 原有聚合 | 前向缓存远端 M，反向仅 AllGather dh |
| `both` | WS/TMA 聚合内核 | 前向缓存远端 M，反向仅 AllGather dh |

聚合优化只替换 Dual-CP 的卡内反向状态归约，当前支持 K=V=128、K×V 布局、BF16 M、FP32 dh。AutoCP 没有选择卡内分区时不会执行这个内核。通信优化沿用现有连续、等长 token 分片约定，不限定两卡或单条序列。

库的默认入口仍使用原实现；pytest 的 `FLASHQLA_CP_OPT` 和 demo 的 `--mode` 用于本次实验。`FLASHQLA_CP_OPT` 目前只在 `tests/test_cp_e2e.py` 中读取。SM90、SM100 实现文件没有修改。

**2. 获取代码和准备环境**

首次获取：

```bash
git clone --branch feat/multi-gpu-cp-opt https://github.com/caijiqq/FlashQLA.git
cd FlashQLA
```

已有仓库则在工作区干净时更新：

```bash
git fetch origin
git switch feat/multi-gpu-cp-opt
git pull --ff-only origin feat/multi-gpu-cp-opt
```

云端需要两张可见的 SM120 GPU、可用的 NVIDIA 驱动、CUDA Toolkit 12.8 或更新版本，以及可用的 C++ 编译器。`nvidia-smi` 中的 CUDA Version 不能代替 `nvcc --version` 对编译工具链的检查。仓库最低要求见 [README](../README.md#installation)。

本地验证使用 Python 3.11.15、PyTorch 2.11.0+cu128、TileLang 0.1.15、Triton 3.6.0、pytest 9.2.0。新环境可按下面的版本安装；已有可用环境可直接检查版本。PyTorch 的 CUDA 12.8 安装源见[官方版本说明](https://pytorch.org/get-started/previous-versions/#v2110)，TileLang 安装方式见[官方安装文档](https://tilelang.com/get_started/Installation.html)。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install tilelang==0.1.15 pytest==9.2.0
python -m pip install --no-deps -e .
```

以下用 `python` 调用已激活的虚拟环境。日志放在仓库已忽略的 `tmp/` 目录。启用 `pipefail`，避免 `tee` 掩盖测试失败；`set -e` 使批量执行在失败后停止。

```bash
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
set -e
set -o pipefail
mkdir -p tmp/dualcp_sm120

git rev-parse HEAD | tee tmp/dualcp_sm120/commit.txt
git status --short | tee tmp/dualcp_sm120/worktree.txt
nvidia-smi | tee tmp/dualcp_sm120/nvidia-smi.txt
nvidia-smi topo -m | tee tmp/dualcp_sm120/topology.txt
nvcc --version | tee tmp/dualcp_sm120/nvcc.txt
g++ --version | tee tmp/dualcp_sm120/compiler.txt
python -m pip freeze > tmp/dualcp_sm120/packages.txt

python - <<'PY' | tee tmp/dualcp_sm120/environment.txt
import sys
from importlib.metadata import version
import torch
import torch.distributed as dist

print("Python:", sys.version)
for name in ("torch", "tilelang", "triton", "pytest"):
    print(name, version(name))
print("Torch CUDA:", torch.version.cuda)
print("NCCL available:", dist.is_nccl_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
if torch.cuda.device_count() >= 2:
    print("CUDA peer access:", torch.cuda.can_device_access_peer(0, 1))
print("NCCL version:", torch.cuda.nccl.version())
PY
```

预期看到两张 5090、计算能力 `(12, 0)`、NCCL 可用。记录 peer access 结果和拓扑；peer access 为 False 本身不等于 NCCL 无法工作。首次测试保持平台默认 NCCL 配置，不预先强制 P2P、IB 或算法选项。

**3. 复用原有 pytest 验证正确性**

pytest 自己通过 `mp.start_processes` 启动每卡一个进程，**不要在 pytest 外再套 torchrun**。

先运行原有上下文、分区及状态打包测试：

```bash
python -m pytest tests/test_cp_context.py -q -x \
  2>&1 | tee tmp/dualcp_sm120/pytest-context.log
```

当前应有 56 项通过。随后四种模式运行完全相同的 8 个两卡 K×V Dual-CP 用例：

```bash
for mode in baseline aggregate comm both; do
  FLASHQLA_CP_OPT="$mode" python -m pytest tests/test_cp_e2e.py \
    -k "ws2 and inter_intra and kv" -x -rs \
    2>&1 | tee "tmp/dualcp_sm120/pytest-${mode}.log"
done
```

每个模式预期 `8 passed, 44 deselected`，总计 32 项执行。`deselected` 是主动排除了其他配置；如果这里出现 `skipped`，先检查跳过原因，不能将其计为通过。

这 8 项沿用原来的输入和 FP64 参考，覆盖单条跨卡序列、无初始状态、非零初始状态和终止梯度、零衰减、packed 序列、非整 chunk 边界和 GVA。比较前向输出、序列最终状态、dq/dk/dv/db/dg 和应归属本卡的 dh0；误差判据沿用仓库原来的 2% 相对 L2 阈值。

原有 `inter_intra` 用例强制开启卡内分区，能明确执行新聚合内核。这里通过表示所测配置的优化实现通过独立参考校验；AutoCP 自行选择分区的情况由下一步 demo 覆盖。

机时特别紧时，可将上面循环的筛选条件换成下面这个，先跑每模式 3 项：

```text
kv and inter_intra and (single-ws2 or single-h0-ws2 or allflags-ws2)
```

这是快速检查；完整的两卡 Dual-CP 回归仍以上面的每模式 8 项为准。

**4. 在真实 AutoCP 下比较性能**

性能 demo 用 `torchrun` 启动。为确保调用当前虚拟环境，下面使用等价的 `python -m torch.distributed.run`。真实进程数取自 `--nproc_per_node`；demo 的 `--world-size` 只用于单卡模拟。

每条命令先检查四种模式在无种子和非零 h0/dht 下与 baseline 的一致性，再计时。即使指定单个 `--mode`，正确性仍检查四种模式，只有计时范围变化。

场景 A：一条全局长度 8192 的序列，每卡 4096 token，16 个头。

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m benchmark.demo_dual_cp \
  --mode all --heads 16 --tokens 8192 --warmup 5 --repeats 21 \
  2>&1 | tee tmp/dualcp_sm120/bench-single-h16-t8192.log
```

场景 B：三条 packed 序列，总长度仍为 8192，跨卡序列的两端都含其他序列。

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m benchmark.demo_dual_cp \
  --mode all --heads 16 --seq-lengths 521 7162 509 \
  --warmup 5 --repeats 21 \
  2>&1 | tee tmp/dualcp_sm120/bench-packed-h16-t8192.log
```

检查每个 rank 打印的 `AutoCP intra=...`、`sequences`、`parts`。若所有 rank 都是 `intra=False`，本次只测到了通信优化，无法据此评价聚合优化。可用更长序列、较少头数的配置补测：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m benchmark.demo_dual_cp \
  --mode all --heads 4 --tokens 32768 --warmup 5 --repeats 21 \
  2>&1 | tee tmp/dualcp_sm120/bench-single-h4-t32768.log
```

`--gate-scale 0` 可以额外测试无衰减的长程状态依赖；默认值为 0.05。`--seq-lengths` 指定全局序列长度并覆盖 `--tokens`，长度之和必须能被实际 GPU 数整除。

完整执行会出现 8 行 `2-GPU check ... max rel error=...`，随后输出四种模式的毫秒耗时表。demo 使用相对最大误差 `< 0.02`，并检查非有限误差；这个判据与 pytest 的相对 L2 判据不同，demo 也不替代 pytest 的 FP64 参考校验。

首次执行包含 JIT 编译，不用整条命令的墙钟时间比较性能。基准在正确性检查和预热后开始计时。计时前关闭调试插桩，确保没有其他 GPU 任务争用资源；不要开启 `CUDA_LAUNCH_BLOCKING=1` 或在 Compute Sanitizer 下采集性能结论。

**5. 如何判读四种模式的性能**

demo 报告算子前向加反向一步的 CUDA event 时间：每个样本取两个 rank 中较慢者，最后报告中位数及最小/最大值；各模式轮换测量顺序。同步 barrier 和用于汇总耗时的 AllReduce 在计时区间外。计时区间仍包含本实现的内存分配及 CPU 发射可能造成的 GPU 空隙。

这不是整模型训练吞吐，也不是单独的 NCCL 耗时或通信占比。按日志中的中位数计算：

| 比值 | 所回答的问题 |
| --- | --- |
| baseline / aggregate | 只开启聚合优化，整步是否变快 |
| baseline / comm | 只开启通信优化，整步是否变快 |
| baseline / both | 两项优化同时开启的整体收益 |
| aggregate / both | 已开启聚合时，通信优化的增量收益 |
| comm / both | 已开启通信时，聚合优化的增量收益 |

比值大于 1 表示加速。若差异很小、落在波动范围内，再独立重跑该场景确认；不把一次轻微领先判为稳定收益。两项优化的加速比不必相乘。

本 demo 的反向数据为 FP32 dh 和 BF16 M，且 K=V=128；dh-only 将每卡逻辑发送载荷从 `H×128×128×6` 字节减到 `H×128×128×4` 字节，即减少三分之一。前向通信保持不变，并增加了 M 的缓存复制；因此三分之一的载荷下降不等于三分之一的整步耗时下降。M 缓存的峰值显存影响尚未单独测量。

**6. 按需要补跑的现有测试**

若要覆盖纯 inter 和 V×K 布局，可对 `baseline`、`comm` 使用全部两卡用例。每种模式预期 32 项通过；其中包含第 3 步已跑过的 8 项，因此机时紧时不必重复整个集合。

```bash
for mode in baseline comm; do
  FLASHQLA_CP_OPT="$mode" python -m pytest tests/test_cp_e2e.py \
    -k ws2 -x -rs 2>&1 | tee "tmp/dualcp_sm120/pytest-all-ws2-${mode}.log"
done
```

`aggregate`、`both` 当前不能用于 V×K。其他原有 pytest 可以继续验证默认实现，但不会因为设置 `FLASHQLA_CP_OPT` 而自动切换到新内核。

如需把 5090 的聚合单算子表现与此前单卡结果比较，使用已有微基准。它通过 CUDA Graph 测量预分配内核，输出微秒，与第 4 步的毫秒整步计时不能直接混用。

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmark.demo_cp_scan \
  --heads 16 --segments 16 --repeats 100 \
  2>&1 | tee tmp/dualcp_sm120/bench-aggregate-only.log
```

两卡能够验证真实 NCCL 调用、跨卡边界修正及 dh-only 通信路径，也能测出两项优化在两卡场景的性能。两卡没有中间 rank，不能验证三卡以上实际消费中间卡缓存 M 的完整链条。以后有三卡资源时直接复用原有用例，不用新增测试：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 FLASHQLA_CP_OPT=comm \
  python -m pytest tests/test_cp_e2e.py \
  -k "ws3 and inter_intra and kv" -x -rs
```

**7. 失败时保留什么信息**

正确性失败先保留具体用例 ID、模式和 traceback。若 baseline 也失败，先定位原路径或环境；若 baseline 通过但某个优化模式失败，就可直接按模式隔离问题。不要放宽误差阈值或把失败自动切回 baseline。

如果遇到 NCCL 报错或超时，用最小 demo 打开一次 NCCL 日志，保留最早的错误；排障运行的耗时不纳入性能比较：

```bash
NCCL_DEBUG=INFO python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m benchmark.demo_dual_cp \
  --mode all --heads 4 --tokens 4096 --warmup 1 --repeats 1 \
  2>&1 | tee tmp/dualcp_sm120/nccl-debug.log
```

如果是 TileLang 编译失败，提供编译器报错、Torch/TileLang/CUDA 版本和 `nvcc` 输出。原有内核可能打印 JIT 警告；最终以具体错误、进程退出状态及数值断言为准。

测试完成后保留 `tmp/dualcp_sm120/`，其中包含提交号、环境、拓扑、各模式 pytest 日志及性能表。可打包后带回：

```bash
tar -czf tmp/dualcp_sm120_logs.tar.gz -C tmp dualcp_sm120
```

**8. 本次提交前已有的验证**

本地只有一张 RTX 5070 Ti Laptop GPU。现有 56 项上下文测试通过；单卡 rank 模拟的 9 种 AutoCP 配置、四种模式及有/无种子比较通过。WS 内核已有单卡压力测试以及 Compute Sanitizer 检查记录，详见 [SM120 验证记录](demo_cp_scan_results_sm120.txt)。

本地两卡 pytest 的 8 项配置因 GPU 数不足而跳过。真实双 5090 NCCL 正确性及性能由本手册的云端运行确认；单卡模拟结果不计作真实多卡测试通过。
