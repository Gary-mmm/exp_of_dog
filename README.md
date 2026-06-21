# CKKS-MLaaS：基于全同态加密的猫狗分类隐私推理系统

## 设计思路

本项目实现了一套基于 CKKS 全同态加密的端到端机器学习即服务（MLaaS）隐私推理系统，以 CIFAR-10 猫狗二分类为应用场景。核心目标是：客户端将图像加密后发送给服务端，服务端在完全无法看到明文图像的情况下完成神经网络推理，将加密的分类结果返回给客户端解密。

整体框架分为四个阶段：（1）**预备实验**——验证 CKKS 同态运算的正确性、近似误差特征及噪声预算限制；（2）**明文预训练**——搭建密文友好的 CNN（用平方函数替代 ReLU、平均池化替代最大池化），在本地训练并导出权重；（3）**密码学上下文构建**——配置 CKKS 参数、生成公私钥和评估密钥；（4）**端到端加密推理**——基于 Flask 搭建服务端、Tkinter 编写 GUI 客户端，实现加密图像传输、密文盲推理、客户端解密与性能评估的完整链路。

---

## 核心理论与安全分析

### 问题一：从 LWE 代数结构解释乘法噪声膨胀与 CNN 架构约束

#### 1.1 RLWE 密文代数结构

Ring-LWE (RLWE) 密文定义在多项式环 $R_q = \mathbb{Z}_q[X]/(X^N+1)$ 上，其中 $N$ 是 2 的幂（我们使用 $N=16384$），$q$ 是系数模数。

一个 RLWE 密文是多项式对 $(a, b) \in R_q^2$：

$$b = a \cdot s + m + e$$

- $a \in R_q$：随机多项式（系数均匀分布在 $\mathbb{Z}_q$）
- $s \in R_q$：秘密钥多项式（系数小，通常 $\{0, \pm1\}$）
- $m$：编码后的明文消息
- $e \in R_q$：**噪声项**（系数小，$\|e\|_\infty \approx B$）

安全性本质：给定 $(a, b)$，由于 $e$ 的存在，区分 $m$ 在计算上是困难的（LWE 假设）。

#### 1.2 加法：线性噪声增长 $O(n)$

对两个密文 $(a_1, b_1)$ 和 $(a_2, b_2)$ 做同态加法：

$$b_1 + b_2 = (a_1 + a_2) \cdot s + (m_1 + m_2) + \underbrace{(e_1 + e_2)}_{\text{噪声直接相加}}$$

噪声的 $\ell_\infty$ 范数：

$$\|e_{\text{add}}\|_\infty = \|e_1 + e_2\|_\infty \leq \|e_1\|_\infty + \|e_2\|_\infty \approx 2B$$

$n$ 次连续加法后噪声 $\leq nB$，呈**线性增长**。

**实验验证**（N=16384, CKKS）：

```
  +5次：max_error = 1.95 × 10⁻⁸
 +10次：max_error = 3.58 × 10⁻⁸
 +15次：max_error = 5.20 × 10⁻⁸
 +20次：max_error = 6.83 × 10⁻⁸
```

20 次加法后误差仅增长 3.5 倍——线性关系，且绝对误差极小。

#### 1.3 乘法：二次噪声膨胀 $O(q \cdot B^2)$

两个密文相乘（张量积形式）：

$$\begin{aligned}
b_1 \cdot b_2 &= (a_1 s + m_1 + e_1)(a_2 s + m_2 + e_2) \\
&= a_1 a_2 \cdot s^2 + \underbrace{a_1(m_2+e_2) \cdot s + a_2(m_1+e_1) \cdot s}_{\text{交叉项}} + (m_1+e_1)(m_2+e_2)
\end{aligned}$$

**关键项**是 $\color{red}{a_1 e_2}$ 和 $\color{red}{a_2 e_1}$：

$$\|a_1 e_2\|_\infty \approx \|a_1\|_\infty \cdot \|e_2\|_\infty \approx \mathbf{q \cdot B}$$

因为 $a_1$ 的系数在 $\mathbb{Z}_q$ 中均匀分布，$\|a_1\|_\infty \approx q/2 \approx 2^{\text{bitlength}-1}$；而 $e_2$ 是小噪声（编码精度级别）。**两者的乘积将噪声放大 $q$ 倍**——这是从 $B$ 到 $qB$ 的指数级跳跃。

乘法后还需**重线性化**（将 $s^2$ 降为 $s$），引入额外噪声 $\approx B_{\text{relin}}$。

CKKS 方案通过**重缩放**（Rescaling）来缓解：

$$\text{Rescale}(ct) = \left\lfloor \frac{ct}{q_i} \right\rceil$$

每次重缩放将 scale 从 $S^2$ 降回 $S$，同时将噪声约化为 $(q/q_i) \cdot B$。**代价是消耗模数链中的一个素数** $q_i$，即消耗一个"层级"。

**实验验证**：

```
x × 2¹：true_err = 7.0      可用层级: 6
x × 2²：true_err = 14.0     可用层级: 5
x × 2³：true_err = 28.0     可用层级: 4
...
x × 2⁷：true_err = 448.0    可用层级: 0
x × 2⁸：层级耗尽! (ValueError: scale out of bounds)
```

每次乘法噪声呈指数增长，且层级递减。第 8 次乘法时模数链耗尽，计算崩溃。

#### 1.4 对 CNN 架构的硬约束

我们的模数链配置 `[60, 40, 40, 40, 40, 40, 40, 60]` 包含 8 个素数，中间有 7 个可消耗的层级。加密 CNN 推理链路的乘法深度分析：

```
操作                    │ 乘法次数 │ 累计层级消耗
────────────────────────┼─────────┼────────────
im2col_encoding (编码)   │    0    │    0
conv2d_im2col (卷积)     │    1    │    1
square (平方激活)         │    1    │    2
per-channel matmul (FC)  │    1    │    3
square (平方激活)         │    1    │    4
matmul (FC)              │    1    │    5
────────────────────────┼─────────┼────────────
总计                     │    5    │  余量仅 2 层
```

这直接导致 CNN 架构必须遵循三重约束：

| 约束 | 原因 | 后果 |
|------|------|------|
| **必须用 $x^2$ 替代 ReLU** | ReLU 需要比较运算或高阶多项式逼近（≥10 次乘法） | 准确率损失 ~15-20% |
| **必须用 AvgPool/stride 替代 MaxPool** | MaxPool 需要逐元素比较 | 特征提取能力下降 |
| **只能使用单层卷积** | 第 2 层 Conv 需要将密文输出重新 im2col 编码，但密文无法送入 `im2col_encoding` | 感受野受限 |

完整约束链：

```
LWE 噪声二次膨胀
    ↓
乘法 → 必须重缩放 → 消耗层级
    ↓
层级有限 (7层) → 最多 5-7 次乘法
    ↓
CNN 架构必须:
  • ReLU → x² (仅 1 次乘法)
  • MaxPool → AvgPool / stride
  • 多层Conv → 单层Conv + stride
  • 深层网络 → 浅层 (Conv→Square→FC→Square→FC)
    ↓
准确率上限 ≈ 70% (vs ReLU CNN 的 90%+)
```

**突破路径**：自举（Bootstrapping）可在密文状态下"刷新"层级。HEonGPU 在 RTX 4090 上 CKKS Slim Bootstrapping 耗时 ~99ms（N=65536），N=16384 预计 ~30ms。加入 1-2 次自举可支持 2-3 层卷积，使准确率接近明文水平。

#### 1.5 能否通过增加模数链素数来提高深度？

**直接结论：不能。** 在固定 $N=16384$ 下，无论加多少 40-bit 素数，最大乘法深度始终卡在 **7**。

实验数据（不同素数数量的 CKKS 上下文，测试连续自乘至层级耗尽）：

| 素数数量 | 总位数 | 理论深度 | **实测最大乘法** | 安全性 |
|---------|--------|---------|----------------|--------|
| 6 素 `[60,40×4,60]` | 280 bit | 4 | 6 | 128 bit |
| 7 素 `[60,40×5,60]` | 320 bit | 5 | 7 | 128 bit |
| **8 素 `[60,40×6,60]`** | **360 bit** | **6** | **7** | **128 bit** |
| 9 素 `[60,40×7,60]` | 400 bit | 7 | 7 | 128 bit |
| 10 素 `[60,40×8,60]` | 440 bit | 8 | 7 | ~124 bit |
| 11 素 `[60,40×9,60]` | 480 bit | 9 | 7 | ~44 bit ⚠️ |

**核心发现**：从 7 素到 11 素，深度始终卡在 7 不变。增加素数只会降低安全性（480 bit 仅 ~44 bit 安全），而毫无深度收益。

**物理本质**：CKKS 重缩放要求每层剩余模数必须大于编码 scale $S = 2^{40}$，否则解密失真：

$$\text{可用深度} \approx \frac{\log_2(q_{\text{total}}) - \log_2(S)}{\text{每层模数消耗}} = \frac{\sum \text{bits} - 40}{40}$$

但实际的**深度天花板由 $N$（多项式次数）决定**，因为模数链总长度受同构加密安全性约束（$q$ 不能无限增大，否则格归约攻击可行）。

**不同 $N$ 下的深度上限**（固定 8 素数 360 bit 配置）：

| 多项式次数 $N$ | 最大乘法深度 | 公钥序列化大小 | 单次乘法耗时 | 适用场景 |
|--------------|------------|--------------|------------|---------|
| $N = 4096$ | 1 | ~0.6 MB | ~10 ms | 仅单次乘法 |
| $N = 8192$ | **3** | ~2.6 MB | ~30 ms | Tiny MLP (FC→Square→FC) |
| $N = 16384$ | **7** | ~18 MB | ~100 ms | 当前 CNN (5层, 余2层) |
| $N = 32768$ | **13** | ~123 MB | ~400 ms | 多层 CNN (≥3 Conv) |

**深度 ∼ $N$ 的解释**：CKKS 中每层重缩放消耗一个素数 $q_i$，$q_i$ 的大小受 $N$ 约束——$q_i \equiv 1 \pmod{2N}$（NTT 友好素数条件）。$N$ 越大，可选的 NTT 友好素数越少，每个素数的相对"消耗"越大。更本质地说，LWE 安全强度由 $(N, \log q)$ 二元组决定，128 bit 下 $\log q$ 的上限随 $N$ 增大而提升：$N=8192 \to \log q \lesssim 200$，$N=16384 \to \log q \lesssim 438$，$N=32768 \to \log q \lesssim 890$。

**N=32768 的代价**：虽然深度达 13 层，但公钥大小 123 MB，单次推理通信量将达 GB 级别，且计算速度慢 4-8 倍，在实际 MLaaS 场景中不可接受。

**结论：当前 $[60, 40\times6, 60]$ 就是 N=16384 下的最优配置。** 要真正突破深度限制，**自举（Bootstrapping）是唯一可行路线**：

| 方案 | 可支持 Conv 层数 | 通信开销 | 计算开销 |
|------|-----------------|---------|---------|
| 当前（N=16384, 8素） | 1 层 | 18 MB/请求 | ~5 s |
| +素数（N=16384, 11素） | 仍 1 层（深度未增） | 18 MB | ~5 s |
| N→32768 | 2-3 层 | 123 MB/请求 | ~30 s |
| **N=16384 + 1 次自举** | 2 层 | 18 MB/请求 | ~5.1 s |
| **N=16384 + 2 次自举** | 3 层 | 18 MB/请求 | ~5.2 s |

---

### 问题二：平方激活的致命缺陷与缓解策略

#### 2.1 传统深度学习为何极少使用 $f(x)=x^2$

ReLU 是主流激活函数，其成功有三个关键属性：

| 属性 | ReLU $f(x)=\max(0,x)$ | 平方 $f(x)=x^2$ |
|------|----------------------|-----------------|
| **梯度稳定性** | $f'(x) \in \{0,1\}$，梯度不放大不缩小 | $f'(x)=2x$，$|x|>1$ 时梯度爆炸，$|x|<1$ 时梯度消失 |
| **稀疏性** | 50% 神经元输出 0，天然稀疏 | 所有神经元始终激活，无稀疏性 |
| **尺度保持** | 正半轴线性保范数 | 输入 >1 被放大、<1 被压缩 |

**平方激活的本质问题**：$f'(x) = 2x$ 意味着梯度与输入值成正比。在深层网络中，这导致梯度以指数速率爆炸或消失。

#### 2.2 深层平方网络的梯度灾难

考虑一个 $L$ 层网络，每层为 $h_i = (W_i h_{i-1})^2$。反向传播时：

$$\frac{\partial \mathcal{L}}{\partial h_0} = \prod_{i=1}^{L} \frac{\partial h_i}{\partial h_{i-1}} = \prod_{i=1}^{L} 2W_i^T \cdot \text{diag}(W_i h_{i-1})$$

梯度范数的缩放因子约等于 $\prod_{i=1}^{L} 2\|W_i\| \cdot \|h_{i-1}\|$。

- 若 $\|h_{i-1}\| > 0.5$：$\|h_i\| = \|W_i h_{i-1}\|^2$ 爆炸式增长，经过 3 层后梯度溢出 $\to$ **NaN**
- 若 $\|h_{i-1}\| < 0.5$：梯度以 $(0.5)^L$ 衰减，深层权重无更新 $\to$ **梯度消失**

**实验**（本项目）：

```
激活  │ 3层网络 train_loss  │ 梯度范数
──────┼────────────────────┼──────────
x²    │ 0.69 (几乎不下降)    │ 振荡 ±10³
x⁴    │ 28.81 (直接爆炸)    │ NaN @ epoch 1
ReLU  │ 0.12 (稳定下降)     │ 稳定 ±0.5
```

$x^4$ 激活在第一轮就梯度爆炸（loss=28.81），完全无法训练。

#### 2.3 工程缓解手段

| 手段 | 机制 | 本项目实现 |
|------|------|-----------|
| **BatchNorm + 融合导出** | BN 将激活值归一化到 $N(0,1)$，使平方输入集中在 $[-1,1]$，梯度不爆炸 | ✅ `fuse_conv_bn` / `fuse_linear_bn` |
| **梯度裁剪** | 将梯度范数限制在阈值内，防止单步更新过大 | ✅ `clip_grad_norm_(max_norm=5.0)` |
| **小学习率 + CosineAnnealing** | 降低步长，让参数在损失景观中平滑演化 | ✅ `lr=3e-4`, `CosineAnnealingLR` |
| **Dropout** | 随机丢弃神经元，等效于训练多个子网络，提高鲁棒性 | ✅ `Dropout2d(0.3)`, `Dropout(0.3)` |
| **Weight Decay** | L2 正则化约束权重大小，间接约束激活值范围 | ✅ `weight_decay=5e-4` |
| **浅层架构** | 限制网络深度 ≤ 3 层，将梯度链长度控制在安全范围 | ✅ 1 Conv + 2 FC |

**效果**：应用全部缓解手段后，单层 Conv + 双层 FC（共 3 个非线性层）的平方激活网络在 CIFAR-10 猫狗二分类上达到 **69.8%** 的测试准确率。无这些手段的网络仅 ~50%（随机水平）。

**为什么不能完全解决**：BN 和 Dropout 缓解了梯度问题，但 $x^2$ 激活本身的信息损失无法补救——它不像 ReLU 那样能学习非线性决策边界，表达能力上限就是约 70%。

---

### 问题三：模型逆向攻击与完整防御协议

#### 3.1 攻击模型：通过 Logits 逆向模型权重

**攻击原理**：客户端每发送一次加密图像 $x_i$，解密后获得精确的 logits 向量 $f_W(x_i) \in \mathbb{R}^2$。对于本项目网络：

$$f_W(x) = W_2 \cdot \sigma\left(W_1 \cdot \sigma\left(\text{Conv}(x; W_{\text{conv}})\right)\right),\quad \sigma(z)=z^2$$

这是一个关于输入 $x$ 的 4 次多项式函数。收集足够多的 $(x_i, f_W(x_i))$ 对：

- **线性层**：若知道中间层输出 $h = \sigma(\text{Conv}(x))$，则 $f = W_2 \cdot \sigma(W_1 h)$ 可直接求解 $W_1, W_2$
- **多项式方法**：将 $f_W(x)$ 视为 $x$ 的 4 次齐次多项式，用多项式插值恢复系数
- **优化方法**：最小化 $\|f_{\hat{W}}(x_i) - f_W(x_i)\|^2$ 用梯度下降搜索 $\hat{W}$

参数总量约 4.2M，理论上 $O(d)$ 次查询即可提取。即使每次查询只获得 argmax 标签，决策边界攻击（Decision Boundary Attack）仍可在 $O(d \log d)$ 次查询内恢复权重。

#### 3.2 防御体系：三级递进方案

##### Level 1：输出标签化（防精确 logits 泄露）

不返回 logits 向量，只返回加密的 one-hot 标签：

$$\text{enc\_diff} = \text{enc\_y}[0] - \text{enc\_y}[1]$$
$$\text{enc\_sign} = -0.5 \cdot \text{enc\_diff}^3 + 1.5 \cdot \text{enc\_diff}$$
$$\text{enc\_label} = \text{enc\_sign} \times 0.5 + 0.5$$

使用 3 次多项式逼近 sign 函数，仅 2 次乘法。客户端每查询仅获得 1 bit 信息，提取 4.2M 参数至少需要 4.2M 次查询。

##### Level 2：差分隐私噪声（防统计推断）

在标签化之前注入 Laplace 噪声：

$$\text{enc\_diff}' = \text{enc\_diff} + \text{Enc}(\eta),\quad \eta \sim \text{Laplace}(0, \Delta f/\varepsilon)$$

- $\Delta f$：敏感度（相邻输入对 logits 的最大影响）
- $\varepsilon$：隐私预算（越小越安全，典型值 0.1-1.0）
- 提供 $(\varepsilon, \delta)$-差分隐私保证

##### Level 3：Merkle 权重承诺 + zk-CNN 协议

完整协议设计如下（协议流程序列见附录）：

**Phase 0：一次性 Setup**

```
服务端:
  1. 训练模型，获得权重 W = {conv_w, conv_b, fc1_w, fc1_b, fc2_w, fc2_b}
  2. 计算权重 Merkle 承诺:
     leaves = {SHA256(w_i.tobytes() || name_i) for each parameter}
     com_W = MerkleRoot(leaves)
  3. 将 com_W 发布于公共审计日志（区块链/公告板/HTTPS 证书透明度）
  4. 部署 zk-SNARK 电路 C，定义约束:
     - 公共输入:  com_x (输入承诺), com_y (输出承诺), com_W (权重承诺)
     - 私有输入 (witness):  x (图像), y (类别), W (权重)
     - 约束:     y == CNN(x, W)  ∧  com_W == MerkleRoot(serialize(W))
```

**Phase 1：客户端推理请求**

```
客户端:
  1. 加载图像 x ∈ ℝ^(3×32×32)
  2. 生成 CKKS 上下文 (pk, sk, evk)
  3. 逐通道 im2col 编码 + 加密:
     enc_x = [im2col_encode(pk, x[c], 3, 3, stride, padding) for c in 0..2]
  4. 计算输入承诺:  com_x = SHA256(serialize(x) || client_nonce)
  5. 发送 → {
       enc_x, windows_nb,
       public_context_bytes(pk, evk),  // 含评估密钥
       com_x                          // 输入承诺（用于后续验证）
     }
```

**Phase 2：服务端加密计算 + 输出保护**

```
服务端:
  1. 反序列化 public_context，恢复评估环境（无私钥，无法解密 enc_x）
  2. 执行同态 CNN 推理:
     enc_y = CKKS_CNN(enc_x, W)  // 纯密态下完成全套运算

  3. 【防逆向攻击：标签化】
     enc_diff = enc_y[0] - enc_y[1]
     enc_diff = enc_diff + Enc(Laplace(0, Δf/ε))  // 差分隐私噪声
     enc_sign = -0.5 × enc_diff³ + 1.5 × enc_diff  // 多项式sign逼近
     enc_label = enc_sign × 0.5 + 0.5              // 映射到 {0,1}

  4. 返回 → {
       enc_label,                                    // 加密标签（非 logits）
       server_timestamp,                             // 时间戳
       merkle_partial_path(W, challenge_seed)        // 部分Merkle路径
     }
```

**Phase 3：客户端验证**

```
客户端:
  1. 解密 enc_label → label ∈ {0, 1}
  2. 可选验证:
     a) 检查 Merkle 路径 → 确认服务端使用了已承诺的权重
     b) 本地明文推理 → 比对 label 是否与本地 CNN(x) 一致（能力更强的客户端）
     c) 记录 (com_x, label, timestamp) 到本地审计日志

  3. 若 label 与本地推理不符 → 触发争议:
     - 发布 (com_x, label, merkle_path) 到审计日志
     - 任何第三方可复现验证
```

**Phase 4：定期审计**

```
审计节点:
  1. 从审计日志读取 {com_W, com_x, label, merkle_path}
  2. 验证 Merkle 路径 → 确认权重未被篡改
  3. 从 Merkle 路径中提取权重片段，独立运行 CNN(com_x) → 比对 label
  4. 若不一致 → 服务端作弊，权重承诺 com_W 被证伪
```

#### 3.3 zk-CNN 扩展（协议框架，非本次实现）

完整的 zk-CNN 方案（参考 Liu et al., CCS 2021）在 Phase 2 和 Phase 3 之间插入零知识证明层：

```
Phase 2-bis：服务端生成 zk 证明

服务端:
  5. 计算 zk 证明 π:
     π = zkSNARK.Prove(
       public_input  = {com_x, com_y, com_W},
       private_input = {x, y, W},
       circuit       = C  // "y == CNN(x, W) ∧ com_W == Merkle(W)"
     )
  6. 返回 → { enc_label, π, merkle_path }

Phase 3-bis：客户端验证 zk 证明

客户端:
  1. 解密 enc_label → label
  2. zkSNARK.Verify(π, {com_x, com_y, com_W}) → True/False
     // 零知识: 验证通过即确认"服务端使用了正确的权重 W",
     //         但客户端不获得任何关于 W 的额外信息
  3. 验证 Merkle 路径一致性
```

**zk-CNN 安全保证**：

| 属性 | 保证 |
|------|------|
| **完备性** | 诚实服务端的证明始终通过验证 |
| **可靠性** | 使用错误权重的证明无法通过验证（概率压倒性） |
| **零知识** | 客户端从 π 中无法提取 W 的任何信息 |
| **简洁性** | π 的大小为常数（~200 bytes for Groth16），与模型大小无关 |
| **不可伪造** | 客户端无法伪造 π 以声称获得了不同的推理结果 |

**实用替代方案**（无需完整 zkSNARK）：
- **交互式随机抽查**：客户端随机指定要揭示的部分权重，服务端开启 Merkle 路径。多次抽查后，客户端统计置信度。
- **可信执行环境（TEE）**：在 Intel SGX / AMD SEV 中运行推理，硬件保证代码完整性。
- **MPC 分片**：$W = W_A + W_B$ 分存两个非共谋服务器，客户端分别查询后本地求和。

#### 3.4 防御有效性分析

| 攻击类型 | 无防御 | Level 1 (标签化) | Level 2 (+DP噪声) | Level 3 (+Merkle+zk) |
|---------|--------|-----------------|-------------------|---------------------|
| Logits 解方程组 | ~4M 查询 | ~32M 查询 (1bit/查询) | 理论上不可行 (DP保证) | 不可行 |
| 决策边界攻击 | ~d·log d | ~10× 更困难 | 统计不可区分 | 不可行 |
| 权重篡改(恶意服务端) | 无法检测 | 无法检测 | 无法检测 | Merkle路径验证 |
| 推理结果伪造 | 无法检测 | 无法检测 | 无法检测 | zk证明验证 |

---

## 项目结构

### 双模型架构

本项目包含两个互补的加密推理模型，分别对应不同的实验目标和复杂程度：

#### Tiny 模型（`he_tiny_model.py`）—— 快速验证全链路

```
客户端预处理:  AvgPool2d(4) → 3×32×32 → 3×8×8 → Flatten → 192 维特征向量
服务端密文计算:
  Enc(x_192) → matmul(W_fc1) + b_fc1 → square() → matmul(W_fc2) + b_fc2 → Enc(logits_2)
```

| 属性 | 值 |
|------|-----|
| 客户端操作 | AvgPool + Flatten（明文） → 加密 192 维向量 |
| 加密操作 | **纯 FC 层**（无卷积） |
| 乘法深度 | 3（matmul + square + matmul） |
| 参数量 | ~7K |
| 准确率 | ~61% |
| 单次推理耗时 | ~1.0 s |
| 用途 | **验证 CKKS 序列化、网络通信、解密精度等全链路正确性** |

**设计意图**：在纯 FC 网络上快速跑通"客户端加密 → 序列化传输 → 服务端密文推理 → 客户端解密评估"的完整流程，避免在复杂 CNN 上调试密码学问题。

#### 完整 HE-CNN 模型（`he_cnn_model.py`）—— 同态卷积验证

```
客户端预处理:  AvgPool2d(2) → 3×32×32 → 3×16×16 → 逐通道 im2col 编码
服务端密文计算:
  im2col(Enc(x_c)) → conv2d_im2col(W_conv) + b_conv → square()
  → 逐通道 matmul(W_fc1切片) + b_fc1 → square() → matmul(W_fc2) + b_fc2 → Enc(logits_2)
```

| 属性 | 值 |
|------|-----|
| 客户端操作 | AvgPool + 逐通道 im2col 编码（3 个加密向量） |
| 加密操作 | **Conv(3→32, 3×3, pad=1) + 2 FC** |
| 乘法深度 | 5（conv + square + fc1 + square + fc2） |
| 参数量 | ~4.2M |
| 准确率 | **69.8%** |
| 单次推理耗时 | ~5.2 s |
| 用途 | **验证同态卷积的可行性与精度** |

**设计意图**：在 Tiny 模型验证全链路正确后，引入 `im2col_encoding` + `conv2d_im2col` 实现真正的密文卷积。为避开 `pack_vectors` 的 scale 问题，FC 层采用 `sum_i matmul(ch_i, W_i.T) + bias` 的逐通道累加实现。

#### 明文参考模型（`train_catdog.py`）—— CatDogSquareCNN

```
Conv(3→32, 5×5) → BN → x² → AvgPool(2) → Conv(32→64, 5×5) → BN → x² → AvgPool(2)
→ Flatten → FC(4096→256) → BN → x² → FC(256→2)
```

| 属性 | 值 |
|------|-----|
| 结构 | **2 Conv + 2 FC**，完整 CNN |
| 加密域可执行 | ❌ 含 2 层卷积，无法全部加密 |
| 准确率 | 77%（未达 90% 目标，持续优化中） |
| 用途 | **明文训练 + BN 融合导出**，为加密模型提供架构参考 |

#### 模型对比一览

```
            Tiny 模型          HE-CNN 模型         明文 CatDogSquareCNN
           ──────────         ───────────         ────────────────────
加密Conv        ✗                   ✓                       ✗
Conv 层数       0                   1                       2
FC 层数         2                   2                       2
激活函数       x²                  x²                      x²
池化      客户端AvgPool(4)    客户端AvgPool(2)         AvgPool(2)
参数量        ~7K               ~4.2M                  ~1.4M
准确率        ~61%              ~69.8%                   ~77%
推理耗时      ~1.0s              ~5.2s                    N/A
```

### 文件结构

### 文件结构

```
exp_of_dog/
├── 明文预训练（子实验一）
│   └── train_catdog.py          # CatDogSquareCNN: 2Conv+2FC, BN融合导出
│
├── 预备实验
│   └── test.py                  # CKKS 基本运算 / BFV噪声预算 / CKKS层级耗尽
│
├── 密码学上下文（子实验二）
│   └── he_context.py            # CKKS上下文 (N=16384, 8素数), 密钥生成, 序列化
│
├── Tiny 模型 —— 快速全链路验证（子实验三·轻量版）
│   ├── he_tiny_model.py         # 模型定义、导出、加载、密文推理 (纯FC)
│   ├── train_he_tiny.py         # 训练: AvgPool→FC→Square→FC
│   ├── he_server.py             # Flask 服务端 (端口 5000)
│   ├── he_client.py             # 命令行客户端
│   └── gui_client.py            # Tkinter GUI 客户端
│
├── HE-CNN 模型 —— 同态卷积验证（子实验三·进阶版）
│   ├── he_cnn_model.py          # 模型定义、BN融合导出、加载 (3ch RGB, pad-1 Conv)
│   ├── he_cnn_ops.py            # im2col编码 / conv2d_im2col / 逐通道matmul / 完整管线
│   ├── he_cnn_server.py         # Flask 服务端 (端口 5001)
│   ├── he_cnn_client.py         # 命令行客户端
│   └── train_he_cnn.py          # 训练: Conv(3→32)→Square→FC(8192→512)→Square→FC→2
│
├── 安全防御
│   └── zk_protocol.py           # Merkle承诺 / 多项式sign / 标签化 / 审计日志
│
├── README.md                    # 本文件（技术文档）
├── report.md                    # 实验报告
├── artifacts/                   # 模型权重与参数
└── data/                        # CIFAR-10 数据集（自动下载）
```

## 环境依赖

- Python 3.9+
- PyTorch ≥ 1.13
- TenSEAL ≥ 0.3
- Flask, Pillow, NumPy, requests, torchvision

```powershell
pip install torch torchvision numpy tenseal flask pillow requests
```

## 使用方法

### 1. 预备实验：验证 CKKS 同态特性

```powershell
python test.py
```

### 2. 子实验一：明文 CNN 训练与参数导出

```powershell
python train_catdog.py --epochs 60 --batch-size 128
```

### 3. 端到端加密推理（Tiny 模型）

```powershell
python train_he_tiny.py --epochs 10 --batch-size 256 --hidden-size 32 --cpu
python he_server.py                    # 终端一
python he_client.py --samples 5        # 终端二
python gui_client.py                   # 终端三（GUI）
```

### 4. 完整 HE-CNN 推理

```powershell
python train_he_cnn.py --epochs 40 --out-channels 32 --hidden-size 256
python he_cnn_server.py                # 终端一
python he_cnn_client.py --samples 10   # 终端二
```

### 5. 防御协议演示

```powershell
python zk_protocol.py
```

## 当前实验结果

| 阶段 | 模型 | 测试准确率 | 备注 |
|------|------|-----------|------|
| 明文 CNN | CatDogSquareCNN (2Conv+2FC) | 77% | 带 BN 融合 |
| Tiny 模型 | AvgPool+FC+Square+FC | 61% | 极轻量，验证全链路 |
| **HE-CNN v1** | 灰度 32×32, stride-2 Conv | 67.6% | 单通道，颜色信息丢失 |
| **HE-CNN v2** | **RGB 16×16, pad-1 Conv** | **69.8%** | 3 通道，当前最佳 |
| HE-CNN 密文推理 | 明/密文 logits 误差 | < 0.001 | 预测 100% 一致 |

## 关键设计决策

| 决策 | 原因 |
|------|------|
| BN 融合导出 | 训练时用 BN 稳定梯度，导出时熔合进权重使密文推理图退化为纯线性+平方 |
| 逐通道 matmul 替代 pack_vectors | `pack_vectors` 引入额外 scale 导致偏置无法对齐，逐通道累加避开了该问题 |
| stride-2 替代 AvgPool | TenSEAL 无显式 Galois 旋转 API，无法在密文域做池化；stride 卷积等价于下采样 |
| 3ch RGB 替代灰度 | 猫狗毛色是关键分类特征，灰度图损失 ~3% 准确率 |
| 8 素数模数链 | 提供 7 个层级，CNN 推理消耗 5 层（Conv+Square+FC+Square+FC），余量 2 层 |

## 测试方法与评估指标

| 指标 | 含义 |
|------|------|
| `he_accuracy` | 密文推理解密后的分类准确率 |
| `avg_abs_plain_he_logit_diff` | 明文 logits 与密文解密 logits 的平均绝对误差 |
| `avg_server_inference_seconds` | 服务端纯密文推理平均耗时 |
| `end_to_end_traffic_mb` | 请求+响应的总通信量 |
