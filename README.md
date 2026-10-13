# AGT-SR：论文模型与训练代码

依据用户提供的 Revised Manuscript v2，重点对应第 3 节、Table 1、第 4.1.4 节。代码已编写，**未启动训练、未下载模型、未安装训练依赖**。

论文明确给出的结构、公式和超参数按原文实现。论文缺少的实现细节单独列在下方；仅凭本文不能声称与作者未提供的原始代码逐行一致，也不能保证复现表中数值。

## 文件

| 文件 | 内容 |
|---|---|
| `model.py` | 双路前端、wav2vec 层融合、Transformer 编解码器、APGQI、CTC heads |
| `losses.py` | 熵置信度、停止梯度的 CAO、CE + 0.3 CAO |
| `data.py` | 变长批次、外部 FMLLR 特征/变换、MUSAN、SpecAugment |
| `train.py` | 四卡 DDP、两阶段冻结、验证、日志、断点续训、best/last 权重 |
| `decode.py` | 可选的 CTC/attention 联合束搜索参考实现 |
| `build_vocab.py` | 由训练集的显式 tokens 构建词表 |
| `config.json` | 论文参数及明确标注的补充配置 |
| `requirements.txt` | 依赖；torch 与 torchaudio 必须安装同一版本 |

## 与论文对应

| 论文依据 | 代码位置/实现 |
|---|---|
| 式 (1) | `data.apply_fmllr`：f @ A.T + b，A/b 来自外部 GMM-HMM |
| 式 (2) | `DualPathFrontend`：0.2 Z6 + 0.3 Z9 + 0.5 Z12，固定系数 |
| 式 (3)-(5) | 对每条有效序列线性插值，各投影 256 维，拼接后线性映射至 512；不增加融合激活层 |
| Table 1 | encoder 12 层、decoder 6 层，512 维、8 heads、2048 FFN、dropout 0.1 |
| 式 (6)-(8) | 有效 encoder 帧均值池化、kernel=3 的时间卷积、局部先验检索 |
| 式 (9) | `APGQI.retrieval_indices`：精确 floor + clip；空 CTC 序列的首步特殊处理 |
| 式 (10)-(13) | concat(global, local) → Linear → tanh；e + 0.1 p → Linear，注入第一层 decoder 之前 |
| 式 (14)-(19) | 第 4/8/10/12 encoder 层各有一个独立 CTC head；按有效帧归一化熵计算 c、alpha 并 detach |
| 式 (20)-(21) | 每条样本 CE 是 token NLL 的求和；CTC 也是序列 NLL；样本间求均值；总损失 CE + 0.3 CAO |
| 第 3.5 节 | 最终 CTC 已属于 CAO，不额外再加一次最终层 CTC |
| 第 4.1.4 节 | AdamW，2e-4，20,000 warm-up steps，全局 batch 32，40 epochs |
| 第 4.1.4 节 | 前 10 epochs 冻结全部 wav2vec；后 30 epochs 仅微调其 Transformer encoder，卷积特征提取器一直冻结 |
| 式 (22) | 可选联合解码：CTC 0.3、attention 0.7、beam 10、无外部 LM |

CTC 输出为 `[B,T,N+1]`，词类包括 blank；decoder 输出为 `[B,U,N+3]`，屏蔽 padding/blank 与 BOS。推理长度估计严格先合并重复，再移除 blank，并在声学编码后只计算一次。

## 数据格式

音频预先准备为 **16 kHz、单声道**。每个数据集单独训练、单独词表、单独输出目录。manifest 使用 UTF-8 JSONL，每行一个样本。音频/FMLLR/transform 路径相对 manifest 所在目录解析，也可使用绝对路径。

推荐提供外部估计的说话人 FMLLR 变换：

```json
{"utt_id":"utt001","speaker":"spk001","audio":"audio/utt001.wav","transform":"transforms/spk001.npz","tokens":["h","e","l","l","o"]}
```

`transform` 是 `.npz`，包括浮点 `A`（80×80）和 `b`（80）。必须来自以匹配的 80 维 filterbank 为输入的外部 GMM-HMM；不要把常见的 40 维 MFCC/LDA 变换直接套用。代码不把 identity transform 或随机矩阵称为 FMLLR。

也支持已计算好的 `.npy`（`T×80`）：

```json
{"utt_id":"utt001","audio":"audio/utt001.wav","fmllr":"fmllr/utt001.npy","tokens":["h","e","l","l","o"]}
```

只有缓存 FMLLR 时不能直接使用在线 MUSAN：噪声改变波形后，FMLLR 必须由同一波形重新计算。默认在线 MUSAN 因此要求训练集每条样本都有 `transform`。如同时提供 `transform` 与 `fmllr`，使用前者重新提取。验证不施加噪声和 SpecAugment。

`tokens` 必须来自你选定的分词方案，以上字符只是格式示例，不代表论文指定字符词表。也支持 `token_ids`（词类 1..N），需要另外提供一致的 `vocab.json`。未知 token 会报错，不静默丢弃。CTC 目标不含 blank/BOS/EOS。

**FMLLR 辅助模型训练与变换估计属于论文的外部环节。** 本文没有提供辅助模型参数、Kaldi recipe、初始识别设置、统计量或原始变换，所以这里实现式 (1) 的应用和接口，不伪造这一环节。测试/验证的变换估计不能使用参考转录；只能使用评估协议允许的说话人分组信息与初始识别结果。在线加噪复用外部估计的说话人变换，并把加噪后的同一声学观察送入两条分支；原文未说明噪声增强后是否重新估计变换。

LibriSpeech 使用官方 960 小时训练数据及官方 test-clean/test-other；本文未明确其 dev 选择，需要在 manifest 中自行落实。CommonVoice 19.0 英文及 AISHELL-1 的自定义 8:1:1 划分未给出样本清单、划分 seed 和分组算法，需要外部提供；代码不会擅自更换为另一种划分并称为论文划分。

## 使用命令（仅供你之后使用，本次未执行）

在 `paper_model_code` 目录准备环境，安装匹配的 PyTorch/torchaudio 版本和适合服务器 CUDA 的构建，再安装其余依赖：

```bash
pip install -r requirements.txt
python build_vocab.py --train-manifest data/train.jsonl --output data/vocab.json
```

填写 `config.json` 的 train/valid manifest、词表、MUSAN 路径与输出目录。默认两种增强均开启；增强强度属于下面注明的补充选择。MUSAN 文件需预先转为 16 kHz 单声道，不自动下载数据。

论文四卡训练配置建议在 Linux GPU 服务器使用：

```bash
torchrun --standalone --nproc_per_node=4 train.py --config config.json
```

每卡 batch 8，全局 batch 32。训练固定 40 轮，无提前停止；不使用额外 label smoothing、梯度裁剪或混合精度。第 11 轮会重建 DDP，使新解冻的参数参与梯度同步。训练集末尾不足完整全局 batch 的样本当轮丢弃（每轮重新 shuffle）；验证按 rank 精确分片，不重复计算补齐样本。

续训（epoch 边界）：

```bash
torchrun --standalone --nproc_per_node=4 train.py --config config.json --resume runs/agt_sr/last.pt
```

断点保存模型、优化器、scheduler、epoch、step、各 rank 随机状态、词表和完整配置。要求续训配置与 world size 不变。`best.pt` 按验证总损失选取，`last.pt` 保存最近一轮；原文未指定模型选择准则，因此这也是补充选择。输出只有运行后才生成。

如明确需要单卡，全局 batch 仍为 32：

```bash
python train.py --config config.json --allow-other-world-size
```

这是硬件配置变化，不代表论文四卡运行。没有设置自动减小模型或 batch，单卡显存不足需要另行调整。

可选解码：

```bash
python decode.py --checkpoint runs/agt_sr/best.pt --audio example.wav --fmllr example.npy --device cuda
```

该实现逐候选计算 CTC 前缀概率并在 EOS 后使用精确序列概率，按论文 Eq. (22) 联合评分。没有额外长度奖励、外部 LM 或只按 attention 预筛候选。为了可读性，它重新计算 decoder prefix 和 CTC DP，速度慢；不能用于声称复现论文 RTF。`max_tokens=500` 是保护上限，不是论文参数。输出 tokens，需用相应 tokenizer 做 detokenization。

## 论文未公开、代码明确补足的细节

| 项目 | 本实现选择 |
|---|---|
| tokenizer、词表规模、文本训练归一化 | 由用户明确提供 tokens；不自动猜测 |
| 特殊符号与 EOS | blank/pad=0，lexical=1..N，BOS=N+1，EOS=N+2；decoder 目标含 EOS，APGQI 训练长度随之含 EOS，CTC 不含 EOS |
| 时间公共网格与插值端点 | 默认 wav2vec 时间网格，align_corners=False；按样本有效长度插值 |
| PE、embedding scale、encoder/decoder norm 顺序、FFN 激活 | 正弦 PE（从位置 0 开始）、不额外缩放 embedding、post-norm、ReLU |
| Conv1D 边界 | kernel 3、stride 1、padding 1、bias=True |
| confidence epsilon 与批次归约 | 1e-8；每样本独立算 confidence/alpha，随后对序列损失取 batch 均值 |
| AdamW beta/eps/weight decay | (0.9,0.999)/1e-8/0.01；配置可修改 |
| 学习率 warm-up 形式及后续曲线 | 线性 warm-up 到 2e-4，之后恒定；可显式改 inverse_sqrt，原文未指定 |
| MUSAN 策略 | 随机一段 MUSAN WAV，加噪概率 0.5，SNR 均匀 0..20 dB；三类文件混合抽取，没有论文未公开的类别权重 |
| SpecAugment 策略及位置 | FMLLR 路径投影之前使用两次时间遮挡（最大 40 帧）、两次频率遮挡（最大 15 维），填零、不做 time warp；原文未指定强度/位置/warp |
| fbank 细节 | torchaudio Kaldi fbank，25 ms 窗、10 ms 帧移、80 mel bins、dither=0、snip_edges=True、16 kHz，波形乘 32768；外部 GMM-HMM 必须匹配 |
| wav2vec checkpoint 的其余设置 | 沿用下载 checkpoint 的 encoder 内部设置；关闭其隐式 SpecAugment，避免另加未知策略；冻结组件 eval，第二阶段 encoder train |
| 数据/随机 seed、验证准则 | seed=42，按 manifest 划分，best 按验证 total loss；无早停 |

以上选择是完整可用代码所需的实现约定，**不能当作论文已经给出的信息**。若能取得作者原始配置、词表、划分文件与 FMLLR recipe，可逐项替换；核心公式和已明确参数已经对应本文。

## 本次检查范围

仅做 Python AST/编译级静态检查、JSON 解析以及依论文核对公式与配置。未执行模型前向/反向、未训练、未验证数据或识别准确率；运行时和多卡行为需要在有依赖和真实数据的环境验证。
