# 内容与素材出处

## 本版的主要依据

作者提供的《Base-and-Sugar Dual-Frame Flow Matching for RNA Co-Design》，29 页版本（本次提供的 `G:\我的云端硬盘\Super Projects\DuetRNA\Pub\iclr\DuetRNA (2).pdf`）。本次调整沿用原版图文组合；科研经历依据作者在对话中提供的描述。

## 各页对应

当前展示 7 张。原第 2 张研究历史暂收在 `cards.html` 的 `template#omitted-history` 中；原始 article ID 保留，展示页码连续。图卡移除章节号和表号，详细出处统一保留在本文件。

| 图卡 | 主要内容 | 出处 |
|---|---|---|
| 01 | DuetRNA 项目封面 | 作者提供的 RNA 示意图；论文摘要 |
| 02 | Base-Plane / Sugar-GS 的分工 | 论文 Figure 1、§3.1 |
| 03 | 碱基框架稳定配对关系；双系改善碱基重建，同时保留核糖框架的骨架重建精度 | 论文第 7 页 Table 1、§4.2、附录 D；具体口径见下文 |
| 04 | 双框架 flow matching、预测终点相对位姿监督、序列与扭转角输出 | 论文 §3.2–3.6、附录 C；原发布包的几何示意素材 |
| 05 | IF / RhoFold、GS / Boltz-1 的配对比较 | 论文 Table 2、Table 4、§4.1–4.5 |
| 06 | 单核糖框架与无相对位姿监督消融 | 论文 Table 3(a)、§4.4 |
| 07 | 结构表示与模型性能；领域共性挑战及 DuetRNA 尚未解决的问题 | 作者稿第 9 页 Conclusion；RNA-FrameFlow §4.4、§5、§7；RiboGen §3.1。领域背景与模型具体局限的归属见下文。 |

## 第 3 页：围绕两类几何组织证据

删去数据规模的并列展示，保留能够解释双系选择的两条结果。均为训练生成模型之前、基于真实结构参考系的表示分析，不是生成样本的结构误差。

- **配对关系稳定性**：canonical cWW 配对的多状态旋转漂移，Sugar-GS 为 13.35°，Base-Plane 为 7.78°；对应相对平移漂移为 1.70 Å 与 0.61 Å，未在本版图卡重复展示。
- **互补重建**：碱基原子重建 RMSD，Sugar-GS 为 6.21 Å，Base-Plane + Sugar-GS 为 1.58 Å。骨架 RMSD 均为 1.88 Å，因此表述为“同时保留核糖框架的骨架重建精度”。单独 Base-Plane 的骨架 RMSD 为 2.98 Å。
- 双系重建中，碱基原子来自碱基通道，骨架与桥接原子来自核糖通道；这里的互补性不等于已解决最终生成结构的原子化学问题。

完整研究比较 7 种参考系构造，覆盖 11,497 条静态 RNA 链及 31,432 个多状态关系组；规模数字保留在来源说明中。

## 第 5、6 页：评估含义与完整统计口径

第 5 页在结果下方各用一句话解释 IF 与 GS，图中百分比均为 scTM ≥ 0.45 的计算自洽率。评估长度 40–150 nt，每种方法 600 个样本；两组结果为各自协议下的配对比较。

- **IF / RhoFold**：对生成结构，用 gRNAde 设计 8 条候选序列，再用 RhoFold 折叠，以最佳 scTM 衡量生成骨架的可设计性。RNA-FrameFlow 为 41.00%，DuetRNA 为 48.67%，提高 7.67 个百分点；差值 95% CI 为 [2.04, 13.23]，p = 0.0076。
- **GS / Boltz-1**：直接折叠模型共同生成的序列，衡量生成序列与结构的匹配。RiboGen 为 34.17%，DuetRNA 为 38.50%，数值提高 4.33 个百分点；差值 95% CI 为 [−1.11, 9.74]，p = 0.119，未达到统计显著。因此图卡仅描述数值变化，不声称统计显著提升。
- 主表 IF 结果来自 4 次独立训练中的最佳检查点；4 次训练的 IF 均值为 45.68% ± 2.33%。检查点选择、训练均值和置信区间移至本文件，不在图卡突出展示。
- 第 6 页以 **IF scTM validity** 标注纵向比较指标。三组沿用相同 IF / RhoFold 评估协议，scTM ≥ 0.45 为有效；单核糖 34.67%，去掉相对位姿监督 30.00%，完整模型 48.67%。

## 第 7 页：领域挑战与模型局限的归属

“领域共性挑战”指 RNA 三维生成研究中仍需应对的问题，并非声称所有方法具有相同缺陷，或整个领域均未开展实验。

- **计算自洽与实验功能之间的距离**：作者稿 Conclusion 明确区分计算评估与生化功能、实验可行性。图卡将从计算指标走向实验验证列为研究方向，并明确 DuetRNA 目前仅完成计算评估。
- **局部化学合理性与链连接**：[RNA-FrameFlow](https://arxiv.org/html/2406.13839) §5 讨论原子冲突、链断裂等物理不合理结构；[RiboGen](https://arxiv.org/html/2503.02058) §3.1 也报告部分断裂或未折叠结构，以及长、复杂序列的失败情形。DuetRNA 的糖苷键精度不足与长度相关断链来自作者稿 Conclusion，是本模型的具体问题，未推及所有方法。
- **有限结构数据下的泛化与新颖性**：[RNA-FrameFlow](https://arxiv.org/html/2406.13839) §4.4、§5、§7 讨论 RNA 结构数据稀缺、训练结构分布偏倚和新颖性/有效性之间的困难。DuetRNA 的有效样本更接近训练集，验证范围限于 40–150 nt 的无条件生成；对更长 RNA 和分布外新折叠的证据不足，来自作者稿 Conclusion。

下一步围绕原子几何、链连接、更广长度范围与实验验证展开，再探索功能条件设计和多构象生成。扩大长度与补充实验验证是针对论文局限组织的后续工作表述。

## 图片

`assets/dual-frame-flow-matching.png`：由作者指定的 `G:\我的云端硬盘\Super Projects\DuetRNA\图\Dual-Frame Flow Matching.png` 原样复制；用于当前第 4 页的双坐标系流匹配示意。

`assets/rna-illustration.png`：作者提供的 RNA 结构示意图。

`assets/dual-frame-panel.png`：原图文发布包中的双框架图面板，来源为 DuetRNA Figure 1。

`assets/scene-noise.png`、`scene-middle.png`、`scene-final.png`：原图文发布包中的解释性插值图，保留原始图像字节。

图卡中的条形图由 HTML/CSS 绘制，其数据沿用论文。

## 原发布包保留的来源索引

1. **DuetRNA · 作者稿** — Base-and-Sugar Dual-Frame Flow Matching for RNA Co-Design. 方法与数据：本次作者提供的 29 页稿，§3–4、表 1–4、附录 C–E。公开预印本另见链接。
   https://www.biorxiv.org/content/10.64898/2026.08.18.745543v1

2. **Wave 的研究脉络** — Shengchao Liu 的研究主页与代表性工作；包括 GraphMVP、GeoSSL、Geom3D、ProteinDT 等。早期工作归于刘老师与合作者的研究积累。
   https://chao1224.github.io/

3. **AssembleFlow** — Hongyu Guo, Yoshua Bengio, Shengchao Liu. Rigid Flow Matching with Inertial Frames for Molecular Assembly. ICLR 2025.
   https://openreview.net/forum?id=jckKNzYYA6

4. **NeuralMD** — Shengchao Liu et al. A Multi-Grained Symmetric Differential Equation Model for Learning Protein-Ligand Binding Dynamics. Nature Communications 2025.
   https://arxiv.org/abs/2401.15122

5. **RigidSSL** — Zhanghan Ni et al. Rigidity-Aware Geometric Pretraining for Protein Design and Conformational Ensembles. ICLR 2026.
   https://arxiv.org/abs/2603.02406

6. **InertialAR** — Haorui Li et al. Autoregressive 3D Molecule Generation with Inertial Frames. ICML 2026.
   https://arxiv.org/abs/2510.27497

7. **InertialGenome** — Yize Zhou, Haorui Li, Shengchao Liu. A Resolution-Agnostic Geometric Transformer for Chromosome Modeling Using Inertial Frame. 2026.
   https://openreview.net/forum?id=OwLl8Xi6JG

8. **RNA-FrameFlow** — Rishabh Anand et al. Flow Matching for de novo 3D RNA Backbone Design. TMLR 2025.
   https://arxiv.org/abs/2406.13839

9. **NuFold** — Yuki Kagaya et al. End-to-end approach for RNA tertiary structure prediction with flexible nucleobase center representation. Nature Communications 2025.
   https://www.nature.com/articles/s41467-025-56261-7

10. **RiboGen** — Dana Rubin et al. RNA Sequence and Structure Co-Generation with Equivariant Multiflow. 2025.
   https://arxiv.org/abs/2503.02058

11. **RiboFlow** — Runze Ma et al. Conditional De Novo RNA Co-Design via Synergistic Flow Matching. NeurIPS 2025.
   https://arxiv.org/abs/2503.17007
