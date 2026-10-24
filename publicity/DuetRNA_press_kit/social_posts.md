# DuetRNA 社交平台短介绍

## 小红书

**给 RNA 两个坐标系，会发生什么？**

RNA 的碱基负责配对和堆叠，核糖–磷酸骨架负责连接与局部构象。我们在 DuetRNA 中为它们各建一个坐标系，再学习二者的相对位置与朝向，让模型同时生成 RNA 序列和三维结构。

视频展示双框架联合设计方法与 40–140 nt 的生成结构。关键发现是：只增加一个框架还不够，两个框架需要学会如何共同工作。

论文：https://tinyurl.com/3w25b62s

网站：https://tinyurl.com/48e7h5hb

#RNA #人工智能 #AI4Science #生成模型 #科研分享


## LinkedIn

How should a generative model represent an RNA nucleotide?

In DuetRNA, we give it two coordinate frames: one anchored to the base, the other to the sugar. They capture complementary geometry—base pairing and stacking on one side, backbone reconstruction on the other.

Coupled flow matching and relative-pose supervision bring the two frames together. Sequence and torsion predictions then produce an RNA sequence and a heavy-atom structure.

Our ablations highlight why the relationship matters: adding a second frame alone was not enough. The 31-second demo presents the dual-frame approach and generated RNA structures across lengths.

Work from Wave Intelligence Lab and CSE at The Chinese University of Hong Kong, with Junzhe Li, Lijian Peng, Yuhao Li, Yize Zhou, Hanqun Cao, Cheng Tan, and Shengchao Liu.

Paper: https://www.biorxiv.org/content/10.64898/2026.08.18.745543v1

Project and interactive samples: https://xjunli.github.io/DuetRNA/

#RNA #AIforScience #GenerativeModels


## X

DuetRNA pairs base and sugar frames to co-design RNA sequences and 3D structures.

Paper: https://www.biorxiv.org/content/10.64898/2026.08.18.745543v1

Project: https://xjunli.github.io/DuetRNA/
