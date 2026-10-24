How should a generative model represent an RNA nucleotide?

In DuetRNA, we give it two coordinate frames: one anchored to the base, the other to the sugar. They capture complementary geometry—base pairing and stacking on one side, backbone reconstruction on the other.

Coupled flow matching and relative-pose supervision bring the two frames together. Sequence and torsion predictions then produce an RNA sequence and a heavy-atom structure.

Our ablations highlight why the relationship matters: adding a second frame alone was not enough. The 31-second demo presents the dual-frame approach and generated RNA structures across lengths.

Work from Wave Intelligence Lab and CSE at The Chinese University of Hong Kong, with Junzhe Li, Lijian Peng, Yuhao Li, Yize Zhou, Hanqun Cao, Cheng Tan, and Shengchao Liu.

Paper: https://www.biorxiv.org/content/10.64898/2026.08.18.745543v1

Project and interactive samples: https://xjunli.github.io/DuetRNA/

#RNA #AIforScience #GenerativeModels
