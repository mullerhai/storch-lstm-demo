# MNIST Classification with LSTM and RMSNorm in Scala

[![Scala Version](https://img.shields.io/badge/Scala-2.13%2F3-blue)](https://www.scala-lang.org)
[![Storch Version](https://img.shields.io/badge/Storch-0.9.0-orange)](https://github.com/mullerhai/storch)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

# MNIST Classification with LSTM and RMSNorm using Storch

An LSTM + RMSNorm model implemented with &zwnj;**Storch Deep Learning Framework**&zwnj; for sequential classification on the MNIST handwritten digit dataset. Explores the potential of non-CNN architectures in image classification tasks.

## ✨ Core Features
- &zwnj;**Innovative Architecture**&zwnj;: Processes image pixel sequences using LSTM networks (alternative to traditional CNNs)
- &zwnj;**Optimization Technique**&zwnj;: Integrated RMSNorm layer enhances training stability
- &zwnj;**Efficient Implementation**&zwnj;: Leverages Scala 3's type-safe programming and Storch's GPU acceleration
- &zwnj;**User-Friendly**&zwnj;: One-click training scripts & pre-trained model loading



使用 &zwnj;**Storch 深度学习框架**&zwnj;实现的 LSTM + RMSNorm 模型，在 MNIST 手写数字数据集上实现序列分类。探索非CNN架构在图像分类任务中的可能性。 现在可以在maven 中央仓库引用storch 相关组件

## ✨ 核心特性
- &zwnj;**创新架构**&zwnj;：用 LSTM 网络处理图像像素序列（替代传统CNN）
- &zwnj;**优化技术**&zwnj;：集成 RMSNorm 层提升训练稳定性
- &zwnj;**高效实现**&zwnj;：基于 Scala 3 类型安全编程与 Storch 的GPU加速
- &zwnj;**易用性**&zwnj;：一键式训练脚本与预训练模型加载

## 📦 依赖安装
https://central.sonatype.com/artifact/io.github.mullerhai/core_3
https://central.sonatype.com/artifact/io.github.mullerhai/vision_3

在 `build.sbt` 中添加：
```scala
libraryDependencies ++= Seq(
 "io.github.mullerhai" % "core_3" % "0.2.3-1.15.1"，
"io.github.mullerhai" % "vision_3" % "0.2.3-1.15.1"
)
