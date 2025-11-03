# Rust で始める GPU ネイティブ機械学習【第 0 版】

> Rust、GPU、深層学習の三位一体で学ぶ、次世代機械学習基盤の実装ガイド

## ⚠️ Disclaimer

**本書の内容はすべて生成 AI (ChatGPT, Claude) によって執筆されています。**

そのため、以下の点にご注意ください：

- 技術的な誤り、不正確な情報が含まれている可能性があります
- コード例が最新のライブラリバージョンと互換性がない場合があります
- 数式や実装の詳細に誤りがある可能性があります
- 引用された論文や文献の情報が不正確な場合があります

**誤りを発見された場合は、以下の方法でお知らせください：**

1. **Pull Request**: 修正内容を直接提案していただく（最も推奨）
2. **Issue**: 問題点を報告していただく
3. **その他の方法**: 直接ご連絡いただく

本書の品質向上に向けて、皆様のご協力をいただけますよう何卒よろしくお願い申し上げます。

---

## 📚 本書について

本書は、Rust 言語を用いて GPU ネイティブな機械学習システムを構築するための実践的ガイドブックです。PyTorch や TensorFlow などの既存フレームワークのブラックボックスを開き、内部メカニズムを理解しながら、安全で高性能な機械学習基盤を自ら実装する方法を学びます。

### Python との比較で学べること

本書の最大の特徴は、**Python（PyTorch/NumPy）との徹底比較**を通じて理解を深める点です：

| 学習項目 | Python で学ぶこと | Rust で深まること |
|---------|-----------------|------------------|
| **型システム** | 実行時のエラー | コンパイル時の型安全性保証 |
| **メモリ管理** | GC任せ（ブラックボックス） | 所有権・ライフタイムで完全制御 |
| **GPU 制御** | 抽象化されたAPI | CUDA/ROCmの直接制御と最適化 |
| **並行処理** | GIL の制約 | 真の並列実行とゼロコスト抽象化 |
| **エラー処理** | 例外ベース | Result型による明示的なエラーハンドリング |
| **パフォーマンス** | インタプリタのオーバーヘッド | ネイティブコードの最適化 |

各章で **Python コード → Rust コード** の対応を示し、段階的に実装の深層を理解できる構成になっています。

### なぜ GPU ラーニングに Rust が適しているのか？

**1. メモリ安全性 × 低レベル制御の両立**

```rust
// Rust: コンパイル時にメモリ安全性を保証しながら、GPUメモリを直接制御
let gpu_buffer: CudaBuffer<f32> = device.allocate(size)?;  // ✅ 型安全
// Python: 実行時エラーのリスク、メモリリークの可能性
```

**2. ゼロコスト抽象化**

- Python: 柔軟性はあるが、抽象化のコストが実行時に発生
- Rust: 抽象化のコストがゼロ（コンパイル時に最適化）

```rust
// 高レベルAPIと低レベルAPIのコストが同じ
tensor.matmul(&other)  // 抽象化されたAPI
cuda_gemm(...)         // 低レベルCUDA呼び出し
// → コンパイル後のパフォーマンスは同等
```

**3. 並行・並列処理の安全性**

| 言語 | 課題 | Rust の解決策 |
|------|------|--------------|
| **Python** | GIL（Global Interpreter Lock）による並列化の制約 | GIL不要、真の並列実行 |
| **C/C++** | データ競合のリスク | Send/Sync トレイトによるコンパイル時検証 |

**4. エコシステムの成長**

- **burn**: Rust ネイティブの深層学習フレームワーク
- **candle**: HuggingFace製の軽量推論エンジン
- **wgpu**: クロスプラットフォームGPU API

**5. プロダクション環境での優位性**

| 要件 | Python の課題 | Rust の強み |
|------|--------------|------------|
| **デプロイ** | 依存関係の複雑さ | 単一バイナリで配布可能 |
| **保守性** | 型ヒントの限界 | 強力な型システム |
| **パフォーマンス** | インタプリタのオーバーヘッド | ネイティブコードの高速性 |
| **信頼性** | 実行時エラー | コンパイル時の大半のバグ検出 |

### 本書で得られる実践的スキル

✅ **理論と実装の統合**: 数式 → Python → Rust の3段階で深層学習の原理を完全理解  
✅ **GPU最適化技術**: メモリ coalescing、shared memory、kernel fusion など実践的な最適化手法  
✅ **型安全な設計**: Rustの型システムを活かした堅牢な機械学習ライブラリの設計  
✅ **パフォーマンスチューニング**: Nsight、perf、flamegraphを使った本格的なプロファイリング  
✅ **実用モデル実装**: CNN、RNN、Transformer を Rust + GPU でゼロから構築  
✅ **本番環境スキル**: テスト、CI/CD、エラーハンドリングのベストプラクティス

### 対象読者

- Python（NumPy/PyTorch）での機械学習経験があり、Rust に興味がある方
- 深層学習フレームワークの内部メカニズムを理解したい方
- GPU プログラミングの基礎から応用まで体系的に学びたい方
- プロダクション環境での高性能・高信頼性な ML システムを構築したい方
- 「なぜこのコードが速いのか」を理論的に説明できるようになりたい方

### 学習の進め方

1. **第0部〜第I部**: 環境構築と基礎理論（Python経験者は復習として）
2. **第II部〜第III部**: Rustの特性とGPUプログラミングの核心
3. **第IV部〜第VI部**: 実装を通じた深い理解（最も重要）
4. **第VII部**: 最新動向と今後の展望

各章の「Pythonとの比較」セクションを重点的に読むことで、既存知識を活かしながら効率的に学習できます。

---

## 📖 目次

### [第 0 部：環境構築とクイックスタート](./00_第0部_環境構築とクイックスタート/)

本書を始める前に必要な環境構築と、最初の GPU プログラムを実行します。

#### [第 0 章：開発環境のセットアップ](./00_第0部_環境構築とクイックスタート/00-00-開発環境のセットアップ.md)

- 0.1 Rust のインストールと設定
- 0.2 CUDA Toolkit / ROCm のインストール
- 0.3 必要なクレートとツールチェーン
- 0.4 Hello World: 最初の GPU プログラム
- 0.5 トラブルシューティングと FAQ

---

### [第 I 部：基礎理論と全体像](./01_第I部_基礎理論と全体像/)

深層学習と GPU の基礎、線形代数、自動微分の原理を学びます。

#### [第 1 章：GPU ネイティブ機械学習とは何か](./01_第I部_基礎理論と全体像/01-01-GPUネイティブ機械学習とは何か.md)

- 1.1 CPU と GPU の構造比較
- 1.2 ディープラーニング計算の特徴（行列演算・並列性）
- 1.3 GPU 最適化の目的と限界
- 1.4 Rust で実装する利点（安全性・性能・所有権モデル）
- 1.5 Python/Rust エコシステムギャップと選択指針

#### [第 2 章：線形代数と数値計算の基礎](./01_第I部_基礎理論と全体像/01-02-線形代数と数値計算の基礎.md)

- 2.1 ベクトル・行列・テンソル
- 2.2 行列積と畳み込みの計算量
- 2.3 メモリレイアウトとキャッシュ効率
- 2.4 BLAS/LAPACK の仕組みと役割
- 2.5 数値安定性と精度

#### [第 3 章：自動微分の仕組み](./01_第I部_基礎理論と全体像/01-03-自動微分の仕組み.md)

- 3.1 Forward/Reverse モード AD
- 3.2 計算グラフと勾配伝播
- 3.3 メモリ再利用とテープ設計
- 3.4 Rust での実装例（dfdx の設計を題材に）
- 3.5 静的計算グラフと動的計算グラフ

---

### [第 II 部：Rust による数値処理と安全設計](./02_第II部_Rustによる数値処理と安全設計/)

Rust の型システム、所有権、並列処理を活用した安全な数値計算を実装します。

#### [第 4 章：Rust 数値計算の基礎構文](./02_第II部_Rustによる数値処理と安全設計/02-04-Rust数値計算の基礎構文.md)

- 4.1 所有権・借用・ライフタイム
- 4.2 ndarray と nalgebra によるテンソル演算
- 4.3 unsafe ブロックを局所化する設計
- 4.4 FFI と `#[repr(C)]` の整合性確認
- 4.5 ゼロコスト抽象化とコンパイル時最適化

#### [第 5 章：並列計算と非同期処理](./02_第II部_Rustによる数値処理と安全設計/02-05-並列計算と非同期処理.md)

- 5.1 CPU 並列: rayon によるデータ並列
- 5.2 非同期 I/O と計算の分離
- 5.3 マルチスレッドとアロケータ最適化
- 5.4 メモリ安全性を保つスレッド通信設計
- 5.5 チャネルと共有状態の使い分け

---

### [第 III 部：GPU プログラミング入門](./03_第III部_GPUプログラミング入門/)

GPU アーキテクチャの理解から、Rust による実践的な GPU プログラミングまで。

#### [第 6 章：GPU アーキテクチャの理解](./03_第III部_GPUプログラミング入門/03-06-GPUアーキテクチャの理解.md)

- 6.1 スレッド・ブロック・ワープの階層
- 6.2 メモリ階層とアクセスパターン（グローバル・シェアード・レジスタ）
- 6.3 L1/L2 キャッシュ・テクスチャメモリ・コンスタントメモリ
- 6.4 同期・バリア・バンクコンフリクト
- 6.5 Occupancy（占有率）とリソース制約
- 6.6 GPU プロファイリングと指標（FLOPS・帯域・Roofline モデル）

#### [第 7 章：Rust から GPU を操作する](./03_第III部_GPUプログラミング入門/03-07-RustからGPUを操作する.md)

- 7.1 CUDA と ROCm の基本 API
- 7.2 cust/cudarc を使ったカーネル呼び出し
- 7.3 wgpu によるプラットフォーム非依存 GPU 実装
- 7.4 Rust-GPU / SPIR-V でシェーダを書く
- 7.5 PTX アセンブリと低レベル最適化

#### [第 8 章：GPU メモリ管理と最適化](./03_第III部_GPUプログラミング入門/03-08-GPUメモリ管理と最適化.md)

- 8.1 ホスト ⇔ デバイス転送コストと最小化戦略
- 8.2 ピン留めメモリ・Unified Memory・ゼロコピー
- 8.3 メモリ合体（Coalescing）アクセスパターン
- 8.4 ストリーム・イベント・非同期実行
- 8.5 複数 GPU・デバイス選択とスケジューリング
- 8.6 メモリプール・カスタムアロケータ

---

### [第 IV 部：機械学習エンジンの構築](./04_第IV部_機械学習エンジンの構築/)

テンソル演算、学習ループ、最適化アルゴリズムを実装し、実用的な ML エンジンを構築します。

#### [第 9 章：テンソル・オペレーター設計](./04_第IV部_機械学習エンジンの構築/04-09-テンソル・オペレーター設計.md)

- (執筆中) 9.1 テンソル構造体とストライド設計
- (執筆中) 9.2 動的形状とバッチ処理への対応
- (執筆中) 9.3 基本演算（加算・積・畳み込み）の GPU 実装
- (執筆中) 9.4 カーネル融合（Fusion）による最適化
- (執筆中) 9.5 勾配計算と逆伝播実装
- (執筆中) 9.6 ブロードキャスティングとビュー操作

#### [第 10 章：学習ループと最適化手法](./04_第IV部_機械学習エンジンの構築/04-10-学習ループと最適化手法.md)

- 10.1 forward → loss → backward → optimizer の流れ
- 10.2 SGD / Adam / AdamW / RMSProp の実装
- 10.3 勾配クリッピング・学習率スケジューラ
- 10.4 mixed precision（FP16/BF16）・量子化・sparsity
- 10.5 勾配チェックポイント（Gradient Checkpointing）
- 10.6 Rust での再現: burn / tch-rs 内部構造解析

#### [第 11 章：デバッグとプロファイリング](./04_第IV部_機械学習エンジンの構築/04-11-デバッグとプロファイリング.md)

- 11.1 数値計算の検証（Python出力との比較）
- 11.2 GPU プロファイリング（NVIDIA Nsight, rocm-profiler）
- 11.3 CPU プロファイリング（cargo-flamegraph, perf）
- 11.4 メモリリーク・競合検出（valgrind, miri, sanitizers）
- 11.5 テスト自動化と CI/CD
- 11.6 エラーハンドリングのベストプラクティス

---

### [第 V 部：応用と高度化](./05_第V部_応用と高度化/)

ONNX 互換、分散学習、セキュリティなど、プロダクション環境での実践技術を学びます。

#### [第 12 章：モデル推論と ONNX 互換](./05_第V部_応用と高度化/05-12-モデル推論とONNX互換.md)

- 12.1 推論と学習の違い
- 12.2 ONNX フォーマットの理解
- 12.3 Rust での ONNX 推論（tract, onnxruntime-rs）
- 12.4 推論最適化テクニック（Graph Fusion, 量子化）
- 12.5 動的バッチング
- 12.6 推論サーバの構築

#### [第 13 章：コンパイラ最適化と DSL 設計](./05_第V部_応用と高度化/05-13-コンパイラ最適化とDSL設計.md)

- 13.1 機械学習コンパイラの概要
- 13.2 JIT コンパイルと実行時最適化
- 13.3 Rust マクロによる DSL 設計
- 13.4 LLVM / MLIR による最適化パイプライン
- 13.5 Triton 風のカーネル記述言語
- 13.6 テンプレートメタプログラミングと型レベル計算

#### [第 14 章：分散・クラスタ対応](./05_第V部_応用と高度化/05-14-分散・クラスタ対応.md)

- 14.1 なぜ分散学習が必要か？
- 14.2 データ並列（Data Parallelism, ZeRO）
- 14.3 Pipeline Parallelism（GPipe）
- 14.4 Tensor Parallelism（Megatron-LM）
- 14.5 NCCL と集団通信（AllReduce, AllGather, Reduce-Scatter）
- 14.6 RPC・gRPC によるノード通信
- 14.7 フォールトトレランスとチェックポイント

#### [第 15 章：セキュリティと信頼性](./05_第V部_応用と高度化/05-15-セキュリティと信頼性.md)

- 15.1 メモリ安全と GPU クラッシュハンドリング
- 15.2 FFI 境界の監査・unsafe 最小化
- 15.3 ライブラリ設計における抽象化戦略
- 15.4 ベンチマーク・テスト自動化・再現性確保
- 15.5 Differential Testing と Fuzzing

---

### [第 VI 部：ケーススタディと実践](./06_第VI部_ケーススタディと実践/)

CNN、RNN、Transformer などの実用的なモデルを Rust で実装し、エンドツーエンドで最適化します。

#### [第 16 章：CNN・RNN・Transformer を実装する](./06_第VI部_ケーススタディと実践/06-16-CNN・RNN・Transformerを実装する.md)

- (執筆中) 16.1 Convolution 層とバックプロパゲーション
- (執筆中) 16.2 Im2Col / Winograd / FFT による畳み込み最適化
- (執筆中) 16.3 LSTM/GRU のメモリアクセス最適化
- (執筆中) 16.4 Transformer の注意機構を Rust で構築
- (執筆中) 16.5 Flash Attention / Multi-Query Attention の実装
- (執筆中) 16.6 推論速度・メモリ比較

#### [第 17 章：エンドツーエンド最適化](./06_第VI部_ケーススタディと実践/06-17-エンドツーエンド最適化.md)

- (執筆中) 17.1 プロファイル解析とボトルネック検出
- (執筆中) 17.2 カーネル最適化・パイプライン分割
- (執筆中) 17.3 演算子融合とグラフ最適化
- (執筆中) 17.4 モデルデプロイ・推論サーバ構築
- (執筆中) 17.5 WebGPU 経由でのブラウザ推論
- (執筆中) 17.6 組み込み・エッジデバイスへの展開

#### [第 18 章：実践プロジェクト](./06_第VI部_ケーススタディと実践/06-18-実践プロジェクト.md)

- (執筆中) 18.1 画像分類モデルの学習と推論
- (執筆中) 18.2 自然言語処理タスクの実装
- (執筆中) 18.3 強化学習エージェントの構築
- (執筆中) 18.4 生成モデル（VAE / GAN）の実装
- (執筆中) 18.5 パフォーマンス比較とベンチマーク

---

### [第 VII 部：展望と設計指針](./07_第VII部_展望と設計指針/)

Rust 機械学習エコシステムの現状と未来を展望します。

#### [第 19 章：Rust での ML エコシステムの進化](./07_第VII部_展望と設計指針/07-19-RustでのMLエコシステムの進化.md)

- (執筆中) 19.1 Burn / Candle / Linfa / dfdx の方向性
- (執筆中) 19.2 PyTorch/TensorFlow との API 互換戦略
- (執筆中) 19.3 Rust が担う高信頼 ML インフラの未来
- (執筆中) 19.4 コミュニティとオープンソース活動
- (執筆中) 19.5 今後の学習ロードマップ

---

## 🚀 はじめに

本書は、以下の順序で学習を進めることを推奨します：

1. **第 0 部**で開発環境を構築し、最初の GPU プログラムを動かす
2. **第 I 部**で基礎理論を理解する（特に第 1 章は必読）
3. **第 II 部**で Rust の数値計算と並列処理を習得する
4. **第 III 部**で GPU プログラミングの基礎を学ぶ
5. **第 IV 部**で実際の機械学習エンジンを構築する
6. **第 V 部**で応用技術（分散学習、コンパイラ最適化）を学ぶ
7. **第 VI 部**でケーススタディと実践プロジェクトに取り組む
8. **第 VII 部**でエコシステムの全体像と今後の展望を把握する

### 学習の進め方

**初学者向け**：第 0 部から順番に読み進めることをお勧めします。特に、第 0 章から第 3 章までは本書全体の基礎となるため、丁寧に読み進めてください。

**中級者向け**：既に CUDA や GPU プログラミングの経験がある方は、第 II 部から始めて、Rust 特有の実装パターンを重点的に学ぶことができます。

**上級者向け**：特定のトピック（分散学習、コンパイラ最適化など）について深く学びたい方は、第 V 部から直接アクセスできますが、第 III 部・第 IV 部の理解を前提としています。

各章は独立性を保ちつつ、前の章の知識を前提としています。コードサンプルは実際に動作するものを提供し、演習問題を通じて理解を深められるよう配慮しています。

---

## 📝 ライセンスと貢献

本書は学習・教育目的で作成されています。コードサンプルは自由に使用・改変できますが、商用利用の際はご連絡ください。

誤字・脱字、技術的な誤りを見つけた場合は、Issue または Pull Request でお知らせいただけると幸いです。

---

## 📚 参考文献

今後の記事執筆にあたり、以下の文献を参考資料としてご活用いただけます。

### 論文（arXiv 等）

#### GPU 最適化・並列処理

- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**  
  Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (2022)  
  https://arxiv.org/abs/2205.14135  
  _Transformer の注意機構を高速化する画期的な手法_

- **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**  
  Tri Dao (2023)  
  https://arxiv.org/abs/2307.08691  
  _Flash Attention のさらなる改善版_

- **Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations**  
  Philippe Tillet, H.T. Kung, David Cox (2019)  
  https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf  
  _GPU カーネルを高レベルに記述するための DSL_

#### 分散学習・大規模モデル

- **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**  
  Mohammad Shoeybi et al. (2019)  
  https://arxiv.org/abs/1909.08053  
  _モデル並列化による大規模言語モデルの学習_

- **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**  
  Samyam Rajbhandari et al. (2019)  
  https://arxiv.org/abs/1910.02054  
  _メモリ最適化による超大規模モデル学習_

- **GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism**  
  Yanping Huang et al. (2019)  
  https://arxiv.org/abs/1811.06965  
  _パイプライン並列化の手法_

#### 数値計算・最適化

- **Mixed Precision Training**  
  Paulius Micikevicius et al. (2017)  
  https://arxiv.org/abs/1710.03740  
  _FP16/FP32 混合精度学習の基礎_

- **Highly Scalable Deep Learning Training System with Mixed-Precision**  
  Xianyan Jia et al. (2018)  
  https://arxiv.org/abs/1807.11205  
  _ImageNet を 4 分で学習する混合精度システム_

#### グラフ処理・GPU 応用

- **Gunrock: A High-Performance Graph Processing Library on the GPU**  
  Yangzihao Wang et al. (2015)  
  https://arxiv.org/abs/1501.05387  
  _GPU 上での高性能グラフ処理_

- **MaxK-GNN: Extremely Fast GPU Kernel Design for Accelerating Graph Neural Networks Training**  
  Hongwu Peng et al. (2023)  
  https://arxiv.org/abs/2312.08656  
  _GNN トレーニング高速化のためのカーネル設計_

### 公式ドキュメント・ガイド

#### NVIDIA CUDA

- **CUDA C++ Programming Guide**  
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/  
  _CUDA プログラミングの公式ガイド（必読）_

- **CUDA C++ Best Practices Guide**  
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/  
  _パフォーマンス最適化のベストプラクティス_

- **cuBLAS Documentation**  
  https://docs.nvidia.com/cuda/cublas/  
  _BLAS 演算の高速ライブラリ_

- **cuDNN Developer Guide**  
  https://docs.nvidia.com/deeplearning/cudnn/developer-guide/  
  _深層学習向け最適化ライブラリ_

- **Nsight Compute Documentation**  
  https://docs.nvidia.com/nsight-compute/  
  _CUDA カーネルのプロファイリングツール_

#### AMD ROCm

- **ROCm Documentation**  
  https://rocm.docs.amd.com/  
  _AMD ROCm の公式ドキュメント_

### 書籍

#### C/C++による GPU プログラミング

- **Programming Massively Parallel Processors: A Hands-on Approach (4th Edition)**  
  David B. Kirk, Wen-mei W. Hwu (2022)  
  _CUDA プログラミングの定番教科書_  
  目次：GPU Architecture, CUDA Programming Model, Memory Hierarchy, Performance Optimization, etc.

- **CUDA C プロフェッショナルプログラミング**  
  John Cheng, Max Grossman, Ty McKercher（日本語訳版）  
  _CUDA の実践的なプログラミング技法_

- **GPU 並列図形処理入門 ― CUDA・OpenGL の導入と活用**  
  伊藤 智義 編著 (2014)  
  https://gihyo.jp/book/2014/978-4-7741-6304-8  
  _日本語の GPU プログラミング入門書_

- **GPU プログラミング入門 ― CUDA5 による実装**  
  伊藤 智義 編 (2013)  
  https://www.kspub.co.jp/book/detail/1538207.html  
  _CUDA5 を用いた実装解説_

#### Python による GPU 機械学習

- **Python によるディープラーニング**  
  François Chollet 著、巣籠 悠輔 訳 (2018)  
  https://tatsu-zine.com/books/deeplearning-with-python  
  _Keras の作者によるディープラーニング解説_

- **ゼロから作る Deep Learning ❸ ― フレームワーク編**  
  斎藤 康毅 著 (2020)  
  _深層学習フレームワークを自作して理解する_

- **機械学習と深層学習 ― Python によるシミュレーション**  
  小高 知宏 著 (2019)  
  https://www.ohmsha.co.jp/book/9784274222269/  
  _Python での機械学習シミュレーション_

- **Python によるはじめての機械学習プログラミング**  
  技術評論社 (2019)  
  https://gihyo.jp/book/2019/978-4-297-10525-9  
  _初学者向けの機械学習入門_

- **Python 機械学習プログラミング 第 3 版**  
  Sebastian Raschka, Vahid Mirjalili 著  
  _scikit-learn, TensorFlow, PyTorch を用いた実践_

#### 深層学習全般

- **Deep Learning (Adaptive Computation and Machine Learning series)**  
  Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)  
  https://www.deeplearningbook.org/  
  _深層学習の理論的基礎（通称「花本」）_

- **ディープラーニング（岡谷貴之 著）**  
  機械学習プロフェッショナルシリーズ (2015)  
  _日本語での深層学習理論解説_

### オンラインリソース

#### チュートリアル・記事

- **PyTorch CUDA Semantics**  
  https://pytorch.org/docs/stable/notes/cuda.html  
  _PyTorch での CUDA 使用方法_

- **CUDA チュートリアル（理化学研究所）**  
  _CUDA プログラミングの基礎から応用まで_

- **The Rust Performance Book**  
  https://nnethercote.github.io/perf-book/  
  _Rust のパフォーマンス最適化_

#### コミュニティ・フォーラム

- **NVIDIA Developer Forums**  
  https://forums.developer.nvidia.com/  
  _CUDA 開発者向けフォーラム_

- **Rust Users Forum**  
  https://users.rust-lang.org/  
  _Rust ユーザーコミュニティ_

---

**Happy Learning! 🦀✨**
