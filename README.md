＃DNC TensorFlow

これは最近のNatureペーパーで紹介されているDeepMindのDNC（Differentiable Neural Computer）アーキテクチャのTensorFlow実装です：
> [Graves、Alex、et al。 "ダイナミックな外部メモリを備えたニューラルネットワークを用いたハイブリッドコンピューティング"自然538.7626（2016）：471から476](http://www.nature.com/articles/nature20101.epdf?author_access_token=ImTXBI8aWbYxYQ51Plys8NRgN0jAjWel9jnR3ZoTv0MggmpDmwljGswxVdeocYSurJ3hxupzWuRNeGvvXnoO8o4jTJcnAyhGuZzXJ1GEaD-Z7E6X_a9R-xqJ9TfJWBqz)

This implementation doesn't include all the tasks that was described in the paper, but it's focused on exploring and re-producing the general task-independent key characteristics of the architecture. However, the implementation was designed with extensibility in mind, so it's fairly simple to adapt it to further tasks.

この実装には、このペーパーで説明したすべてのタスクは含まれていませんが、アーキテクチャーの一般的なタスクに依存しない重要な特徴の探索と再作成に重点を置いています。しかし、実装は拡張性を念頭に置いて設計されているため、それを次のタスクに適用するのはかなり簡単です。

##ローカル環境仕様

すべての実験とテストは、次のマシンで実行されました。
- Intel Core i5 2410M CPU @ 2.30GHz（ハイパースレッディングが有効な2つの物理コア）
- 4GB SO-DIMM DDR3 @ 1333MHz
- GPUなし。
- Ubuntu 14.04 LTS
- TensorFlow r0.11
- Python 2.7

##実験

###動的メモリメカニズム

This experiment is designed to demonstrate the various functionalities of the external memory access mechanisms such as in-order retrieval and allocation/deallocation.

この実験は、外部メモリアクセス機構の様々な機能、例えば、順序検索と割り振り/割り振り解除を示すように設計されています。

A similar approach to that of the paper was followed by training a 2-layer feedforward model with only 10 memory locations on a copy task in which a series of 4 random binary sequences each of which is of size 6 (24 piece of information) was presented as input. Details about the training can be found [here](tasks/copy/).

論文のそれと同様のアプローチに続いて、サイズ6（24個の情報）である一連の4つのランダムバイナリシーケンスがコピータスク上に10個のメモリ位置のみを有する2層フィードフォワードモデルを訓練した。入力として提示される。トレーニングの詳細は [here](tasks/copy/) にあります。

The model was able to learn to copy the input successfully, and it indeed learned to use the mentioned memory mechanisms. The following figure (which resembles **Extended Data Figure 1** in the paper) illustrates that.

モデルは入力を正常にコピーすることを学ぶことができ、実際には言及されたメモリメカニズムを使用することを学んだ。次の図（これは ** Extended Data図1** に似ています ）はそのことを示しています。

*You can re-generate similar figures in the [visualization notebook](tasks/copy/visualization.ipynb)*

*あなたは [可視化ノート](tasks/copy/visualization.ipynb) で同様の数字を再現することができます*

![DNC-Memory-Mechanisms](/assets/DNC-dynamic-mem.png)

- In the **Memory Locations** part of the figure, it's apparent that the model is able to read the memory locations in the same order they were written into.

- 図の **Memory Locations** の部分では、モデルが書き込まれたのと同じ順序でメモリの場所を読み取ることができることは明らかです。

- In the **Free Gate** and the **Allocation Gate** portions of the figure, it's shown that the free gates are fully activated after a memory location is read and becomes obsolete, while being less activated in the writing phase. The opposite is true for the allocation gate. The **Memory Locations Usage** also demonstrates how memory locations are used, freed, and re-used again time after time.

- **Free Gate** と **Allocation Gate**の部分では、メモリ・ロケーションが読み出された後に空きゲートが完全にアクティブになり、書き込みフェーズではアクティブ化されなくなったことが示されています。アロケーションゲートの場合とは逆です。 ** Memory Locations Usage **には、メモリの場所がどのように使用され、解放され、再度使用されるかが示されています。

*The figure differs a little from the one in the paper when it comes to the activation degrees of the gates. This could be due to the small size of the model and the relatively small training time. However, this doesn't affect the operation of the model.*

*数字は、ゲートの活性化度については紙面の数字と少し異なります。これは、モデルのサイズが小さく、トレーニング時間が比較的短いことが原因です。ただし、これはモデルの動作には影響しません。*

###一般化とメモリのスケーラビリティ


This experiment was designed to check:
- if the trained model has learned an implicit copying algorithm that can be generalized to larger input lengths.
- if the learned model is independent of the training memory size and can be scaled-up with memories of larger sizes.

この実験は以下を確認するように設計されました：
- 訓練されたモデルが、より大きな入力長に一般化できる暗黙のコピーアルゴリズムを学習した場合。
- 学習されたモデルがトレーニングメモリサイズとは無関係で、より大きなサイズのメモリでスケールアップできる場合。

To approach that, a 2-layer feedforward model with 15 memory locations was trained on a copy problem in which a single sequence of random binary vectors of lengths between 1 and 10 was presented as input. Details of the training process can be found [here](tasks/copy/).

これにアプローチするために、15のメモリ位置を有する2層フィードフォワードモデルが、1から10の長さのランダムバイナリベクトルの単一シーケンスが入力として提示されるコピー問題に対して訓練された。トレーニングプロセスの詳細は [ここ](tasks/copy/) にあります。

The model was then tested on pairs of increasing sequence lengths and increasing memory sizes with re-training on any of these pairs of parameters, and the fraction of correctly copied sequences out of a batch of 100 was recorded. The model was indeed able to generalize and use the available memory locations effectively without retraining. This is depicted in the following figure which resembles **Extended Data Figure 2** from the paper.

次いで、これらのパラメータ対のいずれかを再トレーニングして、増加する配列長さおよび増加するメモリサイズの対についてモデルを試験し、バッチ100のうち正しくコピーされた配列の割合を記録した。このモデルは、実際に再学習することなく、利用可能なメモリ位置を効果的に一般化して使用することができました。これは、次の図に示されています。これは**論文の** Extended Data図2 **に似ています。

*Similar figures can be re-generated in the [visualization notebook](tasks/copy/visualization.ipynb)*

* [ビジュアル化ノート](tasks/copy/visualization.ipynb) で同様の数値を再生成することができます。*

![DNC-Scalability](/assets/DNC-scalable.png)

##関与する

If you're interested in using the implementation for new tasks, you should first start by **[reading the structure and basic usage guide](docs/basic-usage.md)** to get comfortable with how the project is structured and how it can be extended to new tasks.

新しいタスクにインプリメンテーションを使用することに興味がある場合は、最初に **[構造と基本的な使い方のガイドを読む](docs/basic-usage.md)** から始めて、プロジェクトがどのように構造化されているかに気を配り、どのように新しいタスクに拡張することができます。

If you intend to work with the source code of the implementation itself, you should begin with looking at **[the data flow diagrams](docs/data-flow.md)** to get a high-level overview of how the data moves from the input to the output across the modules of the implementation. This would ease you into reading the source code, which is okay-documented.

実装自体のソースコードを扱うつもりならば、 **[データフロー図](docs/data-flow.md)** を見て、データの仕組みの概要を把握する必要があります実装のモジュール間で入力から出力に移動します。これにより、ソースコードを読みやすくなります。これは問題ありません。

You might also find the **[implementation notes](docs/implementation-notes.md)** helpful to clarify how some of the math is implemented.

また、** [実装上の注意点](docs/implementation-notes.md)**が数学の実装方法を明確にするのに役立つかもしれません。

## To-Do

- **コア：**
     - スパースリンクマトリックス。
     - 同じバッチ全体での可変シーケンス長。
- **タスク**：
     - bAbIタスク。
     - グラフ推論タスク。
     - ミニSHRDLUタスク。
- **ユーティリティ**：
     - 反復、学習率などに関するすべての詳細を設定可能なコマンドライン引数に抽象化し、計算グラフを定義する心配をユーザに残すタスクビルダ。

##著者
Mostafa Samir

[mostafa.3210@gmail.com]（mailto：mostfa.3210@gmail.com）

##ライセンス
MIT
