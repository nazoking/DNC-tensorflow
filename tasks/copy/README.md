###共通設定
<!--
### Common Settings
-->

Both series and single models were trained on 2-layer feedforward controller (with hidden sizes 128 and 256 respectively) with ReLU activations, and both share the following set of hyperparameters:
シリーズおよび単一モデルは両方とも、ReLUアクチベーションを備えた2層フィードフォワードコントローラ（それぞれ隠されたサイズ128および256）で訓練され、両方とも以下のハイパーパラメータのセットを共有する。

- RMSProp Optimizer with learning rate of 10⁻⁴, momentum of 0.9.
- Memory word size of 10, with a single read head.
- A batch size of 1.
- 学習率10-4、運動量0.9のRMSPropオプティマイザ。
- メモリワードサイズは10、読み取りヘッドは1つ。
- バッチサイズは1です。

All output from the DNC is squashed between 0 and 1 using a sigmoid functions and  binary cross-entropy loss (or logistic loss) function of the form:
DNCからのすべての出力は、以下の形式のシグモイド関数およびバイナリクロスエントロピー損失（またはロジスティック損失）関数を使用して、0と1との間で潰される。

![loss](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D%28y%2C%20%5Chat%7By%7D%29%20%3D%20-%5Cfrac%7B1%7D%7BBTS%7D%5Csum_%7Bi%3D1%7D%5E%7BB%7D%5Csum_%7Bj%3D1%7D%5E%7BT%7D%5Csum_%7Bk%3D1%7D%5ES%5Cleft%28%20y_%7Bijk%7D%5Clog%20%5Chat%7By%7D_%7Bijk%7D%20&plus;%20%281%20-%20y_%7Bijk%7D%29%5Clog%281-%5Chat%7By%7D_%7Bijk%7D%29%20%5Cright%29)

is used. That is the mean of the logistic loss across the batch, time steps, and output size.
使用されている。 これは、バッチ、時間ステップ、および出力サイズにわたるロジスティック損失の平均です。

All gradients are clipped between -10 and 10.
すべての勾配は-10と10の間でクリップされます。

*Possible __NaNs__ could occur during training!*
*可能な__NaNs__はトレーニング中に発生する可能性があります！*


<!--
### Series Training
-->
###シリーズトレーニング

The model was first trained on a length-2 series of random binary vectors of size 6. Then starting off from the length-2 learned model, a length-4 model was trained in a **curriculum learning** fashion.
このモデルはまず、サイズ6の長さ2系列のランダムバイナリベクトルで訓練されました。次に、長さ2の学習モデルから始めて、長さ4のモデルが**カリキュラム学習**の方法で訓練されました。

The following plots show the learning curves for the length-2 and length-4 models respectively.
次のプロットは、それぞれ長さ2および長さ4のモデルの学習曲線を示しています。

![series-2](/assets/model-series-2-curve.png)

![series-4](/assets/model-series-4-curve.png)

*Attempting to train a length-4 model directly always resulted in __NaNs__. The paper mentioned using curriculum learning for the graph and mini-SHRDLU tasks, but it did not mention any thing about the copy task, so there's a possibility that this is not the most efficient method.*
*長さ4のモデルを直接訓練しようとすると、常に__NaNs__が返されます。 論文では、グラフとミニSHRDLUのタスクにカリキュラムの学習を使って言及しましたが、コピーのタスクについて何も言及していないので、これが最も効率的な方法ではない可能性があります。

#### Retraining
```
$python tasks/copy/train-series.py --length=2
```
Then, assuming that the trained model from that execution is saved under the name 'step-100000'.
次に、その実行から訓練されたモデルが「step-100000」という名前で保存されていると仮定します。

```
$python tasks/copy/train-series.py --length=4 --checkpoint=step-100000 --iterations=20000
```

### Single Training

The model was trained directly on a single input of length between 1 and 10 and the length was chosen randomly at each run, so no curriculum learning was used. The following plot shows the learning curve of the single model.
このモデルは1から10の間の長さの単一の入力に直接訓練され、長さは各走行でランダムに選択されたので、カリキュラム学習は使用されなかった。 次のプロットは、単一モデルの学習曲線を示しています。

![single-10](/assets/model-single-curve.png)

#### Retraining

```
$python tasks/copy/train.py --iterations=50000
```
