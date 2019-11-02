from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')                   #AGGを指定→savefig(最後)の時PNGファイルを出力する
import matplotlib.pyplot as plt
class Sequence(nn.Module):              #nn.Moduleクラスを継承
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)    #LSTMCellに入れる最初の値を定義(n*51の0配列)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)): #chunkで97*999のテンソルを97*1のテンソルのタプルにして                                                                   いる。forループは999回行われる
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))                  #h_t,c_t,input_t(正解)をLSTMCellに入れて出したh_t,c_tを                                                              また次のh_t,c_tに代入して使う
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)                                  #linear(=affine)でxw+bを行いh_t2を97*51→97*1にする。wと                                                              bも学習対象
            outputs += [output]                                         #outputをoutputsに追加していく(リスト型??)
        for i in range(future):# if we should predict the future 
        #このforループでは正解がわからないのでh_t,c_t,正解ではなくh_t, c_t,自分の予測(output)の3つをLSTMCellに入れている
            h_t, c_t = self.lstm1(output, (h_t, c_t))                   #1個前のforループで最後に出たh_t,c_t値を使って続きから
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))                  #                       〃
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)                    #stackで(97*1)*999→97*999*1, squeezeで97*999にする
        return outputs


if __name__ == '__main__':
    # set random seed to 0
    np.random.seed(0)                                   #推測開始時の初期値などでどこかの関数で使われるシードを固定して、ランダ                                                      ムではあるけど毎回同じ学習内容になるようにする
    torch.manual_seed(0)                                #numpyとtorchで両方使われるので両方のランダムシードを初期化する
    # load data and make training set
    data = torch.load('traindata.pt')
    input = torch.from_numpy(data[3:, :-1])             #numpyのスライス dataの3行目から最後まで、最後を除く列
    target = torch.from_numpy(data[3:, 1:])             #       〃       dataの3行目から最後まで、最初を除く列
    test_input = torch.from_numpy(data[:3, :-1])        #       〃       dataの1行目から3行目まで、最後を除く列
    test_target = torch.from_numpy(data[:3, 1:])        #       〃       dataの1行目から3行目まで、最初を除く列
    # build the model
    seq = Sequence()                                    #Sequenceクラスを初期化してseqに代入(init関数が実行される)
    seq.double()                                        #型の変換(float32から精度が倍のdouble型にしてfloat64との演算可能にする)
    criterion = nn.MSELoss()                            #平均二乗誤差（L2）
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)   #勾配降下法（学習率0.8）
    #begin to train
    for i in range(15):
        print('STEP: ', i)
        def closure():                                  #for分の中で関数を定義??→よくわからない #closureは再計算、RNNで使われる
            optimizer.zero_grad()                       #勾配の初期化(全部の変数が持ってるgradというパラメータがあり、それをす                                               べて0にする)
            out = seq(input)                            #inputを引数にSequenceクラスのforward関数を実行,結果をoutに代入
            loss = criterion(out, target)               #out(=outputs)とtarget(=目的、正解)の二乗誤差
            print('loss:', loss.item())                 #pytorchのスカラーとみなせるテンソルをpythonのスカラー(余計な機能を持た                                             ないただの数字＝メモリを節約できる)に変換して、print
            loss.backward()                             #勾配の計算(まずlossを微分してからlossに携わるすべての変数について微分                                               してそれぞれの変数が持ってるgradに(今入っている値を消して)代入する＝誤                                              差逆伝播法)
            return loss                                 #lossをreturn(lossは使われない 形式的)
        optimizer.step(closure)                         #パラメータの更新 #closure関数が動作している
        # begin to predict, no need to track gradient here
        with torch.no_grad():                           #with no_gradでwith内の記述を勾配計算しない(メモリ節約・計算コスト削減)                                                  testは精度を図るためで学習対象じゃないから微分する必要がない
            future = 1000                               #RNNの回転数
            pred = seq(test_input, future=future)       #test_inputを引数にSequenceクラスのforward関数を実行,結果をpredに代入
            loss = criterion(pred[:, :-future], test_target)#predとtest_targetの二乗誤差
            print('test loss:', loss.item())            #testの二乗誤差=精度をprint
            y = pred.detach().numpy()                   #pred以降の変数を微分対象から外す(メモリ節約)
        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
        draw(y[0], 'r')                                 #学習結果を3つ見てみる(1)
        draw(y[1], 'g')                                 #学習結果を3つ見てみる(2)
        draw(y[2], 'b')                                 #学習結果を3つ見てみる(3)
        plt.savefig('predict%d.png'%i)                  #PNGファイルをPDFにして出力(%dにiを代入してファイル名にする)
        plt.close()                                     