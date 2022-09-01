from src.config import n_train, P, T, epochs, lstm_layers
from src.metaqaoa import gene_random_instance, gene_qaoa_layers, MetaQAOA, MetaQAOALoss
from src.utils import ForwardWithLoss, TrainOneStep
import numpy as np
from mindspore import Tensor, ops, nn, Parameter
import mindspore as ms
import os
os.environ["OMP_NUM_THREADS"] = "1"
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
# ms.set_seed(10000)

if __name__ == "__main__":
    # 生成训练数据，n_train in [6, 12], 说是生成500个训练数据， 其中250个测试
    # 生成测试数据，n_test in [8, 14], 100个测试数据, n_train < n_test
    g = gene_random_instance(n_train)
    
    # 相当于是输入模型的数据
    qnn = gene_qaoa_layers(g, P)

    metaqaoa = MetaQAOA(T=T, QNN=qnn, lstm_layers=lstm_layers)
    loss_fn = MetaQAOALoss()
    optimizer = nn.Adam(metaqaoa.trainable_params(), learning_rate=0.001)


    forward_with_loss = ForwardWithLoss(metaqaoa, loss_fn)
    train_one_step = TrainOneStep(forward_with_loss, optimizer)

   
    for epoch in range(epochs):
        # batch_size, seq_len, input_size
        # num_directions * num_layers, batch_size, hidden_size
        theta = Tensor(np.ones([1, P, 2]).astype(np.float32))
        h = Tensor(np.ones([lstm_layers, 1, 2]).astype(np.float32))
        c = Tensor(np.ones([lstm_layers, 1, 2]).astype(np.float32))
        loss = train_one_step(theta, h, c)

        loss = forward_with_loss(theta, h, c)
        # theta, h, c, _ = metaqaoa(theta, h, c)
        max_cut_loss = qnn(theta.reshape(1, -1))
        print("epoch: ", epoch,"metaqaoa loss: ", loss, "max cut loss: ", max_cut_loss.squeeze())
        