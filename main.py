from ast import For
from src.config import n_train, P, T, epochs
from src.metaqaoa import gene_random_instance, gene_qaoa_layers, MetaQAOA, MetaQAOALoss
from src.utils import TrainOneStep, ForwardWithLoss
import numpy as np
from mindspore import Tensor, ops, nn, Parameter
import mindspore as ms
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")


if __name__ == "__main__":
    # 生成训练数据，n_train in [6, 12], 说是生成500个训练数据， 其中250个测试
    # 生成测试数据，n_test in [8, 14], 100个测试数据, n_train < n_test
    g = gene_random_instance(n_train)
    
    # 相当于是输入模型的数据
    qnn = gene_qaoa_layers(g, P)

    metaqaoa = MetaQAOA(T)
    loss_fn = MetaQAOALoss()
    optimizer = nn.Adam(metaqaoa.trainable_params(), learning_rate=0.001)

    # def forward_with_loss(theta, h, y, QNN):
    #     _, y_list = metaqaoa(theta, h, y, QNN)
    #     loss = loss_fn(y_list)
    #     return loss
    
    # grad_fn = ops.value_and_grad(forward_with_loss, None, optimizer.parameters)

    # def train_step(theta, h, y, QNN):
    #     loss, grads = grad_fn(theta, h, y, QNN)
    #     optimizer(grads)
    #     return loss

    forward_with_loss = ForwardWithLoss(metaqaoa, loss_fn)
    train_one_step = TrainOneStep(forward_with_loss, optimizer)

    for epoch in range(epochs):
        # batch_size, seq_len, input_size
        theta = Parameter(np.ones([1, 2 * P, 1]).astype(np.float32))
        # num_directions * num_layers, batch_size, hidden_size
        h = Tensor(np.zeros([1, 1, 1]).astype(np.float32))
        y = Tensor(np.zeros([1, 1, 1]).astype(np.float32))
        # theta_list, y_list = metaqaoa(theta, h, y, qnn)
        # loss = loss_fn(y_list)
        loss = forward_with_loss(theta, h, y, qnn)
        # loss = train_one_step(theta, h, y, qnn)
        print(loss)

        