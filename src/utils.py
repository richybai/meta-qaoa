from mindspore import nn, ops



class ForwardWithLoss(nn.Cell):
    """定义损失网络"""

    def __init__(self, backbone, loss_fn):
        """实例化时传入前向网络和损失函数作为参数"""
        super(ForwardWithLoss, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, theta, h, y, QNN):
        """连接前向网络和损失函数"""
        _, y_list = self.backbone(theta, h, y, QNN)
        return self.loss_fn(y_list)

    def backbone_network(self):
        """要封装的骨干网络"""
        return self.backbone


class TrainOneStep(nn.TrainOneStepCell):
    """定义训练流程"""

    def __init__(self, network, optimizer):
        """参数初始化"""
        super(TrainOneStep, self).__init__(network, optimizer)
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, theta, h, y, QNN):
        """构建训练过程"""
        weights = self.weights
        loss = self.network(theta, h, y, QNN)
        grads = self.grad(self.network, weights)(theta, h, y, QNN)
        return loss, self.optimizer(grads)