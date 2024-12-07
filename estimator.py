import torch

class SoilMoistureEstimator(torch.nn.Module):

    SMOOTH_KERNEL_SIZE = (4, 4)
    AVERAGE_KERNEL_SIZE = (64, 64)

    def get_estimator(path='estimator_weights.pkl') -> torch.nn.Module:
        model = SoilMoistureEstimator()
        model.load_state_dict(torch.load(path))
        model.eval()
        return model

    def __init__(self):
        super().__init__()

        self.smoothing = torch.nn.AvgPool2d(SoilMoistureEstimator.SMOOTH_KERNEL_SIZE)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Conv2d(3, 10, 1),
            torch.nn.Tanh(),
            torch.nn.Conv2d(10, 10, 1),
            torch.nn.Tanh(),
            torch.nn.Conv2d(10, 1, 1),
            torch.nn.Sigmoid()
        )
        self.average = torch.nn.AvgPool2d(SoilMoistureEstimator.AVERAGE_KERNEL_SIZE)
        self.initialize_weights()
        
    def forward(self, x0):
        x1 = self.smoothing(x0)
        x2 = self.feedforward(x1)
        x3 = self.average(x2).squeeze()
        return x3

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
