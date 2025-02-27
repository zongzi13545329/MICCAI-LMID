import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ConcatLayer(nn.Module):
    def __init__(self, in_size, out_size):
        super(ConcatLayer, self).__init__()
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Identity()  # 移除最终全连接层

        # 添加过渡层，将ResNet特征转换为所需的输入维度
        self.transition = nn.Sequential(
            nn.Linear(512, in_size),  # ResNet18 的特征维度是 512
            nn.ReLU(),
            nn.BatchNorm1d(in_size)
        )

        # 输出层
        self.fc = nn.Linear(in_size, out_size)

    def extract_features(self, x):
        """
        提取图像特征，将输入的图像数据转换为特征向量。
        输入: x - [N, 3, 224, 224]
        输出: features - [N, in_size]
        """
        features = self.feature_extractor(x)  # [N, 512]
        features = self.transition(features)  # [N, in_size]
        return features

    def forward(self, x):
        """
        前向传播。
        输入: x - [N, in_size] 或 [N, 3, 224, 224]
        输出: features - [N, in_size], out - [N, out_size]
        """
        if x.dim() == 4 and x.size(1) == 3:  # 输入为图像数据
            features = self.extract_features(x)
        else:
            raise ValueError("输入的形状不符合要求。")

        out = self.fc(features)  # [N, out_size]
        return features, out


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats):
        x = self.fc(feats)
        return feats, x

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        
    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1), c

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.2, nonlinear=True, passing_v=True):
        super(BClassifier, self).__init__()
        
        self.q = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size ),
            nn.ReLU(),
            nn.Linear(input_size , input_size)
        )
        
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU(),
                nn.Linear(input_size, input_size)
            )
        else:
            self.v = nn.Identity()
        
        # 修改卷积层，使其适应输入维度
        self.fcc = nn.Conv1d(input_size, output_class, kernel_size=1)

    def forward(self, feats, c):
        device = feats.device
        batch_size = feats.shape[0]
        
        # 应用V和Q变换
        V = self.v(feats)  # [N, input_size]
        Q = self.q(feats)  # [N, input_size]
        
        # 计算加权和
        B = torch.mm(Q.transpose(0, 1), V)  # [input_size, input_size]
        
        # 准备卷积输入
        B = B.unsqueeze(0)  # [1, input_size, input_size]
        
        # 应用卷积得到最终输出
        C = self.fcc(B)  # [1, output_class, input_size]
        C = C.mean(dim=2)  # [1, output_class]
        
        return C, Q, B

class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        
    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)
        
        return classes, prediction_bag, A, B
