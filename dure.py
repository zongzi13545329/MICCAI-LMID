import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ConcatLayer(nn.Module):
    def __init__(self, in_size, out_size):
        super(ConcatLayer, self).__init__()
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Identity()  

        self.transition = nn.Sequential(
            nn.Linear(512, in_size),  
            nn.ReLU(),
            nn.BatchNorm1d(in_size)
        )
        self.fc = nn.Linear(in_size, out_size)

    def extract_features(self, x):
        features = self.feature_extractor(x)  # [N, 512]
        features = self.transition(features)  # [N, in_size]
        return features

    def forward(self, x):
        if x.dim() == 4 and x.size(1) == 3:  
            features = self.extract_features(x)
        else:
            raise ValueError("the size is wrong")

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
        
        self.fcc = nn.Conv1d(input_size, output_class, kernel_size=1)

    def forward(self, feats, c):
        device = feats.device
        batch_size = feats.shape[0]
        
        V = self.v(feats)  # [N, input_size]
        Q = self.q(feats)  # [N, input_size]
        
        B = torch.mm(Q.transpose(0, 1), V)  # [input_size, input_size]
        
        B = B.unsqueeze(0)  # [1, input_size, input_size]
        
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
