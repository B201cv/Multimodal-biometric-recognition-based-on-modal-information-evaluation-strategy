from numpy import *
from torch.utils.data import Dataset
# from model import vgg
import torch
# from model2 import resnet18
from torch import nn
import os
import sys
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import cv2
import random
import numpy as np
from collections import Counter
from model3.model_vgg import Vgg16


def testmodel():
    class SKConv(nn.Module):
        """
        1.首先特征图X 经过3x3，5x5, 7x7, 等卷积得到U1,U2,U3三个特征图，然后相加得到了U，U中融合了多个感受野的信息。
          然后沿着H和W维度求平均值，最终得到了关于channel的信息是一个C×1×1的一维向量，结果表示各个通道的信息的重要程度。
        2.接着再用了一个线性变换，将原来的C维映射成Z维的信息，然后分别使用了三个线性变换，从Z维变为原来的C，这样完成了正对channel维度的信息提取。
          然后使用Softmax进行归一化，这时候每个channel对应一个分数，代表其channel的重要程度，这相当于一个mask。
        3.将这三个分别得到的mask分别乘以对应的U1,U2,U3，得到A1,A2,A3。
          然后三个模块相加，进行信息融合，得到最终模块A， 模块A相比于最初的X经过了信息的提炼，融合了多个感受野的信息。
        """
        def __init__(self, features, WH, M, G, r, stride=1, L=32):
            super(SKConv, self).__init__()
            d = max(int(features / r), L) # 取两个中最大的个值
            self.M = M # 有多少路径
            self.features = features
            self.convs = nn.ModuleList([])
            for i in range(M):
                self.convs.append(nn.Sequential(
                    nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=False)
                ))
            # self.gap = nn.AvgPool2d(int(WH/stride))
            self.fc = nn.Linear(features, d)
            self.fcs = nn.ModuleList([])
            for i in range(M):
                self.fcs.append(
                    nn.Linear(d, features)
                )
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            for i, conv in enumerate(self.convs):
                # (0): Conv2d、(1): Conv2d、(2): Conv2d....(n-1)
                # (b, 1, h, w) -->(b, 1, 1, h, w)
                fea = conv(x).unsqueeze_(dim=1)
                if i == 0:
                    # (b, 1, 1, h, w)
                    feas = fea
                else:
                    # (b, 2, 1, h, w)、(b, 3, 1, h, w)
                    feas = torch.cat([feas, fea], dim=1)
            fea_U = torch.sum(feas, dim=1)
            # fea_s = self.gap(fea_U).squeeze_()
            fea_s = fea_U.mean(-1).mean(-1)
            fea_z = self.fc(fea_s)
            for i, fc in enumerate(self.fcs):
                vector = fc(fea_z).unsqueeze_(dim=1)
                if i == 0:
                    attention_vectors = vector
                else:
                    attention_vectors = torch.cat([attention_vectors, vector], dim=1)
            attention_vectors = self.softmax(attention_vectors)
            attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
            fea_v = (feas * attention_vectors).sum(dim=1)
            return fea_v

    # CBMA  通道注意力机制和空间注意力机制的结合
    class ChannelAttention(nn.Module):
        def __init__(self, in_planes, ratio=16):
            super(ChannelAttention, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化高宽为1
            self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化高宽为1

            # 利用1x1卷积代替全连接
            self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # 平均池化---》1*1卷积层降维----》激活函数----》卷积层升维
            avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
            # 最大池化---》1*1卷积层降维----》激活函数----》卷积层升维
            max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
            out = avg_out + max_out  # 加和操作
            return self.sigmoid(out)  # sigmoid激活操作


    class SpatialAttention(nn.Module):
        def __init__(self, kernel_size=7):
            super(SpatialAttention, self).__init__()

            assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
            padding = kernel_size // 2
            # 经过一个卷积层，输入维度是2，输出维度是1
            self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
            self.sigmoid = nn.Sigmoid()  # sigmoid激活操作

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)  # 在通道的维度上，取所有特征点的平均值  b,1,h,w
            max_out, _ = torch.max(x, dim=1, keepdim=True)  # 在通道的维度上，取所有特征点的最大值  b,1,h,w
            x = torch.cat([avg_out, max_out], dim=1)  # 在第一维度上拼接，变为 b,2,h,w
            x = self.conv1(x)  # 转换为维度，变为 b,1,h,w
            return self.sigmoid(x)  # sigmoid激活操作


    class cbamblock(nn.Module):
        def __init__(self, channel, ratio=16, kernel_size=7):
            super(cbamblock, self).__init__()
            self.channelattention = ChannelAttention(channel, ratio=ratio)
            self.spatialattention = SpatialAttention(kernel_size=kernel_size)

        def forward(self, x):
            x = x * self.channelattention(x)  # 将这个权值乘上原输入特征层
            x = x * self.spatialattention(x)  # 将这个权值乘上原输入特征层
            return x

    class ca(nn.Module):
        def __init__(self, num_class=290):
            super(ca, self).__init__()
            self.backbone1 = Vgg16()
            self.backbone2 = Vgg16()
            palmpth = r'vgg16.pth'
            veinpth = r'vgg16.pth'
            palm_state = torch.load(palmpth)
            for k in list(palm_state.keys()):
                if 'classifer' in k:
                    palm_state.pop(k)
                # elif 'fc' in k:
                #     palm_state.pop(k)

            self.backbone1.load_state_dict(palm_state, strict=False)
            vein_state = torch.load(veinpth)
            for k in list(vein_state.keys()):
                if 'classifer' in k:
                    vein_state.pop(k)
                # elif 'fc' in k:
                #     vein_state.pop(k)
            self.backbone2.load_state_dict(vein_state, strict=False)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.CBAM = cbamblock(channel=512, ratio=16, kernel_size=7)

            self.f = nn.Conv2d(1024, 640, kernel_size=1)
            # self.f1 = nn.Conv2d(512, 400, kernel_size=1)
            self.sk = SKConv(features=640, WH=1, M=3, G=32, r=16)
            self.tcp1 = nn.Linear(512, 290)
            self.tcp2 = nn.Linear(512, 290)
            self.fc = nn.Linear(640, num_class)
            self.d = nn.Dropout(0.5)
            self.d1 = nn.Dropout(0.4)

        def forward(self, x1, x2):
            x1 = self.backbone1(x1)
            x2 = self.backbone2(x2)
            B, C, H, W = x1.shape
            x1 = self.pool(x1)
            x2 = self.pool(x2)
            x1 = self.CBAM(x1)
            x2 = self.CBAM(x2)

            x1a = x1.reshape(B, -1)
            x2a = x2.reshape(B, -1)
            tcp_trust_m1 = self.tcp1(x1a)
            # tcp_trust_m1 = self.bn(tcp_trust_m1)
            tcp_trust_m1 = self.d(tcp_trust_m1)
            # tcp_trust_m1 = self.relu(tcp_trust_m1)
            tcp_trust_m2 = self.tcp2(x2a)
            # tcp_trust_m2 = self.bn(tcp_trust_m2)
            tcp_trust_m2 = self.d(tcp_trust_m2)

            # w1 = torch.nn.functional.softmax(tcp_trust_m1, dim=1)
            # w2 = torch.nn.functional.softmax(tcp_trust_m2, dim=1)
            w1 = torch.sigmoid(tcp_trust_m1)
            w2 = torch.sigmoid(tcp_trust_m2)
            w1 = torch.max(w1)
            w2 = torch.max(w2)
            # w11 = torch.exp(self.w1) / torch.sum(torch.exp(self.w1+self.w2))
            w11 = w1
            w21 = w2
            x1 = x1 * w11
            x2 = x2 * w21
            x = torch.cat([x1, x2], dim=1)
            # x = x1+x2
            x = self.f(x)
            x = self.sk(x)
            x = x.reshape(B, -1)
            x = self.fc(x)

            return x, tcp_trust_m1, tcp_trust_m2

    def trans_form(img):

        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])

        img = transform(img)
        # img = img.unsqueeze(0)
        return img



    a = []
    class TestData(Dataset):
        def __init__(self, print_root_dir, vein_root_dir,training=True):
            self.print_root_dir = print_root_dir
            self.vein_root_dir = vein_root_dir
            self.person_path = os.listdir(self.print_root_dir)
        def __getitem__(self, idx):
            person_name = self.person_path[idx //5]
            a.append(person_name)
            bb=Counter(a)
            b=bb[person_name]-1
            print_imgs_path = os.listdir(os.path.join(self.print_root_dir, person_name))
            vein_imgs_path = os.listdir(os.path.join(self.vein_root_dir, person_name))
            length1_imgs = len(print_imgs_path)
            if len(a) == len(print_imgs_path):
                a.clear()
            print_img_path = print_imgs_path[b]
            vein_img_path = vein_imgs_path[b]
            p_img_item_path = os.path.join(self.print_root_dir, person_name, print_img_path)
            v_img_item_path = os.path.join(self.vein_root_dir, person_name, vein_img_path)
            p_img = cv2.imread(p_img_item_path)
            p_img = torch.tensor(p_img / 255.0).to(torch.float).permute(2, 0, 1)
            p_img = trans_form(p_img)
            v_img = cv2.imread(v_img_item_path)
            v_img = torch.tensor(v_img / 255.0).to(torch.float).permute(2, 0, 1)
            v_img = trans_form(v_img)
            return p_img, v_img, person_name
        def __len__(self):
            return len(self.person_path)*5



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_test = TestData('../our/c-Palm-print-test/', '../our/palm-vein-test/')
    batch_size = 4
    loader = torch.utils.data.DataLoader(
        data_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0)
    print("data_loader = ", loader)
    print("start test......")
    model = ca()

    weights_path = "cumt-3.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("using {} device.".format(device))

    model.eval()
    accurate = 0
    arr = []
    for epoch in range(10):
        acc = 0.0  # accumulate accurate number / epoch
        num=0
        running_loss = 0.0
        with torch.no_grad():
            bar = tqdm(loader, file=sys.stdout)
            for data_test in bar:
                p_imgs, v_imgs, person_name = data_test
                p_imgs = p_imgs.to(device)
                v_imgs = v_imgs.to(device)
                outputs, outputs1, outputs2 = model(p_imgs, v_imgs)
                predict_y = torch.max(outputs, dim=1)[1]
                person_labels = [int(_) - 1 for _ in person_name]
                person_labels = torch.tensor(person_labels)
                acc += torch.eq(predict_y, person_labels.to(device)).sum().item()
                num = 1450
        accurate = acc / num
        arr.append(accurate)
        print('[epoch %d] ' % (epoch + 1))
        print('  num:{},test_accuracy:{:.3f},acc:{}'.format(num, accurate, acc))
    # accurate += accurate
    # ave = accurate / 40
    ave = mean(arr)
    std = np.std(arr)
    print(ave)
    print(std)

if __name__ == "__main__":
    testmodel()
