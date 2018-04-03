'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import time
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision
import numpy as np
import sys
# Hyper Parameters 
#input_size = 784
#hidden_size = 500
classes = 10
#num_epochs = 164
#batch_size = 128
#learning_rate = 0.1



parser = argparse.ArgumentParser(description = 'Residual Network Description' )
parser.add_argument('--nepoch', type=int,default = 164 ,help='number of training epoch, default=164')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size, default=64')
parser.add_argument('--model', type=int, default=1, help='input which resNet structure:1=ResNet20, 2=ResNet56, 3=ResNet110, default=1')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate, default=0.1')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
args = parser.parse_args()

print('Model architecture:\t', args.model)
print(args)


cuda_Enable = torch.cuda.is_available()
if(cuda_Enable and args.cuda):
    use_gpu = True
else:
    use_gpu = False
    
transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2471, 0.2435, 0.2616))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize,
                                         shuffle=False, num_workers=2)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=3, stride=stride,padding=1, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


#class Bottleneck(nn.Module):
#    expansion = 4
#
#    def __init__(self, in_planes, planes, stride=1):
#        super(Bottleneck, self).__init__()
#        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#        self.bn1 = nn.BatchNorm2d(planes)
#        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#        self.bn2 = nn.BatchNorm2d(planes)
#        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
#        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
#
#        self.shortcut = nn.Sequential()
#        if stride != 1 or in_planes != self.expansion*planes:
#            self.shortcut = nn.Sequential(
#                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                nn.BatchNorm2d(self.expansion*planes)
#            )
#
#    def forward(self, x):
#        out = F.relu(self.bn1(self.conv1(x)))
#        out = F.relu(self.bn2(self.conv2(out)))
#        out = self.bn3(self.conv3(out))
#        out += self.shortcut(x)
#        out = F.relu(out)
#        return out
#

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=classes):
        #print(block)
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))#conv1 finish
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.layer4(out)
        out = F.avg_pool2d(out, (8,8))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    
def train_model(model):
    
    model.apply(weights_init)
    
    if(use_gpu):  
        model.cuda() 
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=0.9,weight_decay = 0.0001)
    
    #total=0
    #correct =0
    print(' = = = = = = = = = training begin = = = = = = = = = ')
    for epoch in range(args.nepoch):
        if epoch == 80 or epoch == 121:
            for param in optimizer.param_groups:
                
                param['lr'] = args.lr/10.0
                
                
            args.lr = args.lr/10.0
            #optimizer.param_groups['lr'] = learning_rate/10
        #total=0
        #correct =0
        for i, (images, labels) in enumerate(trainloader): 
            
            # Convert torch tensor to Variable
            if use_gpu: 
                images = images.cuda()
                labels = labels.cuda()
            images, labels = Variable(images), Variable(labels)
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            
            #b= labels
            #if(epoch+1)% 15 ==0:
            #    softmax_res = softmax(outputs.data.cpu().numpy()[0])
             #   _, predicted = torch.max(outputs.data, 1)
                
        #print(predicted)
        #print(targets.size(0))
              #  total += b.size(0)
               # correct += predicted.eq(b.data).cpu().sum()
                #acc = 100.*correct/total
                #print("| Test Result\tAcc@1 %.2f%%" %(acc))
            if (i+1) % 100 == 0:
                 #save_checkpoint({
            #'epoch': epoch + 1,
            #'state_dict': net.state_dict(),
            #'optimizer' : optimizer.state_dict()
            #})
                print ('Epoch [%d/%d], Iteration [%d/%d], Loss: %.5f ' 
                       %(epoch+1, args.nepoch, i+1, len(trainset)//args.batchSize, loss.data[0]))
            
    print(' = = = = = = = = = training finish = = = = = = = = = ')
    
    print('= = = = = = = = = begin store model= = = = = = = = = ')
    torch.save(model, './first_train.pt')
    print('= = = = = = = = = finish store= = = = = = = = = ')
    
    
    
def test_model(model):
    print('= = = = = = = = = testing begin= = = = = = = = = ')
    total = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):#dset_loaders['val']):
        if use_gpu: 
            inputs = inputs.cuda()
            targets = targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        
        outputs = model(inputs)
        #print("///////////",outputs)
        #print("++++",outputs.data.cpu().numpy()[0])
        #softmax_res = softmax(outputs.data.cpu().numpy())
        #print("----",outputs.data)
        #print("????",type(softmax_res))
        _, predicted = torch.max(outputs.data, 1)
        #print(predicted)
        #print("...",targets.size(0))
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    acc = 100.*correct/total
    print("| Test Result\tAcc@ %.2f%%" %(acc))

    print('= = = = = = = = = testing finish = = = = = = = = = ')

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
#def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#    torch.save(state, filename)
#    if is_best:
#        shutil.copyfile(filename, 'model_best.pth.tar')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
      
        torch.nn.init.kaiming_uniform(m.weight.data)
        #m.bias.data.fill_(0)
        #torch.nn.init.constant_(m.bias.data,0)
        
    elif classname.find('BatchNorm') != -1:
        #m.weight.data.normal_(0, 1)
        m.bias.data.fill_(0)
        torch.nn.init.uniform(m.weight.data,-1, 1)
        #torch.nn.init.uniform(m.bias.data,-1, 1)
        #torch.nn.init.constant_(m.bias.data,0)
       
    elif classname.find('Linear') != -1:
        torch.nn.init.kaiming_uniform(m.weight.data)
        #torch.nn.init.kaiming_uniform(m.weight.data)
        m.bias.data.fill_(0) 
        #torch.nn.init.constant_(m.bias.data,0)
       
        



    #y = net(Variable(torch.randn(1,3,32,32)))
    #print(y.size())
#def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
 #   torch.save(state, filename)
#    if is_best:
#        shutil.copyfile(filename, 'model_best.pth.tar')
def main():
    tStart = time.time()
    
    if(args.model == 1):
        net = ResNet(BasicBlock, [3,3,3])
    elif args.model ==2:
        net = ResNet(BasicBlock, [9,9,9])
    elif args.model ==3:
        net = ResNet(BasicBlock, [18,18,18])
    else:
         print('Error : Please input 1~3')
         sys.exit(1)
    train_model(net)
    test_model(net) 
    
    tEnd = time.time()#計時結束
    print ("It cost %.2f sec" % (tEnd - tStart))#會自動做近位
    


if __name__ == '__main__':
    main()    

