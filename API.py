import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision, sys, math, copy
import os


class Models:

    def __init__(self, model, num_class):
        self._model_list = ['BASIC_CONV_NET', 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'ResNet']
        assert model in self._model_list, 'Model must be either ' + ' or '.join(self._model_list)
        self._model = model
        self._num_class = num_class
        self._cuda = torch.device('cuda:0')

    def net(self):
        if self._model == 'BASIC_CONV_NET':
            return self._BasicConvNet_(self._num_class).to(self._cuda)
        if self._model[:3] == 'VGG':
            return self._VGG_(int(self._model[3:]), self._num_class).to(self._cuda)
        if self._model == 'ResNet':
            raise NotImplementedError

    class _BasicConvNet_(nn.Module):

        def __init__(self, num_class):
            super().__init__()

            self.global_step = 0
            self._checkpoint = './weights/basic_conv_net.ckpt'

            self._feature_extractor = nn.Sequential(
                nn.Conv2d(1, 32, 5, 1, 2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 5, 1, 2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self._avg_pool = nn.AdaptiveAvgPool2d((7, 7))
            self._classifier = nn.Sequential(
                nn.Linear(7*7*64, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, num_class),
                nn.LogSoftmax(dim=1),
            )

        def forward(self, x):
            x = self._feature_extractor(x)
            x = self._avg_pool(x)
            x = x.view(-1, 7*7*64)
            x = self._classifier(x)
            return x

        def init_conv_layers(self):
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.xavier_normal_(m.weight)
                    torch.nn.init.constant_(m.bias, 0)

        def restore_pretrained_weights(self):
            self.restore_checkpoint()

        def restore_checkpoint(self):
            self.load_state_dict(torch.load(self.checkpoint))

        def save_checkpoint(self):
            torch.save(self.state_dict(), self.checkpoint)

        def num_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def train_independent(self, dataset, optim, eval_freq, epoch):
            collate_loss, correct_predictions, total_processed = torch.zeros(1), torch.zeros(1), torch.zeros(1)
            for e in range(1, epoch+1):
                for data, labels in dataset.train_images:
                    data, labels = data.cuda(), labels.cuda()
                    self.train()
                    optim.zero_grad()
                    output = self.forward(data)
                    loss = nn.NLLLoss()(output, labels)
                    loss.backward()
                    optim.step()
                    self.global_step += 1
                    self._collate_loss += loss
                    predictions = output.argmax(dim=1, keepdim=True)
                    correct_predictions += predictions.eq(labels.view_as(predictions)).sum()
                    total_processed += labels.size()[0]
                    sys.stdout.write('\rIter:%i ' % self.global_step)
                    sys.stdout.flush()
                    if self.global_step % eval_freq == 0:
                        print('--> Loss:%.4f, Accuracy(Train,Eval):%.2f,%.2f' % (
                            collate_loss.item() / eval_freq,
                            100 * (correct_predictions / total_processed).item(),
                            self.evaluate(dataset)
                        ))
                        collate_loss, correct_predictions, total_processed = torch.zeros(1), torch.zeros(
                            1), torch.zeros(1)
                print('\rEpoch', e, 'done.')
                print('Final eval accuracy:', self.evaluate(dataset))

        def evaluate(self, dataset):
            self.eval()
            correct_predictions, total = torch.zeros(1), torch.zeros(1)
            with torch.no_grad():
                for data, labels in dataset.eval_images:
                    data, labels = data.cuda(), labels.cuda()
                    output = self.forward(data)
                    predictions = output.argmax(dim=1, keepdim=True)
                    correct_predictions += predictions.eq(labels.view_as(predictions)).sum()
                    total += data.shape[0]
            return 100*(correct_predictions/total).item()

    class _VGG_(nn.Module):

        def __init__(self, num_layers, num_class):
            super().__init__()

            assert num_layers in [11, 13, 16, 19], 'VGG num_layers != [11, 13, 16, 19]'

            self.global_step = 0
            self._checkpoint = './weights/vgg'+str(num_layers)+'-'+str(num_class)+'.ckpt'

            self._base = [None]

            self._pruned_state = []

            self._optim = None

            if num_layers == 11:
                self._fe = torchvision.models.vgg11_bn().features
            elif num_layers == 13:
                self._fe = torchvision.models.vgg13_bn().features
            elif num_layers == 16:
                self._fe = torchvision.models.vgg16_bn().features
            elif num_layers == 19:
                self._fe = torchvision.models.vgg19_bn().features

            self._c = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(512, num_class),
            )

            self._conv_idx = []
            for i, m in enumerate(self._fe.modules()):
                if isinstance(m, nn.Conv2d):
                    self._conv_idx.append(i-1)

            self.cuda()

        def forward(self, x, inject_at_layer=None, extract_from_layer=None):
            if extract_from_layer is not None:
                x = self._fe[inject_at_layer:extract_from_layer+1](x)
                return x
            x = self._fe(x)
            x = x.view(x.size(0), -1)
            x = self._c(x)
            return x

        def init_conv_layers(self):
            print('Initializing conv layers')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    m.bias.data.zero_()

        def attach_optimizer(self, optim):
            self._optim = optim

        def change_optimizer_learning_rate(self, lr):
            self._optim.param_groups[0]['lr'] = lr

        def restore_checkpoint(self, location=None):
            if location is None:
                self.load_state_dict(torch.load(self._checkpoint))
            else:
                self.load_state_dict(torch.load(location))
            print('Restoring checkpoint from', self._checkpoint)

        def save_checkpoint(self, location=None):
            if location is None:
                torch.save(self.state_dict(), self._checkpoint)
            else:
                torch.save(self.state_dict(), location)
            print('Checkpoint saved at', self._checkpoint)

        def num_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def num_parameters_conv(self, layer=None):
            if layer == None:
                param = 0
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        param += sum(p.numel() for p in m.parameters())
                return param
            else:
                c_idx = str(self._conv_idx[layer])
                modules = list(self.children())[0]._modules
                conv = modules[c_idx]
                return sum(p.numel() for p in conv.parameters())

        def num_parameters_linear(self):
            return sum(p.numel() for p in self._c.parameters())

        def max_layers(self):
            return len(self._conv_idx)

        def max_filters(self, layer):
            return list(self.children())[0]._modules[str(self._conv_idx[layer])].out_channels

        def start_training(self, dataset, eval_freq, epoch):
            if self._optim == None: raise ConnectionError('Optimizer not connected. Use attach_optimizer to connect optimizer.')
            loss_metric = nn.CrossEntropyLoss()
            collate_loss, correct_predictions, total_processed = torch.zeros(1), torch.zeros(1), torch.zeros(1)
            for e in range(1, epoch+1):
                for data, labels in dataset.train_images:
                    data, labels = data.cuda(), labels.cuda()
                    self.train()
                    self._optim.zero_grad()
                    output = self.forward(data)
                    loss = loss_metric(output, labels)
                    loss.backward()
                    self._optim.step()
                    self.global_step += 1
                    collate_loss += loss
                    predictions = output.argmax(dim=1, keepdim=True)
                    correct_predictions += predictions.eq(labels.view_as(predictions)).sum()
                    total_processed += labels.size()[0]
                    sys.stdout.write('\rIter:%i' % self.global_step)
                    sys.stdout.flush()
                    if self.global_step % eval_freq == 0:
                        top1, top5 = self.evaluate(dataset)
                        print(') Loss:%.4f  Acc(Train Eval Top5): %.2f %.2f %.2f' % (
                            collate_loss.item()/eval_freq,
                            100*(correct_predictions/total_processed).item(),
                            top1, top5
                        ))
                        collate_loss, correct_predictions, total_processed = torch.zeros(1), torch.zeros(1), torch.zeros(1)
                print('\rEpoch', e, 'done.')
            print('Final eval accuracy (Top1,Top5):', self.evaluate(dataset))

        def retrain_layer(self, dataset, eval_freq, epoch, layer):
            if self._optim == None: raise ConnectionError('Optimizer not connected. Use attach_optimizer to connect optimizer.')
            if layer == len(self._conv_idx) - 1:
                raise IndexError('cannot retrain last layer. choose previous layer')

            from_layer = self._conv_idx[layer]
            to_layer = self._conv_idx[layer + 1]
            modules = list(self.children())[0]._modules

            loss_metric = nn.CrossEntropyLoss()
            collate_loss, correct_predictions, total_processed = torch.zeros(1), torch.zeros(1), torch.zeros(1)

            for i in range(from_layer):
                for p in modules[str(i)].parameters():
                    p.requires_grad = False
            for i in range(to_layer+1, len(modules)):
                for p in modules[str(i)].parameters():
                    p.requires_grad = False
            for p in self._c.parameters():
                p.requires_grad = False

            step = 0
            for e in range(1, epoch+1):
                for data, labels in dataset.train_images:
                    data, labels = data.cuda(), labels.cuda()
                    self.train()
                    self._optim.zero_grad()
                    output = self.forward(data)
                    loss = loss_metric(output, labels)
                    loss.backward()
                    self_optim.step()
                    step += 1
                    collate_loss += loss
                    predictions = output.argmax(dim=1, keepdim=True)
                    correct_predictions += predictions.eq(labels.view_as(predictions)).sum()
                    total_processed += labels.size()[0]
                    sys.stdout.write('\rIter:%i ' % step)
                    sys.stdout.flush()
                    if step % eval_freq == 0:
                        print('--> Loss:%.4f, Accuracy(Train,Eval):%.2f,%.2f' % (
                            collate_loss.item()/eval_freq,
                            100*(correct_predictions/total_processed).item(),
                            self.evaluate(dataset)
                        ))
                        collate_loss, correct_predictions, total_processed = torch.zeros(1), torch.zeros(1), torch.zeros(1)
                print('\rEpoch', e, 'done.')
            print('Final eval accuracy:', self.evaluate(dataset))

            for p in self.parameters():
                p.requires_grad = True

        def retrain_layer_from_base(self, dataset, eval_freq, epoch, layer):
            if self._optim == None: raise ConnectionError('Optimizer not connected. Use attach_optimizer to connect optimizer.')
            if self._base[0] == None:
                raise LookupError('base not created. create a copy first using commit_to_base.')

            if layer == len(self._conv_idx) - 1:
                raise IndexError('cannot retrain last layer. choose previous layer')

            from_layer = self._conv_idx[layer]
            to_layer = self._conv_idx[layer + 1]
            modules = list(self.children())[0]._modules

            loss_metric = nn.MSELoss()
            collate_loss = torch.zeros(1)
            full_base = list(self._base[0].children())[0]
            pre_base = nn.Sequential(*full_base[:from_layer])
            post_base = nn.Sequential(*full_base[:to_layer+1])
            for i in range(from_layer):
                for p in modules[str(i)].parameters():
                    p.requires_grad = False
            for i in range(to_layer+1, len(modules)):
                for p in modules[str(i)].parameters():
                    p.requires_grad = False
            for p in self._c.parameters():
                p.requires_grad = False

            step = 0
            for e in range(1, epoch+1):
                for data, _ in dataset.train_images:
                    data = data.cuda()
                    self.train()
                    with torch.no_grad():
                        input = pre_base(data)
                        target = post_base(data)
                    self._optim.zero_grad()
                    output = self.forward(input, inject_at_layer=from_layer, extract_from_layer=to_layer)
                    try:
                        loss = loss_metric(output, target)
                    except:
                        raise AssertionError('Base mismatch')
                    loss.backward()
                    self._optim.step()
                    collate_loss += loss
                    step += 1
                    sys.stdout.write('\rIter:%i ' % step)
                    sys.stdout.flush()
                    if step % eval_freq == 0:
                        print('--> Loss:%.4f' % (collate_loss.item()/eval_freq))
                        collate_loss = torch.zeros(1)
                print('\rEpoch', e, 'done.')

            for p in self.parameters():
                p.requires_grad = True

        def commit_to_base(self):
            print('Updating base')
            self._base[0] = nn.Sequential(
                copy.deepcopy(self._fe),
                copy.deepcopy(self._c)
            )
            self._base[0].eval()
            for p in self._base[0].parameters():
                p.requires_grad = False

        def evaluate(self, dataset):
            self.eval()
            correct_predictions = torch.zeros(1)
            correct_topk_predictions = torch.zeros(1)
            with torch.no_grad():
                for data, labels in dataset.eval_images:
                    data, labels = data.cuda(), labels.cuda()
                    output = self.forward(data)
                    predictions = output.argmax(dim=1, keepdim=True)
                    correct_predictions += predictions.eq(labels.view_as(predictions)).sum()
                    _, predictions_topk = output.topk(5, 1, True, True)
                    predictions_topk = predictions_topk.t()
                    correct_topk_predictions += predictions_topk.eq(labels.view(1, -1).expand_as(predictions_topk)).sum()

            return 100*(correct_predictions/dataset.num_eval_images).item(), 100*(correct_topk_predictions/dataset.num_eval_images).item()

        def get_features(self, dataset, after_layer, num_batches):
            after_layer = self._conv_idx[after_layer]
            if after_layer > self._conv_idx[-1]:
                raise IndexError('after_layer = ' + str(after_layer) + ' is out of bounds. Max value = ' + str(self._conv_idx[-1]))

            features_set = []
            labels_set = []

            for i, (data, labels) in enumerate(dataset.train_images):
                with torch.no_grad():
                    features = self.forward(data.cuda(), inject_at_layer=0, extract_from_layer=after_layer)
                features_set.append(features.cpu().numpy())
                labels_set.append(labels.cpu().numpy())
                if i == num_batches-1:
                    break

            features_array = np.concatenate(features_set)
            labels_array = np.concatenate(labels_set)

            return features_array, labels_array

        def get_weights(self, layer, filter=None):
            c_idx = str(self._conv_idx[layer])
            modules = list(self.children())[0]._modules

            conv = modules[c_idx]
            conv_weight = conv.weight.data.clone()
            conv_bias = conv.bias.data.clone()

            if filter is not None:
                conv_weight = conv_weight[filter]
                conv_bias = conv_bias[filter]

            return conv_weight.cpu().numpy(), conv_bias.cpu().numpy()

        def prune(self, layer, filter, verbose=True):
            c0 = str(self._conv_idx[layer])
            b0 = str(self._conv_idx[layer]+1)
            c1 = str(self._conv_idx[layer+1])

            modules = list(self.children())[0]._modules
            if layer == len(modules) - 1:
                raise NotImplementedError('last layer cannot be pruned.')

            conv0 = modules[c0]
            conv0_out_channels = conv0.out_channels
            if filter >= conv0_out_channels:
                raise IndexError('filter index out of bounds. filter=' + str(filter) + '. max_filters=' + str(conv0_out_channels))

            def delete_index(tensor, at_index, dim=0):
                if dim == 0:
                    return torch.cat((tensor[:at_index,...], tensor[at_index+1:,...]))
                elif dim == 1:
                    return torch.cat((tensor[:,:at_index,...], tensor[:,at_index+1:,...]), dim=dim)

            if verbose:
                print('Removing filter=%d from conv_layer=%d, max_filters=%d' % (filter, layer, conv0_out_channels-1))

            self._pruned_state.append((layer, filter))

            conv0_in_channels = conv0.in_channels
            conv0_kernel_size = conv0.kernel_size[0]
            conv0_stride = conv0.stride[0]
            conv0_padding = conv0.padding[0]

            conv0_weight = conv0.weight.data.clone()
            conv0_bias = conv0.bias.data.clone()
            conv0_target_weight = delete_index(conv0_weight, at_index=filter)
            conv0_target_bias = delete_index(conv0_bias, at_index=filter)

            modules[c0] = nn.Conv2d(conv0_in_channels,
                                    conv0_out_channels-1,
                                    conv0_kernel_size,
                                    conv0_stride,
                                    conv0_padding)
            modules[c0].weight.data = conv0_target_weight
            modules[c0].bias.data = conv0_target_bias

            bn = modules[b0]
            bn_num_features = bn.num_features
            bn_weight = bn.weight.data.clone()
            bn_bias = bn.bias.data.clone()
            bn_running_mean = bn.running_mean.data.clone()
            bn_running_var = bn.running_var.data.clone()

            bn_target_num_features = bn_num_features - 1
            bn_target_weight = delete_index(bn_weight, at_index=filter)
            bn_target_bias = delete_index(bn_bias, at_index=filter)
            bn_target_running_mean = delete_index(bn_running_mean, at_index=filter)
            bn_target_running_var = delete_index(bn_running_var, at_index=filter)

            modules[b0] = nn.BatchNorm2d(bn_target_num_features)
            modules[b0].weight.data = bn_target_weight
            modules[b0].bias.data = bn_target_bias
            modules[b0].running_mean.data = bn_target_running_mean
            modules[b0].running_var.data = bn_target_running_var

            conv1 = modules[c1]
            conv1_in_channels = conv1.in_channels
            conv1_out_channels = conv1.out_channels
            conv1_kernel_size = conv1.kernel_size[0]
            conv1_stride = conv1.stride[0]
            conv1_padding = conv1.padding[0]

            conv1_weight = conv1.weight.data.clone()
            conv1_bias = conv1.bias.data.clone()
            conv1_target_weight = delete_index(conv1_weight, at_index=filter, dim=1)
            conv1_target_bias = conv1_bias

            modules[c1] = nn.Conv2d(conv1_in_channels-1,
                                    conv1_out_channels,
                                    conv1_kernel_size,
                                    conv1_stride,
                                    conv1_padding)
            modules[c1].weight.data = conv1_target_weight
            modules[c1].bias.data = conv1_target_bias

        def save_pruned_state(self, name):
            try:
                os.makedirs(name)
            except FileExistsError:
                print('Warning: A pruned_state with name='+name+' already exists. Overwriting...')

            file = open(name+'/pruned_state.txt', 'w+')
            for state in self._pruned_state:
                layer, filter = state
                file.write(str(layer)+','+str(filter)+'\n')
            file.close()
            torch.save(self.state_dict(), name+'/pruned_weights.ckpt')

        def restore_pruned_state(self, name):
            file = open(name+'/pruned_state.txt', 'r').read().strip().split('\n')
            self._pruned_state = []
            for data in file:
                layer, filter = data.strip().split(',')
                layer, filter = int(layer), int(filter)
                self._pruned_state.append((layer, filter))
                self.prune(layer, filter, verbose=False)
            self.load_state_dict(torch.load(name+'/pruned_weights.ckpt'))



class Datasets:

    def __init__(self, dataset, batch_size):
        self._dataset_list = ['MNIST', 'CIFAR10', 'CIFAR100', 'Imagenet']
        assert dataset in self._dataset_list, 'Dataset must be in ' + ' or '.join(self._dataset_list)

        self._root = '/home/milton/.torch/datasets/'

        if dataset == 'MNIST':
            self._train_dataset = torchvision.datasets.MNIST(self._root, train=True, download=False,
                                                            transform=torchvision.transforms.Compose([
                                                                torchvision.transforms.ToTensor(),
                                                                torchvision.transforms.Normalize(
                                                                    (0.1306604762738429,),
                                                                    (0.30810780717887876,)),
                                                            ]))
            self._eval_dataset = torchvision.datasets.MNIST(self._root, train=False, download=False,
                                                           transform=torchvision.transforms.Compose([
                                                               torchvision.transforms.ToTensor(),
                                                               torchvision.transforms.Normalize(
                                                                   (0.1306604762738429,),
                                                                   (0.30810780717887876,)),
                                                           ]))
        elif dataset == 'CIFAR10':
            self._train_dataset = torchvision.datasets.CIFAR10(self._root, train=True, download=False,
                                                              transform=torchvision.transforms.Compose([
                                                                  torchvision.transforms.RandomHorizontalFlip(),
                                                                  torchvision.transforms.RandomCrop(32, padding=4),
                                                                  torchvision.transforms.ToTensor(),
                                                                  torchvision.transforms.Normalize(
                                                                      (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                              ]))
            self._eval_dataset = torchvision.datasets.CIFAR10(self._root, train=False, download=False,
                                                             transform=torchvision.transforms.Compose([
                                                                 torchvision.transforms.ToTensor(),
                                                                 torchvision.transforms.Normalize(
                                                                     (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                             ]))
        elif dataset == 'CIFAR100':
            self._train_dataset = torchvision.datasets.CIFAR100(self._root, train=True, download=False,
                                                               transform=torchvision.transforms.Compose([
                                                                   torchvision.transforms.RandomHorizontalFlip(),
                                                                   torchvision.transforms.RandomCrop(32, padding=4),
                                                                   torchvision.transforms.ToTensor(),
                                                                   torchvision.transforms.Normalize(
                                                                       (0.50707516, 0.48654887, 0.44091784),
                                                                       (0.26733429, 0.25643846, 0.27615047)),
                                                               ]))
            self._eval_dataset = torchvision.datasets.CIFAR100(self._root, train=False, download=False,
                                                              transform=torchvision.transforms.Compose([
                                                                  torchvision.transforms.ToTensor(),
                                                                  torchvision.transforms.Normalize(
                                                                      (0.50707516, 0.48654887, 0.44091784),
                                                                      (0.26733429, 0.25643846, 0.27615047)),
                                                              ]))
        elif dataset == 'Imagenet':
            self._train_dataset = torchvision.datasets.ImageNet(self._root, train=True, download=False,
                                                               transform=torchvision.transforms.Compose([
                                                                   torchvision.transforms.RandomHorizontalFlip(),
                                                                   torchvision.transforms.RandomCrop(32, padding=4),
                                                                   torchvision.transforms.ToTensor(),
                                                                   torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                                                    (0.5, 0.5, 0.5)),
                                                               ]))
            self._eval_dataset = torchvision.datasets.ImageNet(self._root, train=False, download=False,
                                                              transform=torchvision.transforms.Compose([
                                                                  torchvision.transforms.ToTensor(),
                                                                  torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                                                   (0.5, 0.5, 0.5)),
                                                              ]))

        self.train_images = torch.utils.data.DataLoader(self._train_dataset, batch_size=batch_size, shuffle=True, num_workers=20)
        self.eval_images = torch.utils.data.DataLoader(self._eval_dataset, batch_size=batch_size, shuffle=False, num_workers=20)

        self.num_train_images = self._train_dataset.__len__()
        self.num_eval_images = self._eval_dataset.__len__()
        self.num_labels = self._train_dataset.classes.__len__()

