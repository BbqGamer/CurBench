import os
import time
import torch

from ..datasets.graph import get_dataset
from ..backbones.graph import get_net
from ..utils import set_random, create_log_dir, get_logger, init_wandb, log_wandb, finish_wandb



class NodeClassifier():
    def __init__(self, data_name, net_name, gpu_index, num_epochs, random_seed, algorithm_name, 
                 data_prepare, model_prepare, data_curriculum, model_curriculum, loss_curriculum):
        self.data_name = data_name
        self.net_name = net_name
        self.gpu_index = gpu_index
        self.algorithm_name = algorithm_name
        self.random_seed = random_seed
        self.data_prepare = data_prepare
        self.model_prepare = model_prepare
        self.data_curriculum = data_curriculum
        self.model_curriculum = model_curriculum
        self.loss_curriculum = loss_curriculum

        set_random(self.random_seed)
        self._init_dataloader(data_name)
        self._init_model(net_name, gpu_index, num_epochs)
        self._init_logger(algorithm_name, data_name, net_name, num_epochs, random_seed)


    def _init_dataloader(self, data_name):
        self.dataset = get_dataset(data_name)
        # self.dataset.__setattr__('dataset', self.dataset[0])
        # TODO: for data_prepare


    def _init_model(self, net_name, gpu_index, num_epochs):
        self.net = get_net(net_name, self.dataset)
        self.device = torch.device('cuda:%d' % (gpu_index) \
            if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        self.data = self.dataset[0].to(self.device)

        self.epochs = num_epochs
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=0.01, weight_decay=5e-4)
        
        # TODO: model_prepare
    
    def _init_logger(self, algorithm_name, data_name, 
                     net_name, num_epochs, random_seed):
        self.log_interval = 1
        log_info = '%s-%s-%s-%d-%d' % (
            algorithm_name, data_name, net_name, num_epochs, random_seed)
        self.log_dir = create_log_dir(log_info)
        self.logger = get_logger(os.path.join(self.log_dir, 'train.log'))
        init_wandb(config=self._wandb_config(), name=log_info)


    def _wandb_config(self):
        return {
            'algorithm': self.algorithm_name,
            'dataset': self.data_name,
            'model': self.net_name,
            'epochs': self.epochs,
            'seed': self.random_seed,
            'gpu_index': self.gpu_index,
            'log_dir': self.log_dir,
        }


    def _start_wandb(self):
        init_wandb(config=self._wandb_config(), name=os.path.basename(self.log_dir))


    def _gpu_memory(self):
        return torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else 0


    def _train(self):
        best_acc = 0.0

        # TODO: data curriculum, model_curriculum and loss_curriculum
        for epoch in range(self.epochs):
            t = time.time()
            self.net.train()
            self.optimizer.zero_grad()
            masks = self.data.train_mask
            labels = self.data.y
            outputs = self.net(self.data)
            loss = self.criterion(outputs[masks], labels[masks])
            loss.backward()
            self.optimizer.step()

            train_loss = loss.item()
            predicts = outputs.argmax(dim=1)
            correct = predicts[masks].eq(labels[masks]).sum().item()
            total = masks.sum()

            self.logger.info(
                '[%3d]  Train Data = %6d  Train Acc = %.4f  Loss = %.4f  Time = %.2fs'
                % (epoch + 1, total, correct / total, train_loss, time.time() - t))
            
            if (epoch + 1) % self.log_interval == 0:
                valid_acc = self._valid(self.data.val_mask)
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    torch.save(self.net.state_dict(), os.path.join(self.log_dir, 'net.pkl'))
                self.logger.info(
                    '[%3d]  Valid Data = %6d  Valid Acc = %.4f  Best Valid Acc = %.4f' 
                    % (epoch + 1, self.data.val_mask.sum(), valid_acc, best_acc))
                log_wandb({
                    'epoch': epoch + 1,
                    'train/loss': train_loss,
                    'train/acc': correct / total,
                    'valid/acc': valid_acc,
                    'best/valid_acc': best_acc,
                    'gpu/mem_allocated': self._gpu_memory(),
                }, step=epoch + 1)


    def _valid(self, masks):
        self.net.eval()
        with torch.no_grad():
            labels = self.data.y
            outputs = self.net(self.data)
            predicts = outputs.argmax(dim=1)
            correct = predicts[masks].eq(labels[masks]).sum().item()
            total = masks.sum()
        return correct / total
    

    def fit(self):
        set_random(self.random_seed)
        self._start_wandb()
        starttime = time.time()
        self._train()
        endtime = time.time()
        self.logger.info("Training Time = %ds" % (endtime - starttime))
        self.logger.info("Training Mem  = %dB" % (self._gpu_memory()))
        log_wandb({
            'time/train_seconds': endtime - starttime,
            'gpu/mem_allocated_final': self._gpu_memory(),
        })


    def evaluate(self, net_dir=None):
        self._start_wandb()
        self._load_best_net(net_dir)
        valid_acc = self._valid(self.data.val_mask)
        test_acc = self._valid(self.data.test_mask)
        self.logger.info('Valid Data = %6d  Final Valid Acc = %.4f' % (int(self.data.val_mask.sum()), valid_acc))
        self.logger.info('Test Data  = %6d  Final Test Acc = %.4f' % (int(self.data.test_mask.sum()), test_acc))
        log_wandb({
            'eval/valid_acc': valid_acc,
            'eval/test_acc': test_acc,
            'gpu/mem_allocated': self._gpu_memory(),
        })
        finish_wandb()
        return test_acc


    def export(self, net_dir=None):
        self._load_best_net(net_dir)
        return self.net


    def _load_best_net(self, net_dir):
        if net_dir is None: net_dir = self.log_dir
        net_file = os.path.join(net_dir, 'net.pkl')
        assert os.path.exists(net_file), 'Assert Error: the net file does not exist'
        self.net.load_state_dict(torch.load(net_file, map_location='cuda:0'))
