import os
import time
import torch
from tqdm import tqdm

from ..datasets.text import get_dataset, get_metric
from ..backbones.text import get_net, get_tokenizer
from ..utils import set_random, create_log_dir, get_logger, init_wandb, log_wandb, finish_wandb



class TextClassifier():
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
        self._init_dataloader(data_name, net_name)
        self._init_model(net_name, gpu_index, num_epochs)
        self._init_logger(algorithm_name, data_name, net_name, num_epochs, random_seed)


    def _init_dataloader(self, data_name, net_name):
        # standard:  'sst2'
        # noise:     'sst2-noise-0.4', 
        self.tokenizer = get_tokenizer(net_name)
        self.dataset, dataset = get_dataset(data_name, self.tokenizer)  # data format is dict: {train, valid, test}
        self.metric, self.metric_name = get_metric(data_name)
    
        self.train_loader = torch.utils.data.DataLoader(
            dataset['train'], batch_size=50, shuffle=True, pin_memory=True)
        if 'mnli' in data_name:
            self.valid_loader = [torch.utils.data.DataLoader(
                dataset[x], batch_size=50, pin_memory=True) for x in ['validation_matched', 'validation_mismatched']]
            self.test_loader = [torch.utils.data.DataLoader(
                dataset[x], batch_size=50, pin_memory=True) for x in ['test_matched', 'test_mismatched']]
        else:
            self.valid_loader = [torch.utils.data.DataLoader(
                dataset['validation'], batch_size=50, pin_memory=True)]
            self.test_loader = [torch.utils.data.DataLoader(
                dataset['test'], batch_size=50, pin_memory=True)]

        self.data_prepare(self.train_loader, metric=self.metric, metric_name=self.metric_name)        # curriculum part


    def _init_model(self, net_name, gpu_index, num_epochs):
        self.net = get_net(net_name, self.dataset, self.tokenizer)
        self.device = torch.device('cuda:%d' % (gpu_index) \
            if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

        self.epochs = num_epochs
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        if net_name in ['bert', 'gpt']:                                 # for pretrained bert, gpt
            self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=2e-5)
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
        else:                                                           # for lstm
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=1.0)                          
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=1e-5)

        self.model_prepare(self.net, self.device, self.epochs,          # curriculum part
            self.criterion, self.optimizer, self.lr_scheduler)


    def _init_logger(self, algorithm_name, data_name, 
                     net_name, num_epochs, random_seed):
        self.log_interval = 1
        self.best_metrics = [-10.0] * len(self.valid_loader)
        log_info = '%s-%s-%s-%d-%d' % (
            algorithm_name, data_name, net_name, num_epochs, random_seed)
        self.log_dir = create_log_dir(log_info)
        self.logger = get_logger(os.path.join(self.log_dir, 'train.log'), algorithm_name)


    def _wandb_config(self):
        return {
            'algorithm': self.algorithm_name,
            'dataset': self.data_name,
            'model': self.net_name,
            'epochs': self.epochs,
            'seed': self.random_seed,
            'gpu_index': self.gpu_index,
            'log_dir': self.log_dir,
            'metric_name': self.metric_name,
        }


    def _start_wandb(self):
        init_wandb(config=self._wandb_config(), name=os.path.basename(self.log_dir))


    def _gpu_memory(self):
        return torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else 0


    def _train(self):
        for epoch in range(self.epochs):
            t = time.time()
            total = 0
            train_loss = 0.0
            predictions, references = [], []

            loader = self.data_curriculum()                             # curriculum part
            net = self.model_curriculum()                               # curriculum part

            net.train()
            for data in tqdm(loader):
                inputs = {k: v.to(self.device) for k, v in data.items() 
                          if k not in ['labels', 'indices']}
                labels = data['labels'].long().to(self.device)
                indices = data['indices'].to(self.device)

                self.optimizer.zero_grad()
                outputs = net(**inputs)[0] # logits, (hidden_states), (attentions)
                loss = self.loss_curriculum(outputs, labels, indices)   # curriculum part
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                self.optimizer.step()

                train_loss += loss.item() * len(labels)
                total += len(labels)
                references += labels.tolist()
                predictions += outputs.argmax(dim=1).tolist()
            
            self.lr_scheduler.step()
            train_metric = self.metric.compute(predictions=predictions, references=references)[self.metric_name]
            self.logger.info(
                '[%3d]  Train Data = %6d  Train %s = %.4f  Loss = %.4f  Time = %.2fs'
                % (epoch + 1, total, self.metric_name.capitalize(), train_metric, train_loss / total, time.time() - t))

            if (epoch + 1) % self.log_interval == 0:
                valid_metrics = [self._valid(valid_loader) for valid_loader in self.valid_loader]
                if valid_metrics[0] > self.best_metrics[0]:   # for mnli, choose best acc in validation_matched
                    self.best_metrics[:] = valid_metrics[:]
                    torch.save(net.state_dict(), os.path.join(self.log_dir, 'net.pkl'))
                for valid_loader, valid_metric, best_metric in zip(self.valid_loader, valid_metrics, self.best_metrics):
                    self.logger.info(
                        '[%3d]  Valid Data = %6d  Valid %s = %.4f  Best Valid %s = %.4f' 
                        % (epoch + 1, len(valid_loader.dataset), self.metric_name.capitalize(), valid_metric, self.metric_name.capitalize(), best_metric))
                wandb_metrics = {
                    'epoch': epoch + 1,
                    'train/loss': train_loss / total,
                    f'train/{self.metric_name}': train_metric,
                    'lr': self.lr_scheduler.get_last_lr()[0],
                    'gpu/mem_allocated': self._gpu_memory(),
                }
                for i, metric in enumerate(valid_metrics):
                    wandb_metrics[f'valid/{self.metric_name}_{i}'] = metric
                for i, metric in enumerate(self.best_metrics):
                    wandb_metrics[f'best/valid_{self.metric_name}_{i}'] = metric
                log_wandb(wandb_metrics, step=epoch + 1)
            

    def _valid(self, loader):
        predictions, references = [], []

        self.net.eval()
        with torch.no_grad():
            for data in tqdm(loader):
                inputs = {k: v.to(self.device) for k, v in data.items() 
                          if k not in ['labels', 'indices']}
                labels = data['labels'].long().to(self.device)
                outputs = self.net(**inputs)[0]
                references += labels.tolist()
                predictions += outputs.argmax(dim=1).tolist()
        return self.metric.compute(predictions=predictions, references=references)[self.metric_name]


    def fit(self):
        set_random(self.random_seed)
        self._start_wandb()
        starttime = time.time()
        self._train()
        endtime = time.time()
        self.logger.info("Training Time = %ds" % (endtime - starttime))
        self.logger.info("Training Mem = %dB" % (self._gpu_memory()))
        log_wandb({
            'time/train_seconds': endtime - starttime,
            'gpu/mem_allocated_final': self._gpu_memory(),
        })   


    def evaluate(self, net_dir=None):
        self._start_wandb()
        self._load_best_net(net_dir)
        valid_metrics = [self._valid(valid_loader) for valid_loader in self.valid_loader]
        test_metrics = [self._valid(test_loader) for test_loader in self.test_loader]
        for valid_loader, valid_metric, test_loader, test_metric in zip(self.valid_loader, valid_metrics, self.test_loader, test_metrics):
            self.logger.info('Valid Data = %6d  Final Valid %s = %.4f' % (len(valid_loader.dataset), self.metric_name.capitalize(), valid_metric))
            self.logger.info('Test Data  = %6d  Final Test %s = %.4f' % (len(test_loader.dataset), self.metric_name.capitalize(), test_metric))
        log_wandb({
            **{f'eval/valid_{self.metric_name}_{i}': metric for i, metric in enumerate(valid_metrics)},
            **{f'eval/test_{self.metric_name}_{i}': metric for i, metric in enumerate(test_metrics)},
            'gpu/mem_allocated': self._gpu_memory(),
        })
        finish_wandb()


    def export(self, net_dir=None):
        self._load_best_net(net_dir)
        return self.net


    def _load_best_net(self, net_dir):
        if net_dir is None: net_dir = self.log_dir
        net_file = os.path.join(net_dir, 'net.pkl')
        assert os.path.exists(net_file), 'Assert Error: the net file does not exist'
        self.net.load_state_dict(torch.load(net_file, map_location=self.device))
