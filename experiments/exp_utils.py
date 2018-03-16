from torch.autograd import Variable
from collections import defaultdict
import yaml
import argparse
import torch
import copy
from torch.nn import Module

class ClassificationTester(object):
    """
    This class is very similar to the ClassificationTrainer.test, however it does uses
    a fixed size data set and provides hook structure for logging.
    """
    def __init__(self, model,
                    criterion,
                    test_loader,
                    val_size=1000,
                    is_cuda=False):
        self.criterion = criterion
        self.model = model
        self.is_cuda = is_cuda
        self.x = Variable(torch.cat([test_loader.dataset[i][0].unsqueeze(0) for i in range(val_size)]),requires_grad=False)
        self.y = Variable(torch.LongTensor([test_loader.dataset[i][1] for i in range(val_size)]),requires_grad=False)
        if self.is_cuda:
            self.x, self.y = self.x.cuda(), self.y.cuda()

    def add_prebatch_hook(self,f):
        if not hasattr(self, 'prebatch_hooks'):
            self.prebatch_hooks = []
        self.prebatch_hooks.append(f)

    def add_afterbatch_hook(self,f):
        if not hasattr(self, 'afterbatch_hooks'):
            self.afterbatch_hooks = []
        self.afterbatch_hooks.append(f)

    def _default_print(self,loss):
        # afterbatch hook format
        acc = 100. * self.correct / len(self.y)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss, self.correct, len(self.y), acc))

    def apply_prebatch_hooks(self):
        if hasattr(self, 'prebatch_hooks'):
            for f in self.prebatch_hooks:
                f(self)

    def apply_afterbatch_hooks(self,loss):
        #affter backprop before step
        if hasattr(self, 'afterbatch_hooks'):
            for f in self.afterbatch_hooks:
                f(self,loss)
    def get_trainer_hook(self):
        def hook(trainer,*rest):
            self.step = trainer.step
            trainer.optimizer.zero_grad()
            self.model.eval()
            self.test()
            self.model.train()
            trainer.optimizer.zero_grad()
        return hook
    def calculate_loss(self,model=None,x=None,y=None):
        if not model:
            model = self.model
        if not x:
            x = self.x
        if not y:
            y = self.y
        return self.criterion(model(x), y)

    def test(self):
        self.model.eval()
        self.apply_prebatch_hooks()
        self.output = self.model(self.x)
        loss = self.criterion(self.output, self.y)
        # import pdb;pdb.set_trace()
        pred = self.output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        self.correct = pred.eq(self.y.data.view_as(pred)).cpu().sum()
        loss.backward()
        self.apply_afterbatch_hooks(loss)
        return loss

class ClassificationTrainer(object):
    """
    Similar to torch.utils.trainer. It provides prebatch and prestep hooks.
    Look .train(.) to see when they are activated.
    """
    def __init__(self,model,
                    criterion,
                    train_loader,
                    test_loader,
                    optimizer,
                    is_cuda=False):
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.is_cuda = is_cuda
        self.step = 0
        self.epoch = 0
        self.batch_id = 0

    def add_prebatch_hook(self,f,interval):
        if not hasattr(self, 'prebatch_hooks'):
            self.prebatch_hooks = defaultdict(list)
        self.prebatch_hooks[interval].append(f)

    def add_prestep_hook(self,f,interval):
        if not hasattr(self, 'prestep_hooks'):
            self.prestep_hooks = defaultdict(list)
        self.prestep_hooks[interval].append(f)

    def apply_prebatch_hooks(self):
        if hasattr(self, 'prebatch_hooks'):
            for freq in self.prebatch_hooks.keys():
                if self.step % freq == 0:
                    for f in self.prebatch_hooks[freq]:
                        f(self)

    def apply_prestep_hooks(self,loss):
        #affter backprop before step
        if hasattr(self, 'prestep_hooks'):
            for freq in self.prestep_hooks.keys():
                if self.step % freq == 0:
                    for f in self.prestep_hooks[freq]:
                        f(self,loss)

    def train(self,epoch,early_stop=0):
        """
        params: early_stop 0 for no early stop. Otherwise a postive value
        """
        self.model.train()
        self.epoch = epoch
        for batch_idx, (batch_input, batch_target) in enumerate(self.train_loader):
            self.batch_id = batch_idx+1
            self.optimizer.zero_grad()
            self.apply_prebatch_hooks()
            if self.is_cuda:
                batch_input, batch_target = batch_input.cuda(), batch_target.cuda()
            self.batch_input, self.batch_target = Variable(batch_input), Variable(batch_target)
            self.batch_output = self.model(self.batch_input)
            loss = self.criterion(self.batch_output, self.batch_target)
            loss.backward(retain_graph=True)

            self.apply_prestep_hooks(loss)
            self.step += 1
            self.optimizer.step()
            if early_stop>0 and self.batch_id==early_stop:
                print('early stop after batch{}'.format(batch_idx))
                break
        print(f'\nTrain Epoch: {epoch}...done')


    def test(self,is_print=True):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in self.test_loader:
            if self.is_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data,volatile=True), Variable(target)
            output = self.model(data)
            test_loss += self.criterion(output, target, size_average=False).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(self.test_loader.dataset)
        acc = 100. * correct / len(self.test_loader.dataset)
        if is_print:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(self.test_loader.dataset),acc))
        return test_loss,acc

def eval_if_prefixed_recursively(d,prefix='+'):
    """
    call this function on a dictionary if you wanna evaluate some specially prefixed string
    the eval string can be any valid python command.
    Ex:
        lr: +range(0.001,0.01,0.001)
    IDEA: you may want to change prefix to wildflags in the future if needed
    """
    if isinstance(d,dict):
        for k,val in d.items():
            if isinstance(val,str) and val.startswith(prefix):
                if len(val)>len(prefix):
                    d[k] = eval(val[len(prefix):])
                else:
                    #nothing to eval
                    d[k] = val
            else:
                d[k] = eval_if_prefixed_recursively(val,prefix=prefix)
    elif isinstance(d,list):
        for i,val in enumerate(d):
            if isinstance(val,str) and val.startswith(prefix) and len(val)>len(prefix):
                if len(val)>len(prefix):
                    d[i] = eval(val[len(prefix):])
                else:
                    #nothing to eval
                    d[i] = val
            else:
                d[i] = eval_if_prefixed_recursively(val,prefix=prefix)
    return d

def create_arg_parser(arg_dict):
    """
    Creates ArgParser from the a dict. If boolean values exist then a flag created,
    which basically switches the default value if provided.

    """
    parser = argparse.ArgumentParser()
    for k,v in arg_dict.items():
        if isinstance(v,bool):
            # So if we have bool, the commandline arg would reverse it.
            if v:
                parser.add_argument(f'--{k}', action='store_false', help=f'flip {k} to false ')
            else:
                parser.add_argument(f'--{k}', action='store_true', help=f'flip {k} to true ')
        else:
            parser.add_argument(f'--{k}', default=v, type=type(v),help=f'provide one arg with {k} of type: {type(v)}')

    return parser

def read_yaml_args(file_name):
    """
    assumes single level dict. if you have list or dict nested as values in the first multilevel,
    then you can't change them. So put the args that you may wanna overwrite on the
    console to the first level and make sure that they are str,bool,int or float.
    """
    with open(file_name, 'r') as stream:
        try:
            d=yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise(esc)
    return create_arg_parser(d)

def dump_yaml_args(dest,options):
    """
    Saves the provided argParse namespace as yaml.
    """
    with open(dest,'w') as stream:
        yaml.dump(options.__dict__, stream)
