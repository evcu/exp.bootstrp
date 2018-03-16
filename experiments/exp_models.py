import torch
import torch.nn as nn
import torch.nn.functional as F


def _find_fan_out_weights(defs_conv,defs_fc,layer_name,unit_index):
    """
    Arguments:
        - unit_index: int or slice
    finding outgoing layer and the indices associated with the unit provided.
    1. Find layer, check unit_index
    2. Save consecutive layer
    3. If the layer=conv next_layer=fc we need to update the slice
    Example:

        self.defs_conv = [1,
                        ('conv1',8,5,2),
                        ('conv2',16,5,2)]
        self.defs_fc = [16*16,
                        ('fc1',64),
                        ('fc2',10)]
    """
    is_found = False
    next_layer = False

    #if slice we need better error checking
    if isinstance(unit_index,slice):
        compare_f = lambda i,n_out: 0<=min(i.start,i.stop) and max(i.start,i.stop)<n_out
    else:
        compare_f = lambda i,n_out: 0<=i<n_out

    for l,n_out,_,_ in defs_conv[1:]:
        if is_found:
            next_layer = l
            break
        if layer_name==l:
            is_found = True
            #check index is valid
            if not compare_f(unit_index,n_out):
                raise ValueError(f'index:{unit_index} is not 0<=x<{n_out} at layer: {layer_name}')
    if not next_layer:
        if is_found:
            #This means the last conv layer is the layer and we need a special handling
            #since there are might be multiple weights associated with theses outputs
            next_layer = defs_fc[1][0]
            conv_out_numel = defs_fc[0]//n_out
            #we need to expand the slice
            if isinstance(unit_index,slice):
                unit_index = slice(unit_index.start*conv_out_numel,unit_index.stop*conv_out_numel)
            else:
                start_index = unit_index*conv_out_numel
                unit_index = slice(start_index,start_index+conv_out_numel)
        else:
            for l,n_out in defs_fc[1:]:
                if is_found:
                    next_layer = l
                    break
                if layer_name==l:
                    is_found = True
                    if not compare_f(unit_index,n_out):
                        raise ValueError(f'index:{unit_index} is not 0<=x<{n_out} at layer: {layer_name}')
    return next_layer,unit_index




class ConvNet_generic(nn.Module):
    """
    Default is MNIST_8_16
    Params:
        - drop_out: None or p

    """
    def_archs=  {'mnist':('MNIST_8_16',
                                [1,
                                 ('conv1',8,5,2),
                                 ('conv2',16,5,2)],
                                [16*16,
                                 ('fc1',64),
                                 ('fc2',10)]),
                  'cifar10':('CIFAR10_8_16',
                                [3,
                                 ('conv1',8,5,2),
                                 ('conv2',16,5,2)],
                                [25*16,
                                 ('fc1',64),
                                 ('fc2',10)])
                  }
    def __init__(self,non_linearity=F.relu,batch_norm=False,drop_out=False,arch=None,bn_is_affine=False):
        super(ConvNet_generic, self).__init__()
        self.f=non_linearity
        self.layers_helper = {}
        self.drop_out = drop_out
        self.batch_norm = batch_norm

        #first element is n_input_features
        #other elements are each a conv layer with (name,num_inp_features,
        #                                               kernel_size,pooling_size
        if arch is None:
            arch = ConvNet_generic.def_archs['mnist']
        self.model_name = arch[0]
        self.defs_conv = arch[1]
        self.defs_fc = arch[2]
        in_feature = self.defs_conv[0]
        for name,out_feature,k_size,_ in self.defs_conv[1:]:
            setattr(self, name, nn.Conv2d(in_feature,out_feature,kernel_size=k_size))
            if self.drop_out:
                setattr(self, name+'_drop', nn.Dropout2d(self.drop_out))
            if self.batch_norm:
                setattr(self, name+'_bn', nn.BatchNorm2d(out_feature,affine=bn_is_affine))
            in_feature = out_feature
        in_feature = self.defs_fc[0]

        for name,out_feature in self.defs_fc[1:-1]:
            setattr(self, name, nn.Linear(in_feature,out_feature))
            if self.drop_out:
                setattr(self, name+'_drop', nn.Dropout(self.drop_out))
            if self.batch_norm:
                setattr(self, name+'_bn', nn.BatchNorm1d(out_feature,affine=bn_is_affine))
            in_feature = out_feature
        last_layer = self.defs_fc[-1]
        setattr(self,last_layer[0], nn.Linear(in_feature,last_layer[1]))

        self.nonlins = {'conv1':('max_relu',(2,2)),'conv2':('max_relu',(2,2)),'fc1':'relu','fc2':'log_softmax'}

    def forward(self, x):
        for name,_,_,pooling_size in self.defs_conv[1:]:
            x = getattr(self,name)(x)
            if self.drop_out:
                x = getattr(self, name+'_drop')(x)
            x = self.f(F.max_pool2d(x, pooling_size))
            if self.batch_norm:
                x = getattr(self, name+'_bn')(x)
        x = x.view(-1,self.defs_fc[0])
        for name,_ in self.defs_fc[1:-1]:
            x = getattr(self,name)(x)
            if self.drop_out:
                x = getattr(self, name+'_drop')(x)
            x = self.f(x)
            if self.batch_norm:
                x = getattr(self, name+'_bn')(x)
        last_layer_name,_ = self.defs_fc[-1]
        x = F.log_softmax(getattr(self,last_layer_name)(x),dim=1)
        return x

    def get_fan_in_weights(self,layer_name,unit_index,is_bias=False,is_grad=False):
        """
        This is same as getting the weights summing up to a unit
        """
        if is_bias:
            if is_grad:
                return getattr(self,layer_name).bias.grad.data[unit_index]
            else:
                return getattr(self,layer_name).bias.data[unit_index]
        else:
            if is_grad:
                return getattr(self,layer_name).weight.grad.data[unit_index]
            else:
                return getattr(self,layer_name).weight.data[unit_index]

    def get_fan_out_weights(self,layer_name,unit_index,is_grad=False):
        """
        This is same as getting all weights originates from the unit

        """
        ##SEARCH
        next_layer,unit_index = _find_fan_out_weights(self.defs_conv,self.defs_fc,layer_name,unit_index)
        if not next_layer:
            raise ValueError(f'get_fan_out_weights<{layer_name},{unit_index}> called and {layer_name} looks like the last layer.')
        if is_grad:
            return getattr(self,next_layer).weight.grad.data[:,unit_index]
        else:
            return getattr(self,next_layer).weight.data[:,unit_index]

    def find_fan_out_weights(self,layer_name,unit_index):
        return _find_fan_out_weights(self.defs_conv,self.defs_fc,layer_name,unit_index)



def weight_init_xaivier_filtered(m):
    if isinstance(m,(torch.nn.Conv2d,torch.nn.Linear)):
        nn.init.xavier_uniform(m.weight)
