"""
There is two kind of logger structure. Appart from the tensorBoardOutputLogger
and tensorBoardScoreLogger(which I am planning to deprecate or leave there as example
using the main two loggers that I will exlpain now)

Data Type | logger_function | usage&strategy
--- | --- | ---
included in Trainer,loss | create_tb_logger | provide f_extract which process the trainer. It returns a function that can be registered to the Trainer through ClassificationTrainer.add_*_hook()
out,inp (forward) | add_writer_hook() | define a hookEnabler & use add_writer_hook to register the hook, use hookEnabler.get_hook_enabler to get function to register to the Trainer.
gradinp,gradout (backward) | add_writer_hook(is_forward=False) | same as above

add_writer_hook: You need to define a hookEnabler and provide it to the function.
"""
def add_writer_hook(hook_controller,model,layer_name,writer,unit_id=slice(None),
                        pick='out',f=lambda a:a,tag='mean',
                        is_forward=True, writer_mode='add_scalar'):
    if pick=='out':
        def hook(self_layer,inp,out):
            if hook_controller._hook_on:
                if is_forward:
                    dat = f(out[:,unit_id])
                else:
                    dat = f(out[0][:,unit_id])
                if writer_mode=='add_histogram':
                    getattr(writer,writer_mode)(f'{layer_name}_{pick}_{tag}/unit_{unit_id}',dat,hook_controller._step_id, bins='doane')
                else:
                    getattr(writer,writer_mode)(f'{layer_name}_{pick}_{tag}/unit_{unit_id}',dat,hook_controller._step_id)
                hook_controller._hook_on = False

    elif pick=='inp':
        def hook(self_layer,inp,out):
            if hook_controller._hook_on:
                if is_forward:
                    dat = f(inp[:,unit_id])
                else:
                    #CHECK for linear the input_grad is tuple of size 3. (out,inp,w)
                    dat = f(inp[1][:,unit_id])
                if writer_mode=='add_histogram':
                    getattr(writer,writer_mode)(f'{layer_name}_{pick}_{tag}/unit_{unit_id}',dat,hook_controller._step_id,bins='doane')
                else:
                    getattr(writer,writer_mode)(f'{layer_name}_{pick}_{tag}/unit_{unit_id}',dat,hook_controller._step_id)
                hook_controller._hook_on = False
    else:
        raise ValueError(f'pick named argument can be either "out" or "inp": recieved:{pick}')
    if is_forward:
        h=getattr(model,layer_name).register_forward_hook(hook)
    else:
        h=getattr(model,layer_name).register_backward_hook(hook)
    return h

class hookEnabler(object):
    """docstring for hookEnabler.
    the reason we need an object for episodic hook is that there is no way that
    a hook can see the step count and decide whether to log or not. We need a
    mechanism to enable hook to turn on and off. We do this through self.hook_on
    """
    def __init__(self):
        #layer: torch.nn.Module
        super(hookEnabler, self).__init__()
        self._hook_on = False
        self._step_id = -1

    def get_hook_enabler(self):
        def pre_batch_hook_enabler(trainer,**kwargs):
            self._hook_on = True
            self._step_id = trainer.step
        return pre_batch_hook_enabler

def stdout_print(trainer,loss):
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            trainer.epoch, trainer.batch_id * len(trainer.batch_input), len(trainer.train_loader.dataset),
            100. * trainer.batch_id / len(trainer.train_loader), loss.data[0]), end='\r')

def get_batch_loss_logger(writer,tag='training_loss'):
    def batch_loss_logger(trainer,loss,**kwargs):
        writer.add_scalar(tag,loss.data[0],trainer.step)
    return batch_loss_logger

def create_tb_logger(writer,log_name,f_extract,writing_mode='add_scalar'):
    def logger(trainer,_,**kwargs):
        if writing_mode=='add_histogram':
            getattr(writer,writing_mode)(log_name,f_extract(trainer.model),trainer.step, bins='doane')
        else:
            getattr(writer,writing_mode)(log_name,f_extract(trainer.model),trainer.step)
    return logger
