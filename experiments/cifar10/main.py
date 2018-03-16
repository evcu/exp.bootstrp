"""
- trainer calls tester_hook
- tester hook calls test() and zeros the gradient before and after
- tester calls apply_prebatch_hooks and calculates the loss on the test set and calls apply_afterbatch_hooks

"""


import sys
sys.path.insert(0,'../')
from exp_utils import read_yaml_args,dump_yaml_args
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from exp_utils import ClassificationTrainer,ClassificationTester,eval_if_prefixed_recursively
from exp_models import ConvNet_generic,weight_init_xaivier_filtered
from exp_dataloaders import getDataLoader
from exp_loggers import hookEnabler,add_writer_hook
from exp_loggers import create_tb_logger,stdout_print,get_batch_loss_logger

def run_experiment(args,log_folder_name,update_lr=True,write_graph=False):
    torch.set_default_tensor_type('torch.FloatTensor')
    is_cuda = torch.cuda.is_available() and args.use_cuda
    torch.manual_seed(args.seed)
    # This very important to keep speed up and the progress deterministic
    torch.backends.cudnn.deterministic = True
    # not needed with pytorch 0.3
    # if is_cuda:
    #     print('Cuda is enabled')
    #     torch.cuda.manual_seed_all(args.seed)

    train_loader,test_loader = getDataLoader('../data/cifar10','cifar10',train_batch=args.batch_size,test_batch=1024)
        # custom_model_def = ('CIFAR10_8_16_64',
        #               [3,
        #                ('conv1',8,5,2),
        #                ('conv2',16,5,2),
        #                ('conv3',64,5,1)],
        #               [1*64,
        #                ('fc1',32),
        #                ('fc2',10)])
    model = ConvNet_generic(batch_norm=args.use_batchnorm,non_linearity=getattr(F,args.nonlinearity),arch=ConvNet_generic.def_archs['cifar10'])
    model.apply(weight_init_xaivier_filtered)
    print(f'Following model is going to be used: {model}')
    if is_cuda:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


    trainer = ClassificationTrainer(model,
                                    F.nll_loss,
                                    train_loader,
                                    test_loader,
                                    optimizer,
                                    is_cuda = is_cuda)

    tester = ClassificationTester(model,
                                    F.nll_loss,
                                    test_loader,
                                    val_size=1000,
                                    is_cuda=is_cuda)

    if args.print_progress:
        trainer.add_prestep_hook(stdout_print,args.log_freq)

    trainer.add_prebatch_hook(tester.get_trainer_hook(),args.log_freq)

    custom_writer = SummaryWriter(log_folder_name)
    dump_yaml_args(f'{log_folder_name}/args.yaml',args)

    layer_name = args.layer
    unit_i = args.unit_id

    tester.add_afterbatch_hook(get_batch_loss_logger(custom_writer,f'loss/{layer_name}'))

    #layer weight,bias * grad,_
    trainer.add_prestep_hook(create_tb_logger(custom_writer,
                                                f'{layer_name}/unit_{unit_i}_w_norm',
                                                f_extract=lambda m: m.get_fan_in_weights(layer_name,
                                                                                        unit_i).norm())
                            ,args.log_freq)
    trainer.add_prestep_hook(create_tb_logger(custom_writer,
                                                f'{layer_name}/unit_{unit_i}_b',
                                                f_extract=lambda m: m.get_fan_in_weights(layer_name,
                                                                                        unit_i,
                                                                                        is_bias=True))
                            ,args.log_freq)

    trainer.add_prestep_hook(create_tb_logger(custom_writer,
                                                f'{layer_name}/unit_{unit_i}_w_norm_grad',
                                                f_extract=lambda m: m.get_fan_in_weights(layer_name,
                                                                                        unit_i,
                                                                                        is_grad=True).norm())
                            ,args.log_freq)

    trainer.add_prestep_hook(create_tb_logger(custom_writer,
                                                f'{layer_name}/unit_{unit_i}_b_grad',
                                                f_extract=lambda m: m.get_fan_in_weights(layer_name,
                                                                                        unit_i,
                                                                                        is_bias=True,
                                                                                        is_grad=True))
                            ,args.log_freq)



    #fanout weights
    trainer.add_prestep_hook(create_tb_logger(custom_writer,
                                                f'{layer_name}_fanout/unit_{unit_i}_w_norm',
                                                f_extract=lambda m: m.get_fan_out_weights(layer_name,
                                                                                        unit_i).norm())
                            ,args.log_freq)


    trainer.add_prestep_hook(create_tb_logger(custom_writer,
                                                f'{layer_name}_fanout/unit_{unit_i}_w_norm_grad',
                                                f_extract=lambda m: m.get_fan_out_weights(layer_name,
                                                                                        unit_i,
                                                                                        is_grad=True).norm())
                            ,args.log_freq)
    #output of the linear
    #pick='out',f=lambda a:a,tag='mean',
                        # is_forward=True, writer_mode='add_scalar'
    hist_f=lambda a: a.data.cpu().numpy()
    hist_f_nl=lambda a: model.f(a).data.cpu().numpy()
    hkEnabler_out = hookEnabler()
    _ = add_writer_hook(hkEnabler_out,model,layer_name,custom_writer,unit_id=unit_i,f=hist_f,tag='',writer_mode='add_histogram')
    tester.add_prebatch_hook(hkEnabler_out.get_hook_enabler())

    #output after non_linearity
    hkEnabler_out_nl = hookEnabler()
    _ = add_writer_hook(hkEnabler_out_nl,model,layer_name,custom_writer,unit_id=unit_i,f=hist_f_nl,tag=f'after_{args.nonlinearity}',writer_mode='add_histogram')
    tester.add_prebatch_hook(hkEnabler_out_nl.get_hook_enabler())


    if args.use_batchnorm:
        hkEnabler_bn_out = hookEnabler()
        _ = add_writer_hook(hkEnabler_bn_out,model,f'{layer_name}_bn',custom_writer,unit_id=unit_i,f=hist_f,tag='',writer_mode='add_histogram')
        tester.add_prebatch_hook(hkEnabler_bn_out.get_hook_enabler())


    for i in range(1,args.epoch+1):
        trainer.train(i,early_stop=args.early_stop)

    print(f'{trainer.step} steps performed in {args.epoch} epoch')

    tloss,tacc = trainer.test()
    custom_writer.add_text('summary',f'Loss is: {tloss},Acc is: {tacc}',trainer.step+1)
    custom_writer.add_text('args',f'Args are: {str(args)}',trainer.step+1)
    custom_writer.add_graph(model,tester.x)
    custom_writer.close()

if __name__ == "__main__":
    default_conf_path = 'default_conf.yaml'
    parser = read_yaml_args(default_conf_path)
    args = parser.parse_args()

    #this part enables a custom conf_file to be given.
    if args.conf_file != default_conf_path:
        parser = read_yaml_args(args.conf_file)
        args = parser.parse_args()
    if args.eval_prefix:
        eval_if_prefixed_recursively(vars(args),prefix=args.eval_prefix)
    print(args)
    tdy = datetime.today()
    if not args.log_folder.endswith('/'):
        print('WARNING: please provide a folder_name ending with a slash=/')
        args.log_folder = args.log_folder + '/'
    #time stamp at the end to prevent overwriting
    log_folder_name = f'{args.log_folder}{args.layer}_{args.unit_id}_{args.nonlinearity}_{tdy:%d_%m_%H_%M_%S}'

    run_experiment(args,log_folder_name)
