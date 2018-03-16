#!/usr/bin/env python

# coding: utf-8


from exp_utils import read_yaml_args,dump_yaml_args
import shutil
import os.path
from itertools import product
from exp_utils import eval_if_prefixed_recursively
import os

def create_python_calls(exp_dict,log_folder,conf_path):
    list_of_lines = []
    # line below makes each value a list
    product_ready_values = map(lambda a:[a] if isinstance(a,(int,bool,float,str)) else a,exp_dict.values())
    for arg_vals in product(*product_ready_values):
        args = ' '.join([f'--{k} {v}' for k,v in zip(exp_dict.keys(),arg_vals)])
        # print(args,arg_vals)
        list_of_lines.append(f'time python3 main.py --conf_file {conf_path} --log_folder {log_folder} {args}')
    return "\n".join(list_of_lines)

def create_experiment(args,folder_path,script_name='job_script',is_debug=False):
    list_of_jobs=[]
    for i,exp in enumerate(args.experiments):
        log_folder = f'{folder_path}tb_logs/'


        if not is_debug:
            os.makedirs(os.path.dirname(folder_path), exist_ok=True)
            os.makedirs(f'{folder_path}stderrout',exist_ok=True)
            dir_path = os.path.dirname(os.path.realpath(__file__))
            def_conf_path = os.path.join(dir_path,args.exp_name,'default_conf.yaml')
            shutil.copy(def_conf_path,folder_path)

        main_script = create_python_calls(exp,log_folder,
                                            os.path.join(folder_path,'default_conf.yaml'))

        sbatch_flags = args.sbatch_flags
        script_prep = f"""#!/bin/bash
#SBATCH --nodes={sbatch_flags['nodes']}
#SBATCH --gres={sbatch_flags['gres']}
#SBATCH --time={sbatch_flags['time']}
#SBATCH --mem={sbatch_flags['mem']}
#SBATCH --mail-type={sbatch_flags['mail-type']}
#SBATCH --mail-user={sbatch_flags['mail-user']}
#SBATCH --job-name={args.exp_name}/{args.exp_tag}_{i}
#SBATCH --output={folder_path}stderrout/{script_name}_{i}.out
#SBATCH --error={folder_path}stderrout/{script_name}_{i}.err

module load python3/intel/3.6.3
module load cuda/9.0.176
cd $HOME/M.Sc.thesis/experiments/{args.exp_name}/
"""

        script = f'{script_prep}\n{main_script}\n'
        file_path = f'{folder_path}{script_name}_{i}.slurm'
        list_of_jobs.append(file_path)
        if is_debug:
            print(f'File_path is: {file_path}')
            print(f'FILE_START_______________')
            print(script)
            print(f'EOF______________________')
        else:
            print(f'File: {file_path}',end=' ')
            with open(file_path,'w') as f:
                f.write(script)
            print(f'...written')
    submit_script_path = f'{folder_path}submit_all.sh'
    if is_debug:
        print('\n'.join([f'sbatch {j}' for j in list_of_jobs]))
    else:
        with open(submit_script_path,'w') as f:
            f.write('\n'.join([f'sbatch {j}' for j in list_of_jobs]))


    print(f'..to go to the experiment folder\ncd {folder_path}')
    print(f'..to sumbmit all scripts generated run\n bash {submit_script_path}')

if __name__ == "__main__":
    parser = read_yaml_args('default_conf.yaml')
    args = parser.parse_args()
    if args.eval_prefix:
        eval_if_prefixed_recursively(vars(args),prefix=args.eval_prefix)
    # dest_folder: $SCRATCH/my_project/exps/
    # exp_name: test_exp
    # exp_tag: 03.16.test
    folder_path = f'{args.dest_folder}{args.exp_name}/{args.exp_tag}/'

    # tdy = datetime.today()
    print(args)
    create_experiment(args,folder_path,is_debug=args.debug)
