import subprocess

metric_name_choices = ['NLL', 'EM', 'F1']


param_dict = {
    'model': 'BiDAFplus',
    'eval_steps': 50000,
    'lr': 0.5,
    'l2_wd': 0,
    'drop_prob': 0.2,
    'metric_name': 'F1',
    'max_checkpoints': 5,
    'max_grad_norm': 5.0,
    'seed': 224,
    'ema_decay': 0.999,
}

experiments = { # model_name: [param_dict_name1, param_dict_name2]
    'BiDAFplus': ['gru_enc']
}

def shell_list(*, name, **kwargs):
    lst = ['python', 'train.py', '-n', name]

    for kwarg, arg in kwargs.items():
        pname = '--' + kwarg

        lst.append(pname)
        lst.append(arg)

    for kwarg, arg in param_dict.items():
        if kwarg not in kwargs.keys():
            pname = '--' + kwarg

            lst.append(pname)
            lst.append(arg)

    return lst





def run():
    pass

