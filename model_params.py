master = {
    'BiDAFplus': {
        'default': {
            'phrase_encoder': 'lstm', #'gru'
            'out_channels': 100,
            'filters': [[1,5]],
            'drop_prob': 0.2,
            'hwy_layers': 2,
            'model_layers': 2,
            'encoder_layers': 1,
        },

    },
    'SelfAttention': {
        'default': {

        },
    },
}



def get_params(model_name, name='default'):


    return master[model_name][name]

