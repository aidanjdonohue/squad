# TODO restructure this more naturallay

'''USAGE

params = master[model_name][param_config_name]

'''




class BiDAFModelParameters():
    def __init__(self, hidden_size=100, drop_prob=0.2, embedding_layer=None, encoder_layer=None, attention_layer=None, modeling_layer=None, output_layer=None):

        # defaults
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        
        self.d = hidden_size * 2

        self.embedding_layer = {
            'hwy_layers': 2,
            'char_embedder': 'cnn',
            'kernel_size': (1,5),
        }


        self.encoder_layer = {
            'rnn': 'lstm',
            'layers': 1,
            'hidden_size': self.d
        }


        self.attention_layer = {

        }

        self.modeling_layer = {
            'rnn': 'lstm',
            'layers': 2,
            'hidden_size': self.d, # maybe 2*self.hidden_size
        }

        # doesn't affect model yet
        self.output_layer = {
            'rnn': 'lstm',
            'layers': 2,
            'hidden_size': self.d, # maybe 2*self.hidden_size
        }

        if embedding_layer is not None:
            self.embedding_layer.update(embedding_layer)

        if encoder_layer is not None:
            self.encoder_layer.update(encoder_layer)

        if attention_layer is not None:
            self.attention_layer.update(attention_layer)

        if modeling_layer is not None:
            self.modeling_layer.update(modeling_layer)

        if output_layer is not None:
            self.output_layer.update(output_layer)


        
master = {
    'BiDAF' : {
        'default' : BiDAFModelParameters(),
        'gru' : BiDAFModelParameters(
                    encoder_layer={'rnn': 'gru'},
                    modeling_layer={'rnn': 'gru'}
                )
    },
    'BiDAFplus' : {
        'default' : BiDAFModelParameters(),
        'gru' : BiDAFModelParameters(
                    encoder_layer={'rnn': 'gru'},
                    modeling_layer={'rnn': 'gru'}
                )
    }
}
'''
master = { 
    'BiDAF': { 
        'default': {
            'phrase_encoder': 'lstm',
            'encoder_layers': 1,
            'hwy_layers': 2,
            'kernel_size': (1,5),
            'hidden_size': 100,
            'drop_prob': 0.2,
            'modeler': 'lstm',
            'modeler_layers': 2,
            'char_embedder': 'cnn',
            'output_encoder': 'lstm',
            'output_layers': 1,

        },
    },
    'BiDAFplus': {
        'default': {
            'phrase_encoder': 'lstm', #'gru'
            'out_channels': 100,
            'filters': [[1,5]],
            'drop_prob': 0.2,
            'hwy_layers': 2,
            'modeler_layers': 2,
            'encoder_layers': 1,
        },
        'gru_enc': {
            'phrase_encoder': 'gru', #'gru'
            'out_channels': 100,
            'filters': [[1,5]],
            'drop_prob': 0.2,
            'hwy_layers': 2,
            'modeler_layers': 2,
            'encoder_layers': 1,
        },

    },
    'SelfAttention': {
        'default': {

        },
    },
}
'''



def get_params(model_name, name='default'):


    return master[model_name][name]

