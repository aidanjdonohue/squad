# TODO restructure this more naturallay

'''USAGE

params = master[model_name][param_config_name]

'''



class BiDAFModelParameters():
    def __init__(self, name, hidden_size=100, d_size=2, drop_prob=0.2, embedding_layer=None, encoder_layer=None, attention_layer=None, modeling_layer=None, output_layer=None):

        # defaults
        self.name = name
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        
        self.d = int(hidden_size * d_size)

        self.embedding_layer = {
            'drop_prob': self.drop_prob,
            'embedding_size': 100,
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
            'drop_prob': self.drop_prob / 2
        }

        self.modeling_layer = {
            'rnn': 'lstm',
            'layers': 2,
            'hidden_size': self.d, # maybe 2*self.hidden_size
        }

        # doesn't affect model yet
        self.output_layer = {
            'rnn': 'lstm',
            'layers': 1,
            'hidden_size': self.d, # maybe 2*self.hidden_size
        }

        self.filters = [] #For char embeddings idk what this actually does

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



    def __str__(self):


        string = \
        f"Param name  : {self.name}\n" +\
        f"hidden_size : {self.hidden_size}\n" +\
        f"drop_prob   : {self.drop_prob}\n" +\
        f'embedding_layer\n' + \
        str(self.embedding_layer)  + '\n' +\
        f'encoder_layer\n' + \
        str(self.encoder_layer)  + '\n' +\
        f'attention_layer\n' + \
        str(self.attention_layer)  + '\n' +\
        f'modeling_layer\n' + \
        str(self.modeling_layer)  + '\n' +\
        f'output_layer\n' + \
        str(self.output_layer)

        return string

class TransformerModelParameters():
    def __init__(self, name, hidden_size=100, d_size=2, drop_prob=0.2, embedding_layer=None, encoder_layer=None, attention_layer=None, modeling_layer=None):

        # defaults
        self.name = name
        self.hidden_size = hidden_size
        self.d = int(hidden_size * d_size)
        self.drop_prob = drop_prob
        
        self.embedding_layer = {
            'drop_prob': self.drop_prob,
            'embedding_size': 100,
            'hwy_layers': 2,
            'char_embedder': 'cnn',
            'kernel_size': (1,5),
        }

        self.encoder_layer = {
            "num_heads" : 8,
            'drop_prob' : self.drop_prob,
            "num_conv_blocks": 1,
            "input_dim" : self.d,
            "output_dim" : self.d,
            "hidden_dim" : 256,
            "kernel_size" : 5
        }

        # No impact rn
        self.attention_layer = {
        }

        self.modeling_layer = {
            "num_heads" : 8,
            'drop_prob' : self.drop_prob,
            "num_conv_blocks": 7,
            "input_dim" : 4 * hidden_size,
            "output_dim" : 4 * hidden_size,
            "hidden_dim" : 256,
            "kernel_size" : 5
        }

        if embedding_layer is not None:
            self.embedding_layer.update(embedding_layer)

        if encoder_layer is not None:
            self.encoder_layer.update(encoder_layer)

        if attention_layer is not None:
            self.attention_layer.update(attention_layer)

        if modeling_layer is not None:
            self.modeling_layer.update(modeling_layer)



    def __str__(self):


        string = \
        f"Param name  : {self.name}\n" +\
        f"hidden_size : {self.hidden_size}\n" +\
        f"d_model : {self.d}\n" +\
        f"drop_prob   : {self.drop_prob}\n" +\
        f'embedding_layer\n' + \
        str(self.embedding_layer)  + '\n' +\
        f'encoder_layer\n' + \
        str(self.encoder_layer)  + '\n' +\
        f'attention_layer\n' + \
        str(self.attention_layer)  + '\n' +\
        f'decoder_layer\n' + \
        str(self.decoder_layer)

        return string
        
master = {
    'BiDAF' : {
        'default' : BiDAFModelParameters('default'),
        'gru'     : BiDAFModelParameters('gru',
                    encoder_layer={'rnn': 'gru'},
                    modeling_layer={'rnn': 'gru'}

                    ),
        'big_lstm': BiDAFModelParameters('big_lstm',
                encoder_layer={'layers': 2},
                modeling_layer={'layers': 3},
            )

               
    },
    'BiDAFplus' : {
        'default' : BiDAFModelParameters('default'),
        'gru' : BiDAFModelParameters('gru',
                    encoder_layer={'rnn': 'gru'},
                    modeling_layer={'rnn': 'gru'}
                )
    },
    'Transformer' : {
        'default' : TransformerModelParameters('default')
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

    params = master[model_name][name]
    print("="*20)
    print("Model Params")
    print("-"*20)
    print(str(params))
    print("="*20)
    return params

