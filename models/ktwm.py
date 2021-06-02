import keras
import copy
from keras import regularizers
from keras.models import Model
# noinspection PyPep8Naming
from keras import backend as K
from keras.utils import get_custom_objects
from keras.layers import Input, Softmax, Embedding, Add, Lambda, Dense
from keras.layers import RepeatVector, Layer, Concatenate, Reshape

from models.keras_transformer.extras import ReusableEmbedding, TiedOutputEmbedding
from models.keras_transformer.masks import PaddingMaskLayer, SequenceMaskLayer,FactPaddingMaskLayer, SelfPadMaskLayer
from models.keras_transformer.position import TransformerCoordinateEmbedding
from models.keras_transformer.ktwm_transformer_blocks import KTWMEncoderBlock, KTWMDecoderBlock
from models.keras_transformer.ktwm_fact_filter import FactFilter
from models.keras_transformer.ktwm_simulate_vector import SimulateVector
from models.keras_transformer.ktwm_filter_loss import Loss

class KTWMModel:
    def __init__(self, args, 
                 transformer_dropout: float = 0.05,
                 embedding_dropout: float = 0.05,
                 l2_reg_penalty: float = 1e-4,
                 use_same_embedding = True,
                 use_vanilla_transformer = True,
                 ):
        self.args = args
        self.transformer_dropout = transformer_dropout 
        self.embedding_dropout = embedding_dropout

        # prepare layers
        l2_regularizer = (regularizers.l2(l2_reg_penalty) if l2_reg_penalty else None)
        if use_same_embedding:
            self.encoder_embedding_layer = self.decoder_embedding_layer = ReusableEmbedding(
                self.args.vocab_size, self.args.embedding_dim,
                name='embeddings',
                # Regularization is based on paper "A Comparative Study on
                # Regularization Strategies for Embedding-based Neural Networks"
                # https://arxiv.org/pdf/1508.03721.pdf
                embeddings_regularizer=l2_regularizer)
        else:
            self.encoder_embedding_layer = ReusableEmbedding(
                self.args.vocab_size, self.args.embedding_dim,
                name='encoder_embeddings',
                embeddings_regularizer=l2_regularizer)
            self.decoder_embedding_layer = ReusableEmbedding(
                self.args.vocab_size, self.args.embedding_dim,
                name='decoder_embeddings',
                embeddings_regularizer=l2_regularizer)

        self.output_layer = TiedOutputEmbedding(
            projection_dropout=self.embedding_dropout,
            scaled_attention=True,
            projection_regularizer=l2_regularizer,
            name='word_prediction_logits')
        self.output_softmax_layer = Softmax(name='word_predictions')

        self.encoder_src_coord_embedding_layer = TransformerCoordinateEmbedding(
            self.args.src_seq_length,
            1 if use_vanilla_transformer else self.args.transformer_depth,
            name='encoder_src_coordinate_embedding')
        self.decoder_coord_embedding_layer = TransformerCoordinateEmbedding(
            self.args.tar_seq_length,
            1 if use_vanilla_transformer else self.args.transformer_depth,
            name='decoder_coordinate_embedding')

    def __get_encoder(self, input_layer, _name):
        print('This is in Encoder...')

        next_step_input, _ = self.encoder_embedding_layer(input_layer)
        if _name == 'src':
            self_attn_mask = PaddingMaskLayer(name='encoder_%s_src_self_mask'%_name, src_len=self.args.src_seq_length,
                pad_id=self.pad_id)(input_layer)
            next_step_input = self.encoder_src_coord_embedding_layer(next_step_input, step=0)
        elif _name == 'tar':
            self_attn_mask = PaddingMaskLayer(name='encoder_%s_src_self_mask'%_name, src_len=self.args.tar_seq_length,
                pad_id=self.pad_id)(input_layer)
            next_step_input = self.encoder_src_coord_embedding_layer(next_step_input, step=0)
        elif _name == 'fact':
            self_attn_mask = PaddingMaskLayer(name='encoder_%s_fact_self_mask'%_name, src_len=self.args.fact_number * self.args.fact_seq_length,
                pad_id=self.pad_id)(input_layer)
            #next_step_input = self.encoder_fact_coord_embedding_layer(next_step_input, step=0)
            next_step_input = self.encoder_src_coord_embedding_layer(next_step_input, step=0)
        encoder_embedding = next_step_input

        elem_number = 1
        if _name == 'fact':
            elem_number = self.args.fact_number

        for i in range(self.args.transformer_depth):
            encoder_block = KTWMEncoderBlock(
                    name='%s_ktwm_encoder'%_name + str(i), 
                    num_heads=self.args.num_heads,
                    elem_number=elem_number,
                    residual_dropout=self.transformer_dropout,
                    attention_dropout=self.transformer_dropout,
                    activation='relu',
                    vanilla_wiring=True) # use vanilla Transformer instead of Universal Transformer
            next_step_input = encoder_block([next_step_input, self_attn_mask])
        return next_step_input, encoder_embedding

    def __get_decoder(self, input_layer,
                      src_encoder_output, mutual_tar_src_mask,
                      fact_encoder_output, mutual_tar_fact_mask):
        print('This is in Decoder...')
        self_padding_mask = PaddingMaskLayer(name='decoder_self_padding_mask', src_len=self.args.tar_seq_length,
                                             pad_id=self.pad_id)(input_layer)
        seq_mask = SequenceMaskLayer()(input_layer)
        self_attn_mask = Add()([self_padding_mask, seq_mask])
        # greater than 1, means not be padded in both self_padding_mask and seq_mask
        self_attn_mask = Lambda(lambda x: K.cast(K.greater((x), 1), dtype='int32'), name='add_padding_seq_mask')(self_attn_mask)

        next_step_input, self.decoder_embedding_matrix = self.decoder_embedding_layer(input_layer)
        next_step_input = self.decoder_coord_embedding_layer(next_step_input, step=0)
        for i in range(self.args.transformer_depth):
            decoder_block = KTWMDecoderBlock(
                    name='ktwm_decoder' + str(i), 
                    num_heads=self.args.num_heads,
                    fact_number=self.args.fact_number,
                    residual_dropout=self.transformer_dropout,
                    attention_dropout=self.transformer_dropout,
                    activation='relu',
                    vanilla_wiring=True) # use vanilla Transformer instead of Universal Transformer
            next_step_input = decoder_block([next_step_input, self_attn_mask, \
                                             src_encoder_output, mutual_tar_src_mask, \
                                             fact_encoder_output, mutual_tar_fact_mask
                                            ])
        return next_step_input

    def get_model(self, pad_id):
        self.pad_id = pad_id
        inp_src = Input(name='src_input',
                      shape=(self.args.src_seq_length, ), 
                      dtype='int32'
                     )
        inp_tar = Input(name='answer_input',
                            shape=(self.args.tar_seq_length, ), 
                            dtype='int32',
                           )
        inp_facts = Input(name='facts_input',
                            shape=(self.args.fact_number, self.args.fact_seq_length,), 
                            dtype='int32',
                           )
        inp_fact_mask = Input(name='fact_mask',
                            shape=(self.args.fact_number, self.args.fact_seq_length,), 
                            dtype='float32',
                           )

        inp_tar_loss = Input(name='tar_loss_input',
                            shape=(self.args.tar_seq_length, 1), 
                            dtype='int32',
                           )
        inp_is_train = Input(name='is_train',
                            shape=(1, ), 
                            dtype='int32',
                           )


        # shape: (bs, sf_number, seq_len)
        inp_src_exp = Lambda(lambda x: K.expand_dims(x, axis=1))(inp_src)
        src_mask = SelfPadMaskLayer(pad_id=self.pad_id)(inp_src_exp)
        src_encoder_output, src_encoder_embedding = self.__get_encoder(inp_src_exp, 'src')

        #inp_fact_exp = Lambda(lambda x: K.expand_dims(x, axis=1))(inp_facts)
        fact_mask = SelfPadMaskLayer(pad_id=self.pad_id)(inp_facts)
        fact_encoder_output, _ = self.__get_encoder(inp_facts, 'fact')


        inp_tar_exp = Lambda(lambda x: K.expand_dims(x, axis=1))(inp_tar)
        tar_mask = SelfPadMaskLayer(pad_id=self.pad_id)(inp_tar_exp)
        tar_encoder_output, _ = self.__get_encoder(inp_tar_exp, 'tar')

        simulated_vector = SimulateVector(args=self.args)([src_encoder_output, src_mask])

        filter_fact_encoder_output, tar_fact_simis, simulate_fact_simis, final_simis = \
            FactFilter(args=self.args)([tar_encoder_output, fact_encoder_output, simulated_vector, fact_encoder_output, inp_is_train, tar_mask, fact_mask])

        mutual_tar_src_mask = PaddingMaskLayer(name='mutual_tar_src_mask', src_len=self.args.tar_seq_length,
                                            pad_id=self.pad_id)(inp_src_exp)

        # pad_id=10000 to make all of the simulated vector valid
        mutual_tar_fact_mask = PaddingMaskLayer(name='mutual_tar_fact_mask', src_len=self.args.tar_seq_length,
            pad_id=10000)(inp_facts)
 
        decoder_output = self.__get_decoder(inp_tar_exp, 
            src_encoder_output, mutual_tar_src_mask, 
            filter_fact_encoder_output, mutual_tar_fact_mask)

        decoder_output = Reshape((self.args.tar_seq_length, self.args.embedding_dim, ))(decoder_output)
        # build model part
        word_predictions = self.output_softmax_layer(
                self.output_layer([decoder_output, self.decoder_embedding_matrix]))
        print('word_predictions: ', word_predictions )

        loss = Loss(name='loss', pad_id=self.pad_id)([tar_fact_simis, simulate_fact_simis, final_simis, inp_fact_mask, \
            word_predictions, inp_tar_loss])

        model = Model(
                      inputs=[inp_src, inp_tar, inp_tar_loss, inp_facts, inp_fact_mask, inp_is_train],
                      outputs=[loss, word_predictions]
                     )

        #rets = [fact_encoder_output]
        rets = [tar_fact_simis, simulate_fact_simis, final_simis]
        emb_fn = K.function([inp_src, inp_tar, inp_tar_loss, inp_facts, inp_fact_mask, inp_is_train], rets)

        return model, emb_fn

