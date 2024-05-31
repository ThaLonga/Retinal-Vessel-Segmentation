# Building Unet by dividing encoder and decoder into blocks
import tensorflow as tf
from keras import Sequential
from keras.models import Model
from keras.layers import Input, Conv2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, UpSampling2D, Activation, MaxPool2D, Concatenate, Dense, Multiply, Add, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape
from keras.optimizers import Adam
import math

def CBRD(input, num_filters, dropout): #ok
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout)(x)
    return(x)

def CFFA(input, num_filters): #concat aangepast, denk ok
    fine = Conv2D(num_filters, (3, 1), padding="same")(input)
    fine = BatchNormalization()(fine)
    fine = Activation("relu")(fine)
    
    fine = Conv2D(num_filters, (1, 3), padding="same")(fine)
    fine = BatchNormalization()(fine)
    fine = Activation("relu")(fine)


    coarse = Conv2D(num_filters, 7, padding="same")(input)
    coarse = BatchNormalization()(coarse)  
    coarse = Activation("relu")(coarse)

    coarse = Conv2D(num_filters, 1, padding="same")(coarse)
    coarse = BatchNormalization()(coarse)  
    coarse = Activation("relu")(coarse)
    
    x = Concatenate()([fine, coarse])
    x = Conv2D(num_filters, 1, padding="same")(x)
    x = BatchNormalization()(x)  
    x = Activation("relu")(x)
    return x
    
def ACA(input):
    channel = input.shape[-1]
    shared_layer_one = Dense(channel//8,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
    shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	
    avg_pool = GlobalAveragePooling2D()(input)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel//8)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)

    max_pool = GlobalMaxPooling2D()(input)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1,1,channel//8)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)

    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    return(cbam_feature * input + input)
    
def MSCAF(input, out_channels, dropout):
    x1 = Conv2D(filters=out_channels//2, kernel_size=3, padding='same', use_bias=False)(input)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    
    x2 = Conv2D(filters=out_channels//4, kernel_size=3, dilation_rate=3, padding='same', use_bias=False)(input)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)

    x3 = Conv2D(filters=out_channels//4, kernel_size=3, dilation_rate=5, padding='same', use_bias=False)(input)
    x3 = BatchNormalization()(x3)
    x3 = Activation("relu")(x3)
    
    out = Concatenate()([x1, x2, x3])
    out = ACA(out)
    out = Conv2D(out_channels, 1, use_bias = False)(out)
    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Dropout(dropout)(out)
    return(out)
    
    
def padding(x,y):
    diffY = tf.shape(y)[1] - tf.shape(x)[1]
    diffX = tf.shape(y)[2] - tf.shape(x)[2]
    padding = [[0, 0], # No padding for the batch size dimension
       [diffY // 2, diffY - diffY // 2], # Pad height
       [diffX // 2, diffX - diffX // 2], # Pad width
       [0, 0]] # No padding for the channel dimension
    x_padded = tf.pad(x, padding, "CONSTANT")
    return x_padded

def PFF(input1, input2, input3, input4, num_filters, dropout):
    x1 = Conv2D(num_filters // 8, 1, use_bias = False)(input1)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    x1 = UpSampling2D(8, interpolation='bilinear')(x1)
    x1 = Dropout(dropout)(x1)

    x2 = Conv2D(num_filters // 8, 1, use_bias = False)(input2)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)
    x2 = UpSampling2D(4, interpolation='bilinear')(x2)
    x2 = Dropout(dropout)(x2)


    x3 = Conv2D(num_filters // 4, 1, use_bias = False)(input3)
    x3 = BatchNormalization()(x3)
    x3 = Activation("relu")(x3)
    x3 = UpSampling2D(2, interpolation='bilinear')(x3)
    x3 = Dropout(dropout)(x3)

    x4 = Conv2D(num_filters // 2, 1, use_bias = False)(input4)
    x4 = BatchNormalization()(x4)
    x4 = Activation("relu")(x4)
    x4 = Dropout(dropout)(x4)

    
    x1 = padding(x1, x4) # wrong: same size als input 4
    x2 = padding(x2, x4)
    x3 = padding(x3, x4)
    print(x1.shape)

    out = Concatenate()([x1, x2, x3, x4])
    print(out.shape)
    out = MSCAF(out, num_filters, dropout) + input4
    print(out.shape)
    return out

def encoder_block(input, num_filters, dropout): #ok
    x = CFFA(input, num_filters)
    x = Dropout(dropout)(x)
    x = CFFA(x, num_filters)
    x = Dropout(dropout)(x)
    return x
    
def decoder_block(input, num_filters, dropout):
    x = MSCAF(input, num_filters, dropout) 
    x = CBRD(x, num_filters, dropout)
    return x
    
def down(input, num_filters, dropout): #ok
    x = MaxPool2D(2)(input)
    x = encoder_block(x, num_filters, dropout)
    return x
    
def up(input, y, num_filters, dropout):
    x = Conv2DTranspose(num_filters, 2, strides=2, padding="same")(input)
    x = Dropout(dropout)(x)
    x_padded = padding(x, y)
    out = Concatenate()([x_padded, y]) #wrong place?
    out = decoder_block(out, num_filters, dropout)
    return out

def build_CMP_UNet(input_shape, dropout=0):
    inputs = Input(input_shape)
    k = 32
    #CBRD block
    x = CBRD(inputs, k, dropout)
    x0 = CBRD(x, k, dropout) #ok
    
    x1 = down(x0, 2*k, dropout)
    x2 = down(x1, 4*k, dropout)
    x3 = down(x2, 8*k, dropout)
    y = down(x3, 16*k, dropout)
    
    y1 = up(y,x3, 8*k, dropout) # x4 in paper
    y2 = up(y1,x2, 4*k, dropout)
    y3 = up(y2,x1, 2*k, dropout)
    y4 = up(y3, x0, k, dropout)
    
    out = PFF(y1, y2, y3, y4, k, dropout)
    out = Conv2D(1, 1)(out)
    out = Activation("sigmoid")(out)
    model = Model(inputs=inputs, outputs=out)
    return(model)
    

    
    
    
    
def spatial_attention_CMP(input_feature):
    kernel_size = 7
    channel = input_feature.shape[-1]
    cbam_feature = input_feature
    output_shape = (input_feature.shape[1], input_feature.shape[2], 1)


    avg_pool = Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True), output_shape)(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True), output_shape)(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature.shape[-1] == 1

    return (cbam_feature * input + input)
    
def MSCAF_CBAM(input, out_channels, dropout):
    x1 = Conv2D(filters=out_channels//2, kernel_size=3, padding='same', use_bias=False)(input)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    
    x2 = Conv2D(filters=out_channels//4, kernel_size=3, dilation_rate=3, padding='same', use_bias=False)(input)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)

    x3 = Conv2D(filters=out_channels//4, kernel_size=3, dilation_rate=5, padding='same', use_bias=False)(input)
    x3 = BatchNormalization()(x3)
    x3 = Activation("relu")(x3)
    
    out = Concatenate()([x1, x2, x3])
    out = ACA(out)
    out = spatial_attention_CMP(out)
    out = Conv2D(out_channels, 1, use_bias = False)(out)
    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Dropout(dropout)(out)
    return(out)
    
    

    


    
def build_CMP_UNet_CBAM(input_shape, dropout=0):
    inputs = Input(input_shape)
    k = 32
    #CBRD block
    x = CBRD(inputs, k, dropout)
    x0 = CBRD(x, k, dropout) #ok
    
    x1 = down(x0, 2*k, dropout)
    x2 = down(x1, 4*k, dropout)
    x3 = down(x2, 8*k, dropout)
    y = down(x3, 16*k, dropout)
    
    y1 = up(y,x3, 8*k, dropout) # x4 in paper
    y2 = up(y1,x2, 4*k, dropout)
    y3 = up(y2,x1, 2*k, dropout)
    y4 = up(y3, x0, k, dropout)
    
    out = PFF(y1, y2, y3, y4, k, dropout)
    out = Conv2D(1, 1)(out)
    out = Activation("sigmoid")(out)
    model = Model(inputs=inputs, outputs=out)
    return(model)