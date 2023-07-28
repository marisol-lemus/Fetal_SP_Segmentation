from this import d
from math import gamma
import enum
import sys
from time import process_time_ns
from numpy.core.arrayprint import array2string
import tensorflow as tf
#tf.config.run_functions_eagerly(True)
class Unet_network:
    def __init__(self, input_shape, out_ch, loss='dice_loss', metrics=['dice_coef', 'dis_dice_coef'], style='basic', ite=2, depth=4, dim=32, weights='', init='he_normal',acti='elu',lr=1e-4,optimizer = 'Adam'):
        from tensorflow.keras.layers import Input
        self.style=style
        self.input_shape=input_shape
        self.out_ch=out_ch
        self.metrics = metrics
        self.loss = loss
        self.ite=ite
        self.depth=depth
        self.dim=dim
        self.init=init
        self.acti=acti
        self.weight=weights
        self.lr=lr
        self.I = Input(input_shape)
        self.ratio = None
        self.optimizer = optimizer
        self.b_kernel = None
    def conv_block(self,inp,dim):
        from tensorflow.keras.layers import BatchNormalization as bn, Activation, Conv2D, Dropout
        x = bn()(inp)
        x = Activation(self.acti)(x)
        x = Conv2D(dim, (3,3), padding='same', kernel_initializer=self.init)(x)
        return x

    def conv1_block(self,inp,dim):
        from tensorflow.keras.layers import BatchNormalization as bn, Activation, Conv2D, Dropout
        x = bn()(inp)
        x = Activation(self.acti)(x)
        x = Conv2D(dim, (1,1), padding='same', kernel_initializer=self.init)(x)
        return x

    def tconv_block(self,inp,dim):
        from tensorflow.keras.layers import BatchNormalization as bn, Activation, Conv2DTranspose, Dropout
        x = bn()(inp)
        x = Activation(self.acti)(x)
        x = Conv2DTranspose(dim, 2, strides=2, padding='same', kernel_initializer=self.init)(x)
        return x

    def basic_block(self, inp, dim):
        for i in range(self.ite):
            inp = self.conv_block(inp,dim)
        return inp

    def res_block(self, inp, dim):
        from tensorflow.keras.layers import BatchNormalization as bn, Activation, Conv2D, Dropout, Add
        inp2 = inp
        for i in range(self.ite):
            inp = self.conv_block(inp,dim)
        cb2 = self.conv1_block(inp2,dim)
        return Add()([inp, cb2])

    def dense_block(self, inp, dim):
        from tensorflow.keras.layers import BatchNormalization as bn, Activation, Conv2D, Dropout, concatenate
        for i in range(self.ite):
            cb = self.conv_block(inp, dim)
            inp = concatenate([inp,cb])
        inp = self.conv1_block(inp,dim)
        return inp

    def RCL_block(self, inp, dim):
        from tensorflow.keras.layers import BatchNormalization as bn, Activation, Conv2D, Dropout, Add
        RCL=Conv2D(dim, (3,3), padding='same',kernel_initializer=self.init)
        conv=bn()(inp)
        conv=Activation(self.acti)(conv)
        conv=Conv2D(dim,(3,3),padding='same',kernel_initializer=self.init)(conv)
        conv2=bn()(conv)
        conv2=Activation(self.acti)(conv2)
        conv2=RCL(conv2)
        conv2=Add()([conv,conv2])
        for i in range(0, self.ite-2):
            conv2=bn()(conv2)
            conv2=Activation(self.acti)(conv2)
            conv2=Conv2D(dim, (3,3), padding='same',weights=RCL.get_weights())(conv2)
            conv2=Add()([conv,conv2])
        return conv2

    def build_U(self, inp, dim, depth):
        from tensorflow.keras.layers import MaxPooling2D, concatenate, Dropout
        if depth > 0:
            if self.style == 'basic':
                x = self.basic_block(inp, dim)
            elif self.style == 'res':
                x = self.res_block(inp, dim)
            elif self.style == 'dense':
                x = self.dense_block(inp, dim)
            elif self.style == 'RCL':
                x = self.RCL_block(inp, dim)
            else:
                sys.exit('Available style : basic, res, dense, RCL')
            x2 = MaxPooling2D()(x)
            x2 = self.build_U(x2, int(dim*2), depth-1)
            x2 = self.tconv_block(x2,int(dim*2))
            x2 = concatenate([x,x2])
            if self.style == 'basic':
                x2 = self.basic_block(x2, dim)
            elif self.style == 'res':
                x2 = self.res_block(x2, dim)
            elif self.style == 'dense':
                x2 = self.dense_block(x2, dim)
            elif self.style == 'RCL':
                x2 = self.RCL_block(x2, dim)
            else:
                sys.exit('Available style : basic, res, dense, RCL')
        else:
            if self.style == 'basic':
                x2 = self.basic_block(inp, dim)
            elif self.style == 'res':
                x2 = self.res_block(inp, dim)
            elif self.style == 'dense':
                x2 = self.dense_block(inp, dim)
            elif self.style == 'RCL':
                x2 = self.RCL_block(inp, dim)
            else:
                sys.exit('Available style : basic, res, dense, RCL')
        return x2

    def UNet(self):
        from tensorflow.keras.layers import Conv2D
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.optimizers import SGD
        from keras.losses import CategoricalCrossentropy
        ##Create the first layer layer for the network
        o = self.build_U(self.I, self.dim, self.depth)
        ##Create the activation layer from there
        o = Conv2D(self.out_ch, 1, activation='softmax')(o)
        #Create the model instance
        model = Model(inputs=self.I, outputs=o)
        if  self.optimizer == 'Adam':
            #Define the compilation options based on the metrics
            print('Using Adam Optimizer')
            if len(self.metrics)==2:
                print('here metric 2')
                model.compile(optimizer=Adam(learning_rate=self.lr,clipnorm= 1), loss=getattr(self, self.loss), metrics=[getattr(self, self.metrics[0]),getattr(self, self.metrics[1])])

            else:
                print('here metric 1')
                model.compile(optimizer=Adam(learning_rate=self.lr,clipnorm = 1), loss=getattr(self, self.loss), metrics=[getattr(self, self.metrics[0])])

            #If the weights was defined, load them
            if self.weight:
                model.load_weights(self.weight)
        elif self.optimizer == 'SGD':
            print('Using SGD Optimizer')
            #Define the compilation options based on the metrics
            if len(self.metrics)==2:
                model.compile(optimizer=SGD(learning_rate=self.lr,clipnorm = 1), loss=getattr(self, self.loss), metrics=[getattr(self, self.metrics[0]),getattr(self, self.metrics[1])])
                print('here metric 2')
                #model.compile(optimizer=SGD(learning_rate=self.lr,clipnorm = 1), loss=focal_tversky_loss(), metrics=[getattr(self, self.metrics[0]),getattr(self, self.metrics[1])])
            else:
            #    model.compile(optimizer=SGD(learning_rate=self.lr,clipnorm = 1), loss=getattr(self, self.loss), metrics=[getattr(self, self.metrics[0])])
                print('here metric 1')
                model.compile(optimizer=SGD(learning_rate=self.lr,clipnorm = 1), loss=getattr(self, self.loss), metrics=[getattr(self, self.metrics[0])])
            #    model.compile(optimizer=SGD(learning_rate=self.lr,clipnorm = 1), loss=focal_tversky_loss(), metrics=[getattr(self, self.metrics[0])])
            
            #If the weights was defined, load them
            if self.weight:
                model.load_weights(self.weight)
        elif self.optimizer == 'RAdam':
            from keras_radam import RAdam
            print('Using RAdam Optimizer')
            #Define the compilation options based on the metrics
            if len(self.metrics)==2:
                #model.compile(optimizer=RAdam(learning_rate=self.lr,clipnorm = 1), loss=getattr(self, self.loss), metrics=[getattr(self, self.metrics[0]),getattr(self, self.metrics[1])])
                model.compile(optimizer=RAdam(learning_rate=self.lr,clipnorm = 1), loss=getattr(self, self.loss), metrics=[getattr(self, self.metrics[0]),getattr(self, self.metrics[1])])
            else:
                #model.compile(optimizer=RAdam(learning_rate=self.lr,clipnorm = 1), loss=getattr(self, self.loss), metrics=[getattr(self, self.metrics[0])])
                model.compile(optimizer=RAdam(learning_rate=self.lr,clipnorm = 1), loss=getattr(self, self.loss), metrics=[getattr(self, self.metrics[0])])

            #If the weights was defined, load them
            if self.weight:
                model.load_weights(self.weight)
        #Throw error
        else:
            sys.exit('Error, Available modes : Adam, SGD')
        return model

    def build(self):
        return self.UNet()

    def dice_coef(self, y_true, y_pred):
        from tensorflow.keras import backend as K
        smooth = 0.001
        intersection = K.sum(y_true * K.round(y_pred), axis=[1,2])
        union = K.sum(y_true, axis=[1,2]) + K.sum(K.round(y_pred), axis=[1,2])
        dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
        return K.mean(dice[1:])

    def SSIM_loss(self,y_true, y_pred):
        return 1-tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

    def dice_loss(self, y_true, y_pred):
        from tensorflow.keras import backend as K
        smooth = 0.001
        intersection = K.sum(y_true * y_pred, axis=[1,2])
        union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
        dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
        return K.mean(K.pow(-K.log(dice[1:]),0.3))

    def ori_dice_loss(self, y_true, y_pred):
        from tensorflow.keras import backend as K
        smooth = 0.001
        intersection = K.sum(y_true * y_pred, axis=[1,2])
        union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
        dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
        return -K.mean(dice[1:])

    def dis_dice_loss(self, y_true, y_pred):
        import cv2, numpy as np
        from tensorflow.keras import backend as K
        si=K.int_shape(y_pred)[-1]
        riter=3
        smooth = 0.00001
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(riter*2+1,riter*2+1))
        kernel=kernel/(np.sum(kernel))
        kernel=np.repeat(kernel[:,:,np.newaxis],si,axis=-1)
        if self.b_kernel is None:
            self.b_kernel = tf.compat.v1.Variable(kernel[:,:,:,np.newaxis],dtype=tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        #kernel=K.variable(kernel[:,:,:,np.newaxis])
        y_true_s=tf.nn.depthwise_conv2d(y_true,self.b_kernel,strides = [1,1,1,1],data_format="NHWC",padding="SAME")
        y_pred_s=tf.nn.depthwise_conv2d(y_pred,self.b_kernel,strides = [1,1,1,1],data_format="NHWC",padding="SAME")
        y_true_s = y_true_s > 0.8
        y_pred_s = y_pred_s > 0.8
        y_true_s = y_true - K.cast(y_true_s,'float32')
        y_pred_s = y_pred - K.cast(y_pred_s,'float32')
        intersection = K.sum(y_true_s * y_pred_s, axis=[1,2])
        union = K.sum(y_true_s, axis=[1,2]) + K.sum(y_pred_s, axis=[1,2])
        dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
        return K.sum(K.pow(-K.log(dice[1:]),0.3))
    
    def dis_dice_coef(self, y_true, y_pred):
        import cv2, numpy as np
        from tensorflow.keras import backend as K
        si=K.int_shape(y_pred)[-1]
        riter=3
        smooth = 0.001
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(riter*2+1,riter*2+1))
        kernel=kernel/(np.sum(kernel))
        kernel=np.repeat(kernel[:,:,np.newaxis],si,axis=-1)
        if self.b_kernel is None:
            self.b_kernel = tf.compat.v1.Variable(kernel[:,:,:,np.newaxis],dtype=tf.float32)
        #tf.cast(y_true, tf.float64)
        #kernel=K.variable(kernel[:,:,:,np.newaxis])
        y_true_s=tf.nn.depthwise_conv2d(y_true,self.b_kernel,strides = [1,1,1,1],data_format="NHWC",padding="SAME")
        y_pred_s=tf.nn.depthwise_conv2d(y_pred,self.b_kernel,strides = [1,1,1,1],data_format="NHWC",padding="SAME")
        y_true_s = y_true_s > 0.8
        y_pred_s = y_pred_s > 0.8
        y_true_s = y_true - K.cast(y_true_s,'float32')
        y_pred_s = y_pred - K.cast(y_pred_s,'float32')
        intersection = K.sum(y_true_s * y_pred_s, axis=[1,2])
        union = K.sum(y_true_s, axis=[1,2]) + K.sum(y_pred_s, axis=[1,2])
        dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
        return K.mean(dice[1:])

    def hd_loss(self, y_true, y_pred):
        from tensorflow.keras import backend as K
        import cv2, numpy as np
        def in_func(in_tensor,in_kernel,in_f):
            return K.clip(K.depthwise_conv2d(in_tensor,in_kernel,data_format="channels_last",padding="same")-0.5,0,0.5)*in_f
        si=K.int_shape(y_pred)[-1]
        f_qp=K.square(y_true-y_pred)*y_pred
        f_pq=K.square(y_true-y_pred)*y_true
        p_b=K.cast(y_true,'float32')
        p_bc=1-p_b
        q_b=K.cast(y_pred>0.5,'float32')
        q_bc=1-q_b
        rtiter=0
        si=K.int_shape(y_pred)[-1]
        for riter in range(3,19,3):
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(riter*2+1,riter*2+1))
            kernel=kernel/(np.sum(kernel))
            kernel=np.repeat(kernel[:,:,np.newaxis],si,axis=-1)
            kernel=K.variable(kernel[:,:,:,np.newaxis])
            if rtiter == 0:
                loss=K.mean(in_func(p_bc,kernel,f_qp)+in_func(p_b,kernel,f_pq)+in_func(q_bc,kernel,f_pq)+in_func(q_b,kernel,f_qp),axis=0)
            else:
                loss=loss+K.mean(in_func(p_bc,kernel,f_qp)+in_func(p_b,kernel,f_pq)+in_func(q_bc,kernel,f_pq)+in_func(q_b,kernel,f_qp),axis=0)
        return K.mean(loss[1:])

    def asymmetric_focal_tversky_loss(self,y_true, y_pred):
        from loss_functions import asymmetric_focal_tversky_loss
        return asymmetric_focal_tversky_loss(gamma=0.75)(y_true=y_true,y_pred=y_pred)

    def focal_loss(self,y_true, y_pred):
        from loss_functions import focal_loss
        return focal_loss(gamma_f=0.75,alpha=0.25)(y_true=y_true,y_pred=y_pred)

    def focal_loss_2(self,y_true, y_pred):
        from loss_functions import focal_loss_2
        return focal_loss_2(gamma_f=0.0)(y_true=y_true,y_pred=y_pred)

    def focal_loss_3(self,y_true,y_pred):
        from loss_functions import focal_loss_3
        return focal_loss_3(gamma=2.0,alpha=1)(y_true=y_true,y_pred=y_pred)


    def focal_tversky_loss(self,y_true, y_pred):
        from loss_functions import focal_tversky_loss
        return focal_tversky_loss()(y_true=y_true,y_pred=y_pred)

    def hyb_disdice_focal_loss(self, y_true, y_pred):
        focal_loss = self.focal_tversky_loss(y_true,y_pred)
        h_loss=self.dis_dice_loss(y_true, y_pred)
        return 0.1*h_loss + tf.cast(focal_loss,dtype=tf.float32)

    def hyb_ssim_focal_tversky_loss(self, y_true, y_pred):
        focal_loss = self.focal_tversky_loss(y_true,y_pred)
        h_loss=self.SSIM_loss(y_true, y_pred)
        return 0.1*h_loss + tf.cast(focal_loss,dtype=tf.float32)

    def hyb_loss(self, y_true, y_pred):
        d_loss=self.dice_loss(y_true, y_pred)
        h_loss=self.dis_dice_loss(y_true, y_pred)
        return 0.1*h_loss + d_loss

    def hyb_loss2(self, y_true, y_pred):
        d_loss=self.dice_loss(y_true, y_pred)
        h_loss=self.hd_loss(y_true, y_pred)
        if self.ratio==None:
            loss = h_loss + d_loss
        else:
            loss = self.ratio * h_loss + (1-self.ratio) * d_loss
        self.ratio = d_loss / h_loss
        return loss

def reset_gpu():
    r"""
    Reset the gpu memory so that cached memory can be cleared.
    """
    from tensorflow.keras import backend as K
    import tensorflow as tf
    K.clear_session()
    tf.compat.v1.reset_default_graph()

def set_gpu(gpu_num=0):
    r"""
    Set the gpu to be used when training/predicting.
    
    Parameters
    ----------
    gpu_num : Define the gpu to be set (other gpus are ignored) when training and predicting.
    
    Returns
    -------
    None
    """    

    import tensorflow as tf
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_num)
    if str(gpu_num) != '-1':
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus is not None:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.experimental.set_virtual_device_configuration(gpus[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20087)])
            print("setting gpu: "+str(gpus[0]))
        else:
            print("No GPU device found")

def fliper(array,dim,out_ch, f=0):
    import numpy as np
    if dim == 'axi':
        array = array[:,:,::-1,:]
    elif dim == 'cor':
        array = array[:,::-1,:,:]
    if f:
        if out_ch == 7:
            array2 = np.concatenate((array[:,:,:,0,np.newaxis],array[:,:,:,2,np.newaxis],array[:,:,:,1,np.newaxis],
                                    array[:,:,:,4,np.newaxis],array[:,:,:,3,np.newaxis],
                                    array[:,:,:,6,np.newaxis],array[:,:,:,5,np.newaxis]),axis=-1)
        elif out_ch==5:
            array2 = np.concatenate((array[:,:,:,0,np.newaxis],array[:,:,:,2,np.newaxis],array[:,:,:,1,np.newaxis],
                                 array[:,:,:,4,np.newaxis],array[:,:,:,3,np.newaxis]),axis=-1)
        elif out_ch==3:
            array2 = np.concatenate((array[:,:,:,0,np.newaxis],array[:,:,:,2,np.newaxis],array[:,:,:,1,np.newaxis]),axis=-1)
        elif out_ch==2:
            array2 = np.concatenate((array[:,:,:,0,np.newaxis],array[:,:,:,1,np.newaxis]),axis=-1)
             
        return array2
    return array

def sagfliper(array,out_ch):
    import numpy as np
    if out_ch==4:
        array2 = np.stack((array[...,0], array[...,1], array[...,1],
                            array[...,2], array[...,2], array[...,3],
                            array[...,3]),axis=-1)
    elif out_ch==3:
        array2 = np.stack((array[...,0], array[...,1], array[...,1],
                        array[...,2], array[...,2]),axis=-1)
    elif out_ch==2:
        array2 = np.stack((array[...,0], array[...,1], array[...,1]),axis=-1)          
    return array2

def crop_pad_ND(img, target_shape):
    r"""
    Resize an image based on target shape
    
    Parameters
    ----------
    target_shape : the shape to be resized to.
    
    Returns
    -------
    image : The resized image.
    """ 
    import operator, numpy as np
    if (img.shape > np.array(target_shape)).any():
        target_shape2 = np.min([target_shape, img.shape],axis=0)
        start = tuple(map(lambda a, da: a//2-da//2, img.shape, target_shape2))
        end = tuple(map(operator.add, start, target_shape2))
        slices = tuple(map(slice, start, end))
        img = img[tuple(slices)]
    offset = tuple(map(lambda a, da: a//2-da//2, target_shape, img.shape))
    slices = [slice(offset[dim], offset[dim] + img.shape[dim]) for dim in range(img.ndim)]
    result = np.zeros(target_shape)
    result[tuple(slices)] = img
    return result

def get_maxshape(img_list):
    r"""
    Get the maxshape from the list you gave, there are cases where you give different input sizes, 
    so having the biggest is necessary for resizing 
    
    Parameters
    ----------
    img_list : The list of paths to the reconstructed MRIs.
    
    Returns
    -------
    arrmax : An array of three elements with the biggest dimensions for each anatomical view.
    """
    import nibabel as nib
    arrmax = [-1,-1,-1]
    for i in range(0,len(img_list)):
        img = nib.load(img_list[i])
        data = img.shape
        #print(data)
        if data[0]>arrmax[0]:
            arrmax[0]=data[0]
        if data[1]>arrmax[1]:
            arrmax[1]=data[1]
        if data[2]>arrmax[2]:
            arrmax[2]=data[2]
    return arrmax

def make_dic(img_list, gold_list, input_size, labels_dict, view, flip = False):
    r"""
    Function that creates the dictionary for the segmentation and reconstructions, they will be used for training/predicting.  
    
    Parameters
    ----------
    img_list : The list of paths to the reconstructed MRIs.
    gold_list : The list of paths to the segmentation list MRIs.
    input_size : The input size for training the network, using for resizing purposes, ej [192,192]
    dim : The anatomical axis, ['axi','cor','sag']
    max_shape : The max shape of the given dataset.
    labels_dict : The dictionary that maps from a label to its name.
    shape: The dimension for each axis, i.e: how many labels are used for training/predicting, default = [7,7,4]; axi,cor,sag respectively 
    flip: Flip the image for training, this only should be set to true if you want that for training, not for predicting. default = False

    Returns
    -------
    arrmax : An array of three elements with the biggest dimensions for each anatomical view.
    """

    import numpy as np
    import nibabel as nib
    from tqdm import tqdm

    def get_data(img, label):
        r"""
        Return resized image and label, data is also standarized from 0 to 1 for the intensity.
        
        Parameters
        ----------
        img : The path to the reconstruction currently being processed.
        label : The path to the segmentation currently being processed.
        dim : The anatomical axis, ['axi','cor','sag']

        Returns
        -------
        img, label : A touple having the np.arrays from the current image and label.
        """
        import nibabel as nib
        import matplotlib.pyplot as plt

        img_data = np.squeeze(nib.load(img).get_fdata())
        assert not np.isnan(img_data).any()
        img = img_data.astype('float16')
        img = ((img - float(img.min())) / float(img.max() +1E-7))*float(255)
        label = np.squeeze(nib.load(label).get_fdata())
        if view.string == 'axi':
            input_size_temp = [*input_size, view.max_slices]
        elif view.string == 'cor':
            input_size_temp = [input_size[0], view.max_slices, input_size[1]]
        elif view.string == 'sag':
            input_size_temp = [view.max_slices, *input_size]
        else:
            sys.exit('available: axi, cor, sag.   Your: '+view.string)

        img = crop_pad_ND(img, input_size_temp)
        label = crop_pad_ND(label, input_size_temp)
        return img, label
    
    def make_seg(seg,labels_dict,i,img2):
        r"""
        Create the segmentation dictionary, given the axis, the mask and the labels.
        
        Parameters
        ----------
        label_num : dimension of the label
        labels_dict : Dictionary mapping the labels to their names
        dim_num : Mapping from axis to number. Axi: 2, Cor: 1, sag: 0
        i : Current image being processed/added to the dictionary.
        seg : np.array to be returned 
        img2 : Current labeled image
        sagital : Flag if the sag view is being processed, default =False
        Returns
        -------
        seg : Segmented dictionary but with the added volume
        """

        mask = np.zeros_like(img2)
        for ilabel,(_,value) in enumerate(labels_dict.items()):
            label_loc = np.where((img2>value-0.25)&(img2<value+0.25))
            mask[label_loc]=1
            if view.string == 'sag':
                if ilabel==0:
                    seg[view.max_slices*i:view.max_slices*(i+1),:,:,ilabel] = mask
                    mask[:]=0
                else:
                    if ilabel%2==0:
                        seg[view.max_slices*i:view.max_slices*(i+1),:,:,int(ilabel/2)] = mask
                        mask[:]=0
            else:
                seg[view.max_slices*i:view.max_slices*(i+1),:,:,ilabel] = mask
                mask[:]=0

    def view_dic(view_num):
        r"""
        Return resized image and label, data is also standarized from 0 to 1 for the intensity.
        
        Parameters
        ----------
        view_num : is the dimension index, axi=2, cor=1, sag=0

        Returns
        -------
        dic, label : A tuple having the np.arrays from the dictionary and labels.
        """
        dic = np.zeros([view.max_slices*len(img_list), input_size[0], input_size[1], 1], dtype=np.float16)
        seg = np.zeros([view.max_slices*len(img_list), input_size[0], input_size[1], view.out_ch], dtype=np.float16)
        for i in tqdm(range(0, len(img_list)),desc=view.string+' dic making..'):
            img, label = get_data(img_list[i], gold_list[i])
            if view.string == 'sag':    
                dic[view.max_slices*i:view.max_slices*(i+1),:,:,0]= img
                img2 = label
            else:
                dic[view.max_slices*i:view.max_slices*(i+1),:,:,0]= np.swapaxes(img,view_num,0)
                img2 = np.swapaxes(label,view_num,0)
            make_seg(seg,labels_dict=labels_dict,i=i,img2=img2)
        if flip:
            if view.string == 'sag':
                dic = np.concatenate((dic, dic[:,:,::-1,:], dic[:,::-1,:,:]),axis=0)
                print("dictionary shape: " + str(dic.shape))
                seg = np.concatenate((seg, seg[:,:,::-1,:], seg[:,::-1,:,:]),axis=0)
            elif view.string == 'cor':
                dic=np.concatenate((dic,dic[:,:,::-1,:]),axis=0)
                seg=np.concatenate((seg,seg[:,:,::-1,:]),axis=0)
                dic=np.concatenate((dic, fliper(dic,dim = view.string,out_ch= view.out_ch)),axis=0)
                seg=np.concatenate((seg, fliper(seg, f = 1, dim = view.string,out_ch= view.out_ch)),axis=0)                
            else:
                dic=np.concatenate((dic,dic[:,::-1,:,:]),axis=0)
                seg=np.concatenate((seg,seg[:,::-1,:,:]),axis=0)
                dic=np.concatenate((dic, fliper(dic,dim = view.string,out_ch= view.out_ch)),axis=0)
                seg=np.concatenate((seg, fliper(seg, f = 1, dim = view.string,out_ch= view.out_ch)),axis=0)                
        return dic,seg

    if view.string == 'axi':
        dic,seg = view_dic(view_num=2)
    elif view.string == 'cor':
        dic,seg = view_dic(view_num=1)
    elif view.string == 'sag':
        dic,seg = view_dic(view_num=0)
    else:
        sys.exit('available: axi, cor, sag.   Your: '+view.string)
    print("dictionary shape: " + str(dic.shape))
    print("segmentation shape: "+str(seg.shape))
    return dic, seg    

def make_verify(input_name, result_loc):
    r"""
    Create a png file overlaying the segmentation and the reconstruction
    
    Parameters
    ----------
    input_name : Is the path to the reconstruction
    result_loc : The path to save the png file.
    
    """
    import numpy as np
    import nibabel as nib
    import matplotlib.pyplot as plt
    import sys

    img = nib.load(input_name).get_fdata()
    label_name = input_name.split('/')[-1:][0].split('.nii')[0]
    label_name = label_name+'_deep_sum.nii.gz'
    label = nib.load(label_name).get_fdata()

    f,axarr = plt.subplots(3,3,figsize=(9,9))
    f.patch.set_facecolor('k')

    f.text(0.4, 0.95, label_name, size="large", color="White")

    axarr[0,0].imshow(np.rot90(img[:,:,np.int(img.shape[-1]*0.4)]),cmap='gray')
    axarr[0,0].imshow(np.rot90(label[:,:,np.int(label.shape[-1]*0.4)]),alpha=0.3, cmap='gnuplot2')
    axarr[0,0].axis('off')

    axarr[0,1].imshow(np.rot90(img[:,:,np.int(img.shape[-1]*0.5)]),cmap='gray')
    axarr[0,1].imshow(np.rot90(label[:,:,np.int(label.shape[-1]*0.5)]),alpha=0.3, cmap='gnuplot2')
    axarr[0,1].axis('off')

    axarr[0,2].imshow(np.rot90(img[:,:,np.int(img.shape[-1]*0.6)]),cmap='gray')
    axarr[0,2].imshow(np.rot90(label[:,:,np.int(label.shape[-1]*0.6)]),alpha=0.3, cmap='gnuplot2')
    axarr[0,2].axis('off')

    axarr[1,0].imshow(np.rot90(img[:,np.int(img.shape[-2]*0.4),:]),cmap='gray')
    axarr[1,0].imshow(np.rot90(label[:,np.int(label.shape[-2]*0.4),:]),alpha=0.3, cmap='gnuplot2')
    axarr[1,0].axis('off')

    axarr[1,1].imshow(np.rot90(img[:,np.int(img.shape[-2]*0.5),:]),cmap='gray')
    axarr[1,1].imshow(np.rot90(label[:,np.int(label.shape[-2]*0.5),:]),alpha=0.3, cmap='gnuplot2')
    axarr[1,1].axis('off')

    axarr[1,2].imshow(np.rot90(img[:,np.int(img.shape[-2]*0.6),:]),cmap='gray')
    axarr[1,2].imshow(np.rot90(label[:,np.int(label.shape[-2]*0.6),:]),alpha=0.3, cmap='gnuplot2')
    axarr[1,2].axis('off')

    axarr[2,0].imshow(np.rot90(img[np.int(img.shape[0]*0.4),:,:]),cmap='gray')
    axarr[2,0].imshow(np.rot90(label[np.int(label.shape[0]*0.4),:,:]),alpha=0.3, cmap='gnuplot2')
    axarr[2,0].axis('off')

    axarr[2,1].imshow(np.rot90(img[np.int(img.shape[0]*0.5),:,:]),cmap='gray')
    axarr[2,1].imshow(np.rot90(label[np.int(label.shape[0]*0.5),:,:]),alpha=0.3, cmap='gnuplot2')
    axarr[2,1].axis('off')

    axarr[2,2].imshow(np.rot90(img[np.int(img.shape[0]*0.6),:,:]),cmap='gray')
    axarr[2,2].imshow(np.rot90(label[np.int(label.shape[0]*0.6),:,:]),alpha=0.3, cmap='gnuplot2')
    axarr[2,2].axis('off')
    f.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(result_loc+'/'+label_name.split('/')[-1].split('.nii')[0]+'_verify.png', facecolor=f.get_facecolor())
    return 0

def make_verify_window(input_name, parent_dir,window_center,window_width):
    r"""
    Create a png file overlaying the segmentation and the reconstruction
    
    Parameters
    ----------
    input_name : Is the path to the reconstruction
    parent_dir : The path to save the png file.
    window_center : The center for window the image in range
    window_width : The value 

    """

    import numpy as np
    import nibabel as nib
    import matplotlib.pyplot as plt
    import sys
    def window_image(image,window_center, window_width):
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        window_image = image.copy()
        window_image[window_image < img_min] = img_min
        window_image[window_image > img_max] = img_max
        return window_image

    img = nib.load(input_name).get_fdata()
    f,axarr = plt.subplots(3,3,figsize=(9,9))
    f.patch.set_facecolor('k')

    f.text(0.4, 0.95, input_name.split('/')[-1].split('.nii')[0], size="large", color="White")
    img = window_image(img,window_center,window_width)
    axarr[0,0].imshow(np.rot90(img[:,:,np.int(img.shape[-1]*0.4)]),cmap='gray')
    axarr[0,0].axis('off')

    axarr[0,1].imshow(np.rot90(img[:,:,np.int(img.shape[-1]*0.5)]),cmap='gray')
    axarr[0,1].axis('off')

    axarr[0,2].imshow(np.rot90(img[:,:,np.int(img.shape[-1]*0.6)]),cmap='gray')
    axarr[0,2].axis('off')

    axarr[1,0].imshow(np.rot90(img[:,np.int(img.shape[-2]*0.4),:]),cmap='gray')
    axarr[1,0].axis('off')

    axarr[1,1].imshow(np.rot90(img[:,np.int(img.shape[-2]*0.5),:]),cmap='gray')
    axarr[1,1].axis('off')

    axarr[1,2].imshow(np.rot90(img[:,np.int(img.shape[-2]*0.6),:]),cmap='gray')
    axarr[1,2].axis('off')

    axarr[2,0].imshow(np.rot90(img[np.int(img.shape[0]*0.4),:,:]),cmap='gray')
    axarr[2,0].axis('off')

    axarr[2,1].imshow(np.rot90(img[np.int(img.shape[0]*0.5),:,:]),cmap='gray')
    axarr[2,1].axis('off')

    axarr[2,2].imshow(np.rot90(img[np.int(img.shape[0]*0.6),:,:]),cmap='gray')
    axarr[2,2].axis('off')

    f.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(parent_dir+'/'+input_name.split('/')[-1].split('.nii')[0]+'_MR_verify.png', facecolor=f.get_facecolor())
    return 0

def make_verify_experiment(input_name, label_name,parent_dir):
    import numpy as np
    import nibabel as nib
    import matplotlib.pyplot as plt
    import sys
    def window_image(image,window_center, window_width):
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        window_image = image.copy()
        window_image[window_image < img_min] = img_min
        window_image[window_image > img_max] = img_max
        return window_image

    img = nib.load(input_name).get_fdata()
    label = nib.load(label_name).get_fdata()
    print(img.shape)
    print(label.shape)
    label[label == 0] = np.nan
    f,axarr = plt.subplots(3,3,figsize=(9,9))
    f.patch.set_facecolor('k')

    f.text(0.4, 0.95, label_name.split('/')[-1].split('.nii')[0], size="large", color="White")

    axarr[0,0].imshow(np.rot90(img[:,:,np.int(img.shape[-1]*0.4)]),cmap='gray')
    axarr[0,0].imshow(np.rot90(label[:,:,np.int(label.shape[-1]*0.4)]),alpha=0.3, cmap='Accent')
    axarr[0,0].axis('off')

    axarr[0,1].imshow(np.rot90(img[:,:,np.int(img.shape[-1]*0.5)]),cmap='gray')
    axarr[0,1].imshow(np.rot90(label[:,:,np.int(label.shape[-1]*0.5)]),alpha=0.3, cmap='Accent')
    axarr[0,1].axis('off')

    axarr[0,2].imshow(np.rot90(img[:,:,np.int(img.shape[-1]*0.6)]),cmap='gray')
    axarr[0,2].imshow(np.rot90(label[:,:,np.int(label.shape[-1]*0.6)]),alpha=0.3, cmap='Accent')
    axarr[0,2].axis('off')

    axarr[1,0].imshow(np.rot90(img[:,np.int(img.shape[-2]*0.4),:]),cmap='gray')
    axarr[1,0].imshow(np.rot90(label[:,np.int(label.shape[-2]*0.4),:]),alpha=0.3, cmap='Accent')
    axarr[1,0].axis('off')

    axarr[1,1].imshow(np.rot90(img[:,np.int(img.shape[-2]*0.5),:]),cmap='gray')
    axarr[1,1].imshow(np.rot90(label[:,np.int(label.shape[-2]*0.5),:]),alpha=0.3, cmap='Accent')
    axarr[1,1].axis('off')

    axarr[1,2].imshow(np.rot90(img[:,np.int(img.shape[-2]*0.6),:]),cmap='gray')
    axarr[1,2].imshow(np.rot90(label[:,np.int(label.shape[-2]*0.6),:]),alpha=0.3, cmap='Accent')
    axarr[1,2].axis('off')

    axarr[2,0].imshow(np.rot90(img[np.int(img.shape[0]*0.4),:,:]),cmap='gray')
    axarr[2,0].imshow(np.rot90(label[np.int(label.shape[0]*0.4),:,:]),alpha=0.3, cmap='Accent')
    axarr[2,0].axis('off')

    axarr[2,1].imshow(np.rot90(img[np.int(img.shape[0]*0.5),:,:]),cmap='gray')
    axarr[2,1].imshow(np.rot90(label[np.int(label.shape[0]*0.5),:,:]),alpha=0.3, cmap='Accent')
    axarr[2,1].axis('off')

    axarr[2,2].imshow(np.rot90(img[np.int(img.shape[0]*0.6),:,:]),cmap='gray')
    axarr[2,2].imshow(np.rot90(label[np.int(label.shape[0]*0.6),:,:]),alpha=0.3, cmap='Accent')
    axarr[2,2].axis('off')
    f.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(parent_dir+'/'+label_name.split('/')[-1].split('.nii')[0]+'_verify.png', facecolor=f.get_facecolor())
    return 0

def savepngs(nparray,start,end,step,axis,image,dim):
    import numpy as np
    import nibabel as nib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import sys
    from matplotlib import colors

    cmap = colors.ListedColormap(['blue', 'red','purple','#1F45FC','green','yellow'])
    bounds=[0,0.9,2,5,43,161,162]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    for i in range(start,end,step):
        plt.imsave('trash/'+str(dim)+str(image)+'_'+str(axis)+'_'+str(i)+'.png', nparray[i,:,:,axis],cmap='gray')
        

def make_result(output, img_list, result_loc, axis, ext=''):
    import nibabel as nib
    import numpy as np, ipdb

    if type(img_list) != np.ndarray: 
        sys.exit('\'img_list\' must be list')
    for i2 in range(len(img_list)):
        print('filename : '+img_list[i2])
        img = nib.load(img_list[i2])
        img_data = np.squeeze(img.get_fdata())
        pr4=output[i2*(np.int(output.shape[0]/len(img_list))):(i2+1)*(np.int(output.shape[0]/len(img_list)))]
        if axis == 'axi':
            pr4=np.swapaxes(np.argmax(pr4,axis=3).astype(np.int),0,2)
            pr4=crop_pad_ND(pr4, img.shape)
        elif axis == 'cor':
            pr4=np.swapaxes(np.argmax(pr4,axis=3).astype(np.int),0,1)
            pr4=crop_pad_ND(pr4, img.shape)
        elif axis == 'sag':
            pr4=np.argmax(pr4,axis=3).astype(np.int)
            pr4=crop_pad_ND(pr4, img.shape)
        else:
            sys.exit('available: axi, cor, sag.   Your: '+axis)

        img_data[:] = 0
        img_data=pr4
        new_img = nib.Nifti1Image(img_data, img.affine, img.header)
        filename=img_list[i2].split('/')[-1:][0].split('.nii')[0]
        if axis== 'axi':
            filename=filename+'_deep_axi'+ext+'.nii.gz'
        elif axis== 'cor':
            filename=filename+'_deep_cor'+ext+'.nii.gz'
        elif axis== 'sag':
            filename=filename+'_deep_sag'+ext+'.nii.gz'
        else:
            sys.exit('available: axi, cor, sag.   Your: '+axis)
        print('save result : '+result_loc+filename)
        nib.save(new_img, result_loc+str(filename))

    return 1

def return_shape(test_list, predic_array, dim,shape = [7,7,4]):
    import nibabel as nib
    import numpy as np

    img = nib.load(test_list[0])
    img_data = np.squeeze(img.get_fdata())
    output = np.zeros([len(test_list),*img_data.shape,shape[0]])

    for i2 in range(len(test_list)):
        predic=predic_array[i2*(np.int(predic_array.shape[0]/len(test_list))):(i2+1)*(np.int(predic_array.shape[0]/len(test_list)))]
        if dim == 'axi':
            predic=np.swapaxes(predic,0,2)
            predic=crop_pad_ND(predic, (*img.shape,predic.shape[-1]))
        elif dim == 'cor':
            predic=np.swapaxes(predic,0,1)
            predic=crop_pad_ND(predic, (*img.shape,predic.shape[-1]))
        elif dim == 'sag':
            predic=crop_pad_ND(predic, (*img.shape,predic.shape[-1]))
        else:
            sys.exit('available: axi, cor, sag.   Your: '+dim)
        output[i2]=predic
    return output

def return_shape_single(test_list, predic_array, dim,shape = [7,7,4]):
    import nibabel as nib
    import numpy as np

    img = nib.load(test_list)
    img_data = np.squeeze(img.get_fdata())
    output = np.zeros([1,*img_data.shape,shape[0]])

    for i2 in range(1):
        predic=predic_array[i2*(np.int(predic_array.shape[0]/1)):(i2+1)*(np.int(predic_array.shape[0]/1))]
        if dim == 'axi':
            predic=np.swapaxes(predic,0,2)
            predic=crop_pad_ND(predic, (*img.shape,predic.shape[-1]))
        elif dim == 'cor':
            predic=np.swapaxes(predic,0,1)
            predic=crop_pad_ND(predic, (*img.shape,predic.shape[-1]))
        elif dim == 'sag':
            predic=crop_pad_ND(predic, (*img.shape,predic.shape[-1]))
        else:
            sys.exit('available: axi, cor, sag.   Your: '+dim)
        output[i2]=predic
    return output

def argmax_sum(test_list, result_loc, ext='', *args):
    import numpy as np
    import nibabel as nib
    total = np.zeros(args[0].shape)
    temp = np.zeros(total.shape[1:])
    for item in args:
        if item.shape[0] != len(test_list):
            sys.exit('Error; Lenght mismatch error: args length: %d, test_list length: %d ' % (item.shape[0], len(test_list)))
        for i in range(0,item.shape[0]):
            result_argmax = np.argmax(item[i],axis=-1)
            for i2 in range(0,item[i].shape[-1]):
                loc = np.where(result_argmax==i2)
                temp[...,i2][loc]=1
            total[i] = total[i] + temp
            temp[:]=0

    for i in range(0,len(test_list)):
        img = nib.load(test_list[i])
        new_img = nib.Nifti1Image(np.argmax(total[i],axis=-1), img.affine, img.header)
        filename=test_list[i].split('/')[-1:][0].split('.nii')[0]
        filename=filename+'_deep_argmax.nii.gz'
        #nib.save(new_img, result_loc+str(filename))

def relabel(inputfile,inlabel,outlabel):
    import nibabel as nib
    import os
    import numpy as np
    img=nib.load(inputfile)
    data = np.squeeze(img.get_fdata())
    ori_label = np.array(inlabel)
    relabel = np.array(outlabel)
    for itr in range(len(ori_label)):
        loc = np.where((data>ori_label[itr]-0.5)&(data<ori_label[itr]+0.5))
        data[loc]=relabel[itr]

    os.remove(inputfile)
    new_img = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(new_img, inputfile)

def argmax_sum_single(test_list, filename,result_loc, ext='', *args):
    import numpy as np
    import nibabel as nib
    total = np.zeros(args[0].shape)
    temp = np.zeros(total.shape[1:])
    for item in args:
        if item.shape[0] != 1:
            sys.exit('Error; Lenght mismatch error: args length: %d, test_list length: %d ' % (item.shape[0], len(test_list)))
        for i in range(0,item.shape[0]):
            result_argmax = np.argmax(item[i],axis=-1)
            for i2 in range(0,item[i].shape[-1]):
                loc = np.where(result_argmax==i2)
                temp[...,i2][loc]=1
            total[i] = total[i] + temp
            temp[:]=0

    img = nib.load(test_list)
    new_img = nib.Nifti1Image(np.argmax(total[0],axis=-1), img.affine, img.header)
    import os
    parent = os.path.dirname(test_list)
    short_name=test_list.split('/')[-1:][0].split('.nii')[0]
    short_name = short_name+'_deep_prob_argmax.nii'
    filename=filename+short_name
    #nib.save(new_img, args.result_loc+str(filename))
    nib.save(new_img, parent+'/'+str(short_name))
  
def make_sum(axi_filter, cor_filter, sag_filter, input_name, result_loc):
    import nibabel as nib
    import numpy as np
    import sys, glob

    # 1-->axi 2-->cor 3-->sag
    axi_list = sorted(glob.glob(axi_filter))
    cor_list = sorted(glob.glob(cor_filter))
    sag_list = sorted(glob.glob(sag_filter))
    axi = nib.load(axi_list[0])
    cor = nib.load(cor_list[0])
    sag = nib.load(sag_list[0])

    bak = np.zeros(np.shape(axi.get_fdata()))
    left_in = np.zeros(np.shape(axi.get_fdata()))
    right_in = np.zeros(np.shape(axi.get_fdata()))
    left_subplate = np.zeros(np.shape(axi.get_fdata()))
    right_subplate = np.zeros(np.shape(axi.get_fdata()))
    left_plate = np.zeros(np.shape(axi.get_fdata()))
    right_plate = np.zeros(np.shape(axi.get_fdata()))
    total = np.zeros(np.shape(axi.get_fdata()))

    for i in range(len(axi_list)):
        axi_data = nib.load(axi_list[i]).get_fdata()
        cor_data = nib.load(cor_list[i]).get_fdata()
        if len(sag_list) > i:
            sag_data = nib.load(sag_list[i]).get_fdata()

        loc = np.where(axi_data==0)
        bak[loc]=bak[loc]+1
        loc = np.where(cor_data==0)
        bak[loc]=bak[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==0)
            bak[loc]=bak[loc]+1

        loc = np.where(axi_data==1)
        left_in[loc]=left_in[loc]+1
        loc = np.where(cor_data==1)
        left_in[loc]=left_in[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==1)
            left_in[loc]=left_in[loc]+1

        loc = np.where(axi_data==2)
        right_in[loc]=right_in[loc]+1
        loc = np.where(cor_data==2)
        right_in[loc]=right_in[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==1)
            right_in[loc]=right_in[loc]+1

        loc = np.where(axi_data==3)
        left_subplate[loc]=left_subplate[loc]+1
        loc = np.where(cor_data==3)
        left_subplate[loc]=left_subplate[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==2)
            left_subplate[loc]=left_subplate[loc]+1

        loc = np.where(axi_data==4)
        right_subplate[loc]=right_subplate[loc]+1
        loc = np.where(cor_data==4)
        right_subplate[loc]=right_subplate[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==2)
            right_subplate[loc]=right_subplate[loc]+1

        loc = np.where(axi_data==5)
        left_plate[loc]=left_plate[loc]+1
        loc = np.where(cor_data==5)
        left_plate[loc]=left_plate[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==3)
            left_plate[loc]=left_plate[loc]+1

        loc = np.where(axi_data==6)
        right_plate[loc]=right_plate[loc]+1
        loc = np.where(cor_data==6)
        right_plate[loc]=right_plate[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==3)
            right_plate[loc]=right_plate[loc]+1

    result = np.concatenate((bak[np.newaxis,:], left_in[np.newaxis,:], right_in[np.newaxis,:], left_subplate[np.newaxis,:], right_subplate[np.newaxis,:], left_plate[np.newaxis,:], right_plate[np.newaxis,:]),axis=0)
    result = np.argmax(result, axis=0)

    filename=input_name.split('/')[-1:][0].split('.nii')[0]
    filename=filename+'_deep_final.nii.gz'
    new_img = nib.Nifti1Image(result, axi.affine, axi.header)
    nib.save(new_img, result_loc+'/'+filename)

def axfliper(array,f=0):
    import numpy as np
    if f:
        array = array[:,:,::-1,:]
        array2 = np.concatenate((array[:,:,:,0,np.newaxis],array[:,:,:,2,np.newaxis],array[:,:,:,1,np.newaxis],
                                 array[:,:,:,4,np.newaxis],array[:,:,:,3,np.newaxis],
                                 array[:,:,:,6,np.newaxis],array[:,:,:,5,np.newaxis]),axis=-1)
        return array2
    else:
        array = array[:,:,::-1,:]
    return array

def cofliper(array,f=0):
    import numpy as np
    if f:
        array = array[:,::-1,:,:]
        array2 = np.concatenate((array[:,:,:,0,np.newaxis],array[:,:,:,2,np.newaxis],array[:,:,:,1,np.newaxis],
                                 array[:,:,:,4,np.newaxis],array[:,:,:,3,np.newaxis],
                                 array[:,:,:,6,np.newaxis],array[:,:,:,5,np.newaxis]),axis=-1)
        return array2
    else:
        array = array[:,::-1,:,:]
    return array

def make_dic_prev(img_list, gold_list, input_size, dim, flip=0,max_shape=[-1,-1,-1],out_ch = 7):
    import numpy as np
    import nibabel as nib
    from tqdm import tqdm
    def get_data(img, label, input_size, dim):
        import nibabel as nib
        img = np.squeeze(nib.load(img).get_fdata())
        img = img - img.min()
        img = img / img.max()
        img = img * 255
        label = np.squeeze(nib.load(label).get_fdata())
        if dim == 'axi':
            input_size = [*input_size, max_shape[2]]
        elif dim == 'cor':
            input_size = [input_size[0], max_shape[1], input_size[1]]
        elif dim == 'sag':
            input_size = [max_shape[0], *input_size]
        else:
            print('available: axi, cor, sag.   Your: '+dim)
            exit()

        img = crop_pad_ND(img, input_size)
        label = crop_pad_ND(label, input_size)
        return img, label

    #max_shape = [117,159,126]
    if dim == 'axi':
        dic = np.zeros([max_shape[2]*len(img_list), input_size[0], input_size[1], 1], dtype=np.float16)
        seg = np.zeros([max_shape[2]*len(img_list), input_size[0], input_size[1], out_ch], dtype=np.float16)
    elif dim == 'cor':
        dic = np.zeros([max_shape[1]*len(img_list), input_size[0], input_size[1], 1], dtype=np.float16)
        seg = np.zeros([max_shape[1]*len(img_list), input_size[0], input_size[1], out_ch], dtype=np.float16)
    elif dim == 'sag':
        dic = np.zeros([max_shape[0]*len(img_list), input_size[0], input_size[1], 1], dtype=np.float16)
        seg = np.zeros([max_shape[0]*len(img_list), input_size[0], input_size[1], out_ch], dtype=np.float16)
    else:
        print('available: axi, cor, sag.   Your: '+dim)
        exit()

    for i in tqdm(range(0, len(img_list)),desc=dim+' dic making..'):
        if dim == 'axi':
            img, label = get_data(img_list[i], gold_list[i], input_size, 'axi')
            dic[max_shape[2]*i:max_shape[2]*(i+1),:,:,0]= np.swapaxes(img,2,0)
            img2 = np.swapaxes(label,2,0)
        elif dim == 'cor':
            img, label = get_data(img_list[i], gold_list[i], input_size, 'cor')
            dic[max_shape[1]*i:max_shape[1]*(i+1),:,:,0]= np.swapaxes(img,1,0)
            img2 = np.swapaxes(label,1,0)
        elif dim == 'sag':
            img, label = get_data(img_list[i], gold_list[i], input_size, 'sag')
            dic[max_shape[0]*i:max_shape[0]*(i+1),:,:,0]= img
            img2 = label
        else:
            print('available: axi, cor, sag.   Your: '+dim)
            exit()
        
        if (dim == 'axi') | (dim == 'cor'):
            img3 = np.zeros_like(img2)
            back_loc = np.where(img2<0.5)
            left_plate_loc = np.where((img2>0.5)&(img2<1.5))
            right_plate_loc = np.where((img2>41.5)&(img2<42.5))
            left_in_loc = np.where((img2>160.5)&(img2<161.5))
            right_in_loc = np.where((img2>159.5)&(img2<160.5)) 
            left_subplate_loc = np.where((img2>4.5)&(img2<5.5))
            right_subplate_loc = np.where((img2>3.5)&(img2<4.5))
            img3[back_loc]=1
        elif dim == 'sag' and out_ch==4:
            img3 = np.zeros_like(img2)
            back_loc = np.where(img<0.5)
            plate_loc = np.where(((img2>0.5)&(img2<1.5))|((img2>41.5)&(img2<42.5)))
            in_loc = np.where(((img2>160.5)&(img2<161.5))|((img2>159.5)&(img2<160.5)))
            subplate_loc = np.where(((img2>3.5)&(img2<4.5))|((img2>4.5)&(img2<5.5)))
            img3[back_loc]=1
        elif dim == 'sag' and out_ch==7:
            img3 = np.zeros_like(img2)
            back_loc = np.where(img2<0.5)
            left_plate_loc = np.where((img2>0.5)&(img2<1.5))
            right_plate_loc = np.where((img2>41.5)&(img2<42.5))
            left_in_loc = np.where((img2>160.5)&(img2<161.5))
            right_in_loc = np.where((img2>159.5)&(img2<160.5)) 
            left_subplate_loc = np.where((img2>4.5)&(img2<5.5))
            right_subplate_loc = np.where((img2>3.5)&(img2<4.5))
            img3[back_loc]=1
        else:
            print('available: axi, cor, sag.   Your: '+dim)
            exit()

        if dim == 'axi':
            seg[max_shape[2]*i:max_shape[2]*(i+1),:,:,0]=img3
            img3[:]=0
            img3[left_in_loc]=1
            seg[max_shape[2]*i:max_shape[2]*(i+1),:,:,1]=img3
            img3[:]=0
            img3[right_in_loc]=1
            seg[max_shape[2]*i:max_shape[2]*(i+1),:,:,2]=img3
            img3[:]=0
            img3[left_subplate_loc]=1
            seg[max_shape[2]*i:max_shape[2]*(i+1),:,:,3]=img3
            img3[:]=0
            img3[right_subplate_loc]=1
            seg[max_shape[2]*i:max_shape[2]*(i+1),:,:,4]=img3
            img3[:]=0
            img3[left_plate_loc]=1
            seg[max_shape[2]*i:max_shape[2]*(i+1),:,:,5]=img3
            img3[:]=0
            img3[right_plate_loc]=1
            seg[max_shape[2]*i:max_shape[2]*(i+1),:,:,6]=img3
            img3[:]=0            
        elif dim == 'cor':
            seg[max_shape[1]*i:max_shape[1]*(i+1),:,:,0]=img3
            img3[:]=0
            img3[left_in_loc]=1
            seg[max_shape[1]*i:max_shape[1]*(i+1),:,:,1]=img3
            img3[:]=0
            img3[right_in_loc]=1
            seg[max_shape[1]*i:max_shape[1]*(i+1),:,:,2]=img3
            img3[:]=0
            img3[left_subplate_loc]=1
            seg[max_shape[1]*i:max_shape[1]*(i+1),:,:,3]=img3
            img3[:]=0
            img3[right_subplate_loc]=1
            seg[max_shape[1]*i:max_shape[1]*(i+1),:,:,4]=img3
            img3[:]=0
            img3[left_plate_loc]=1
            seg[max_shape[1]*i:max_shape[1]*(i+1),:,:,5]=img3
            img3[:]=0
            img3[right_plate_loc]=1
            seg[max_shape[1]*i:max_shape[1]*(i+1),:,:,6]=img3
            img3[:]=0
        elif dim == 'sag' and out_ch==4:
            seg[max_shape[0]*i:max_shape[0]*(i+1),:,:,0]=img3
            img3[:]=0
            img3[in_loc]=1
            seg[max_shape[0]*i:max_shape[0]*(i+1),:,:,1]=img3
            img3[:]=0
            img3[subplate_loc]=1
            seg[max_shape[0]*i:max_shape[0]*(i+1),:,:,2]=img3
            img3[:]=0
            img3[plate_loc]=1
            seg[max_shape[0]*i:max_shape[0]*(i+1),:,:,3]=img3
            img3[:]=0
        elif dim == 'sag' and out_ch==7:
            seg[max_shape[0]*i:max_shape[0]*(i+1),:,:,0]=img3
            img3[:]=0
            img3[left_in_loc]=1
            seg[max_shape[0]*i:max_shape[0]*(i+1),:,:,1]=img3
            img3[:]=0
            img3[right_in_loc]=1
            seg[max_shape[0]*i:max_shape[0]*(i+1),:,:,2]=img3
            img3[:]=0
            img3[left_subplate_loc]=1
            seg[max_shape[0]*i:max_shape[0]*(i+1),:,:,3]=img3
            img3[:]=0
            img3[right_subplate_loc]=1
            seg[max_shape[0]*i:max_shape[0]*(i+1),:,:,4]=img3
            img3[:]=0
            img3[left_plate_loc]=1
            seg[max_shape[0]*i:max_shape[0]*(i+1),:,:,5]=img3
            img3[:]=0
            img3[right_plate_loc]=1
            seg[max_shape[0]*i:max_shape[0]*(i+1),:,:,6]=img3
            img3[:]=0
        else:
            print('available: axi, cor, sag.   Your: '+dim)
            exit()
    if flip:
        if dim == 'axi':
            dic=np.concatenate((dic,dic[:,::-1,:,:]),axis=0)
            seg=np.concatenate((seg,seg[:,::-1,:,:]),axis=0)
            dic=np.concatenate((dic, axfliper(dic)),axis=0)
            seg=np.concatenate((seg, axfliper(seg, 1)),axis=0)
        elif dim == 'cor':
            dic=np.concatenate((dic,dic[:,:,::-1,:]),axis=0)
            seg=np.concatenate((seg,seg[:,:,::-1,:]),axis=0)
            dic=np.concatenate((dic, cofliper(dic)),axis=0)
            seg=np.concatenate((seg, cofliper(seg, 1)),axis=0)
        elif dim == 'sag':
            dic = np.concatenate((dic, dic[:,:,::-1,:], dic[:,::-1,:,:]),axis=0)
            seg = np.concatenate((seg, seg[:,:,::-1,:], seg[:,::-1,:,:]),axis=0)
        else:
            print('available: axi, cor, sag.   Your: '+dim)
            exit()
    print(dic.dtype)
    print(seg.dtype)
    return dic, seg