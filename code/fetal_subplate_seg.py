from dataclasses import dataclass
import numpy as np
import glob, os, time, pickle, sys
#os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import argparse
from deep_util_training_sp import *
from deep_classes_sp import *
from deep_util_records import *
import imgaug.augmenters as iaa
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
# previous values, using lanas and jinwoo


#full dataset ,march 17, validation and train split 10 percent
train_len = {
    'axi':39060,
    'cor':47628,
    'sag':25515,
}
val_len = {
    'axi':4960,
    'cor':6048,
    'sag':3240,
}
'''
'''


def main():
    def parse_tfr_element(element):
        #use the same structure as above; it's kinda an outline of the structure we now want to create
        data = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width':tf.io.FixedLenFeature([], tf.int64),
            'channels':tf.io.FixedLenFeature([], tf.int64),
            'raw_image' : tf.io.FixedLenFeature([], tf.string),
        }
        for elem in range(currview.out_ch):
            data['label'+str(elem)] = tf.io.FixedLenFeature([], tf.string)  
        content = tf.io.parse_single_example(element, data)
        height = content['height']
        width = content['width']
        channels = content['channels']
        raw_image = content['raw_image']
        if currview.out_ch == 7:
            label0 = content['label0']
            label1 = content['label1']
            label2 = content['label2']
            label3 = content['label3']
            label4 = content['label4']
            label5 = content['label5']
            label6 = content['label6']
        elif currview.out_ch == 4:
            label0 = content['label0']
            label1 = content['label1']
            label2 = content['label2']
            label3 = content['label3']

        #get our 'feature'-- our image -- and reshape it appropriately
        feature = tf.io.parse_tensor(raw_image, out_type=tf.float16)
        feature = tf.reshape(feature, shape=[height,width,1])
        if currview.out_ch == 7:
            label0_mask = tf.io.parse_tensor(label0, out_type=tf.float16)
            label1_mask = tf.io.parse_tensor(label1, out_type=tf.float16)
            label2_mask = tf.io.parse_tensor(label2, out_type=tf.float16)
            label3_mask = tf.io.parse_tensor(label3, out_type=tf.float16)
            label4_mask = tf.io.parse_tensor(label4, out_type=tf.float16)
            label5_mask = tf.io.parse_tensor(label5, out_type=tf.float16)
            label6_mask = tf.io.parse_tensor(label6, out_type=tf.float16)
            label0_mask = tf.reshape(label0_mask, shape=[height,width,1])
            label1_mask = tf.reshape(label1_mask, shape=[height,width,1])
            label2_mask = tf.reshape(label2_mask, shape=[height,width,1])
            label3_mask = tf.reshape(label3_mask, shape=[height,width,1])
            label4_mask = tf.reshape(label4_mask, shape=[height,width,1])
            label5_mask = tf.reshape(label5_mask, shape=[height,width,1])
            label6_mask = tf.reshape(label6_mask, shape=[height,width,1])
            label = tf.concat([label0_mask, label1_mask,label2_mask,label3_mask,label4_mask,label5_mask,label6_mask], axis=2)
        if currview.out_ch == 4:
            label0_mask = tf.io.parse_tensor(label0, out_type=tf.float16)
            label1_mask = tf.io.parse_tensor(label1, out_type=tf.float16)
            label2_mask = tf.io.parse_tensor(label2, out_type=tf.float16)
            label3_mask = tf.io.parse_tensor(label3, out_type=tf.float16)
            label0_mask = tf.reshape(label0_mask, shape=[height,width,1])
            label1_mask = tf.reshape(label1_mask, shape=[height,width,1])
            label2_mask = tf.reshape(label2_mask, shape=[height,width,1])
            label3_mask = tf.reshape(label3_mask, shape=[height,width,1])
            label = tf.concat([label0_mask, label1_mask,label2_mask,label3_mask], axis=2)
        return (feature, label)

    def get_dataset_small(filename):
        #create the dataset
        dataset = tf.data.TFRecordDataset(filename)
        #pass every single feature through our mapping function
        dataset = dataset.map(parse_tfr_element)    
        return dataset

    def train_view(view):
        r"""
        Train_view creates the dictionaries for training, the model, 
        callbacks and start training if requested.

        Parameters
        ----------
        dim : Is the dimension in string, choices [axi,cor,sag]
        dim_index : goes from 0 to 2, is directly related to the chosen dim, axi:0, cor:1, sag:2  

        Returns
        -------
        None
        """
        #If mirrored strategy is selected, then variables, model, 
        if args.multi_gpu:
            with strategy.scope():
                print('shape : '+ str(view.string))
                model = Unet_network([*args.isize,1], view.out_ch, loss=args.loss, metrics=args.coef, style=args.style, ite=args.ite, 
                                        depth=args.depth, dim=args.dim, init=args.init, acti=args.acti, lr=args.lr,optimizer=args.opt)
                model = model.build()
        else:
            model = Unet_network([*args.isize,1], view.out_ch, loss=args.loss, metrics=args.coef, style=args.style, ite=args.ite, 
                                        depth=args.depth, dim=args.dim, init=args.init, acti=args.acti, lr=args.lr,optimizer=args.opt)
            model = model.build()
        callbacks=make_callbacks(weight_loc+'fold'+str(ii)+str(view.string)+'.h5', hist_loc+'fold'+str(ii)+str(view.string)+'.tsv', view_string = view.string,
            epoch_size=train_len[view.string]/(BATCH_SIZE), batch_size=BATCH_SIZE,SGR_schedule=args.restart,monitor=args.monitor,min_lr=args.min_lr,max_lr=args.max_lr, use_tensorboard=args.tensorboard)
        
        if args.lr_finder:
            lr_finder = LRFinder(min_lr=args.min_lr, 
                max_lr=args.max_lr, 
                steps_per_epoch=train_len[view.string]/(BATCH_SIZE), 
                epochs=args.epoch)
            callbacks = [lr_finder]
        
        if os.path.exists(weight_loc+'fold'+str(ii)+str(view.string)+'.h5'):
            model.load_weights(weight_loc+'fold'+str(ii)+str(view.string)+'.h5')
        
        if args.to:
            SHUFFLE_BUFFER_SIZE = 150
            train_dataset = get_dataset_small(args.in_fol_rec+'/'+view.string+'train_data.tfrecords')
            print('Loaded training tf records')
            #for sample in train_dataset.take(1):
            #    print(sample[0].shape)
            #    print(sample[1].shape)
            
            train_dataset = (
                train_dataset
                .shuffle(SHUFFLE_BUFFER_SIZE)
                .batch(BATCH_SIZE,drop_remainder=True)
                .map(Augment(augmentFlag = args.aug), 
                    num_parallel_calls=AUTOTUNE,
                    # Order does not matter.
                    deterministic=True
                )
                .repeat()
                .prefetch(AUTOTUNE)
            )
            if args.no_validate:
                #create the dataset based on the tf.dataset
                # Prefetch some batches.
                model.fit(train_dataset,
                            steps_per_epoch=train_len[view.string]/(BATCH_SIZE),
                            max_queue_size=10,
                            workers=10,
                            epochs=args.epoch, verbose=2, callbacks=callbacks)

            else:
                val_dataset = get_dataset_small(args.in_fol_rec+'/'+view.string+'val_data.tfrecords')
                print('Loaded validation tf records')
                val_dataset = (
                    val_dataset
                    .shuffle(SHUFFLE_BUFFER_SIZE)
                    .batch(BATCH_SIZE,drop_remainder=True)
                    .map(Augment(augmentFlag = args.aug), 
                        num_parallel_calls=AUTOTUNE,
                        # Order does not matter.
                        deterministic=True
                    )
                    .repeat()
                    .prefetch(AUTOTUNE)
                )
                model.fit(train_dataset,
                        steps_per_epoch=train_len[view.string]/(BATCH_SIZE),
                        validation_data=val_dataset,
                        validation_steps = val_len[view.string]/(BATCH_SIZE),
                        max_queue_size=6,
                        workers=8,
                        epochs=args.epoch, verbose=2, callbacks=callbacks)
                del val_dataset
            if args.lr_finder:
                lr_finder.plot_loss()
                lr_finder.plot_lr()
        del model, callbacks, train_dataset
        reset_gpu()
        return

    parser = argparse.ArgumentParser('   ==========   Fetal U_Net segmentation script made by IBHM (11.28 ver.0)   ==========   ')
    parser.add_argument('-infol_MR', '--input_MR_folder',action='store',dest='in_fol_MR',type=str, help='input MR folder name for training')
    parser.add_argument('-infol_rec', '--input_records_folder',action='store',dest='in_fol_rec',required = True,type=str, help='input record folder name for training')
    parser.add_argument('-infol_GT', '--input_GT_folder',action='store',dest='in_fol_GT',type=str, help='input GT folder name for training')
    parser.add_argument('-rl', '--result_save_locaiton', action='store',dest='result_loc', type=str, help='Output folder name, default: result/conv_style/')
    parser.add_argument('-wl', '--weight_save_location', action='store',dest='weight_loc', type=str, help='Output folder name, default: weights/conv_style/')
    parser.add_argument('-hl', '--history_save_location', action='store',dest='hist_loc', type=str, help='Output folder name, default: history/conv_style/')
    parser.add_argument('-f', '--num_fold',action='store',dest='num_fold',default=10, type=int, help='number of fold for training')
    parser.add_argument('-fi', '--stratified_info_file',action='store', dest='stratified_info',type=str, help='information for stratified fold')
    parser.add_argument('-fs', '--start_fold',action='store',dest='start_fold', type=int, help='number of fold for training')
    parser.add_argument('-fe', '--end_fold',action='store',dest='end_fold', type=int, default=1, help='number of fold for training')
    parser.add_argument('-is', '--input_shape',action='store', dest='isize',type=int, nargs='+',default=[192,192], help='Input size ex.-is 100 100')
    parser.add_argument('-bs', '--batch_size',action='store', dest='bsize',type=int, default=30, help='batch size')
    parser.add_argument('-e', '--epoch',action='store',dest='epoch',default=1500,  type=int, help='Number of epoch for training')
    parser.add_argument('-s', '--conv_style', choices=['basic','res','dense','RCL'], default='basic', action='store',dest='style', type=str, help='Conv block style')
    parser.add_argument('-n', '--conv_num', default=3, action='store',dest='ite', type=int, help='Number of convolution in block')
    parser.add_argument('-mnt', '--monitor', choices=['val_loss','loss','dice_coef','val_dice_coef'], default='val_loss', action='store',dest='monitor', type=str, help='variable to monitor')
    parser.add_argument('-d', '--model_depth', default=4, action='store',dest='depth', type=int, help='Deep learning model detph')
    parser.add_argument('-c', '--n_channel', default=32, action='store',dest='dim', type=int, help='Start convolution channel size')
    parser.add_argument('-i', '--kernel_initial', choices=['he_normal', 'TruncatedNormal', 'RandomNormal'], default='he_normal', action='store',dest='init', type=str, help='Convolution weight initial method')
    parser.add_argument('-a', '--activation', choices=['elu', 'relu'], default='elu', action='store',dest='acti', type=str, help='Activation method')
    parser.add_argument('-lr', '--learning_rate', default=1e-4, action='store',dest='lr', type=float, help='Learning rate')
    parser.add_argument('-l', '--loss', choices=['hyb_loss', 'hyb_loss2', 'ori_dice_loss', 'dice_loss','dis_dice_loss','focal_tversky_loss','dis_loss','focal_loss','hyb_disdice_focal_loss','asymmetric_focal_tversky_loss','focal_loss_2','SSIM_loss','hyb_ssim_focal_tversky_loss','focal_loss_3'], default='dice_loss', action='store',dest='loss', type=str, help='Loss function')
    parser.add_argument('-m', '--metric', choices=['dice_coef', 'dis_dice_coef'], default=['dice_coef'], nargs='*', action='store',dest='coef', help='Eval metric')
    parser.add_argument('-p', '--predict_only', action='store_false', dest='to', help='Training off option')
    parser.add_argument('-gpu', '--gpu_number',action='store',dest='gpu',type=int, default=-1, help='Select GPU')
    parser.add_argument('-il', '--input_axisdim',action='store', dest='iaxis',type=int, nargs='+', default=[7,7,4], help='Input size ex.-is 7 7 4')
    parser.add_argument('-opt', '--optimizer',action='store', dest='opt',type=str, choices=['Adam', 'SGD','RAdam'],default="Adam", help='define which optimizer to use')
    parser.add_argument('-wr','--SGDR_restart', dest='restart', action='store_true')
    parser.add_argument('-axi','--axial_training', dest='axial_training', action='store_true')
    parser.add_argument('-cor','--cor_training', dest='cor_training', action='store_true')
    parser.add_argument('-sag','--sag_training', dest='sag_training', action='store_true')
    parser.add_argument('-all','--all_training', dest='all_training', action='store_true')
    parser.add_argument('-fp','--flip', dest='flip_images', action='store_true')
    parser.add_argument('-nv','--no_validate', dest='no_validate', action='store_true')
    parser.add_argument('-tb','--use_tensorboard', dest='tensorboard', action='store_true')
    parser.add_argument('-uf','--use_folder', dest='use_folder', action='store_true')
    parser.add_argument('-mgpu','--multigpu', dest='multi_gpu', action='store_true')
    parser.add_argument('-mw','--multi_worker', dest='multi_worker', action='store_true')
    parser.add_argument('-minlr', '--min_learning_rate', default=1e-7, action='store',dest='min_lr', type=float, help='min Learning rate')
    parser.add_argument('-maxlr', '--max_learning_rate', default=0.1, action='store',dest='max_lr', type=float, help='max Learning rate')
    parser.add_argument('-lrf', '--lr_finder', dest='lr_finder', action='store_true',help='learning rate finder')
    parser.add_argument('-aug', '--augment', dest='aug', action='store_true',help='augment images')
    parser.add_argument('-sm', '--split_method', choices=['skf', 'no_split','percent'], default=['percent'], action='store',dest='split_method', help='split method to use')
    args = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_usage()
        exit()

    if args.in_fol_MR is not None:
        #img_list = np.asarray(sorted(glob.glob(args.in_fol_MR+'/*nuc.nii')))
        img_list = np.asarray(sorted(glob.glob(args.in_fol_MR+'/*nuc_nonmasked.nii')))
        gold_list = np.asarray(sorted(glob.glob(args.in_fol_GT+'/*dilate.nii')))

    if args.multi_gpu == False:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    import tensorflow as tf
    # This is tf.data.experimental.AUTOTUNE in older tensorflow.
    AUTOTUNE = tf.data.AUTOTUNE
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    BATCH_SIZE_PER_REPLICA = args.bsize
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    start=0
    end=args.num_fold
    if args.start_fold is not None:
        if args.start_fold<end:
            start=args.start_fold
    if args.end_fold is not None:
        if args.end_fold<=end:
            end=args.end_fold

    coef = str(args.coef).replace('[','').replace(']','').replace('\'','').replace(', ',',')
    if len(args.coef)>1:
        coef='both'

    if args.weight_loc is None:
        weight_loc='weight/'+args.style+'_nc'+str(args.ite)\
        +'d'+str(args.depth)+'c'+str(args.dim)\
        +'_loss_'+args.loss+'_metric_'+coef+'/'
        hist_loc='history/'+args.style+'_nc'+str(args.ite)\
        +'d'+str(args.depth)+'c'+str(args.dim)\
        +'_loss_'+args.loss+'_metric_'+coef+'/'
    else:
        weight_loc=args.weight_loc
        hist_loc=args.weight_loc

    if os.path.exists(weight_loc)==False:
        os.makedirs(weight_loc, exist_ok=True)

    print('\n\n')
    if args.in_fol_rec is not None:
        print('{0:37}  {1}'.format('TFrecords folder:',os.path.realpath(args.in_fol_rec)))
    print('{0:37}  {1}'.format('Weights save location:',weight_loc))
    print('{0:37}  {1}'.format('input_shape:',str(args.isize)))
    print('{0:37}  {1}'.format('batch_size:',str(args.bsize)))
    print('{0:37}  {1}'.format('Convolution style:',args.style))
    print('{0:37}  {1}'.format('Number of convolution in block:',str(args.ite)))
    print('{0:37}  {1}'.format('Model depth:',str(args.depth)))
    print('{0:37}  {1}'.format('First convolution channel size:',str(args.dim)))
    print('{0:37}  {1}'.format('Kernel initialization method:',args.init))
    print('{0:37}  {1}'.format('Activation kernel:',args.acti))
    print('{0:37}  {1}'.format('Learning rate:',str(args.lr)))
    print('{0:37}  {1}'.format('Training loss:',args.loss))
    print('{0:37}  {1}'.format('Training metric:',str(args.coef)))
    print('{0:37}  {1}'.format('Epochs:',str(args.epoch)))
    print('{0:37}  {1}'.format('GPU number:',str(args.gpu)))
    print('{0:37}  {1}'.format('Labels per axis:',str(args.iaxis)))
    print('{0:37}  {1}'.format('Optimizer to use:',str(args.opt)))
    print('{0:37}  {1}'.format('Use restart for SGD:',str(args.restart)))
    print('{0:37}  {1}'.format('Flip images for dictionaries:',str(args.flip_images)))
    print('{0:37}  {1}'.format('Not to use validation data:',str(args.no_validate)))
    print('{0:37}  {1}'.format('Min_lr:',str(args.min_lr)))
    print('{0:37}  {1}'.format('Max_lr:',str(args.max_lr)))
    print('{0:37}  {1}'.format('Use tensorboard:',str(args.tensorboard)))


    if args.in_fol_MR is not None:
        max_shape = get_maxshape(img_list)
        print(max_shape)
    else:
        #Default value lanas
        #max_shape = [135, 189, 155]
        #Default value prev
        max_shape = [117,159,126]

    shape = args.iaxis

    ##LABEL STATEMENT
    ##label order matters, remember to put left and then right in that order, not skipping.
    labels_dict = {
        'back_label':0,
        'left_plate_label':1,
        'right_plate_label':42,
        'left_subplate_label':5,
        'right_subplate_label':4,
        'left_in_label':161,
        'right_in_label':160
    }

    #labels index to use
    for ii in range(start, end):
        if args.axial_training or args.all_training:
            print('\n\n fold'+str(ii)+' axi processing... \n\n')
            currview = anatomicalview(string='axi',max_slices=max_shape[2],out_ch=args.iaxis[0])
            train_view(currview)
            reset_gpu()
        if args.cor_training or args.all_training:
            print('\n\n fold'+str(ii)+' cor processing... \n\n')
            currview = anatomicalview(string='cor',max_slices=max_shape[1],out_ch=args.iaxis[1])
            train_view(currview)
            reset_gpu()

        if args.sag_training or args.all_training:
            print('\n\n fold'+str(ii)+' sag processing... \n\n')
            currview = anatomicalview(string='sag',max_slices=max_shape[0],out_ch=args.iaxis[2])
            train_view(currview)            
            reset_gpu()

        print("Sucess")

if __name__ == "__main__":
    main()
