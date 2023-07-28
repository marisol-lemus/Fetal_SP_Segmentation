from dataclasses import dataclass
import numpy as np
import glob, os, time, pickle, sys
import argparse
from deep_util_training_sp import *
from deep_classes_sp import *
import imgaug.augmenters as iaa
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from deep_util_records import *

def main():
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
        #train_dic, train_seg = make_dic(train_list, train_gold_list,args.isize, view = view, labels_dict=labels_dict,flip=args.flip_images)
        #val_dic, val_seg = make_dic(val_list, val_gold_list,args.isize, view = view, labels_dict=labels_dict,flip=args.flip_images)
        
        train_dic, train_seg = make_dic_prev(train_list, train_gold_list, args.isize, view.string,max_shape=max_shape,flip=args.flip_images,out_ch = view.out_ch)
        val_dic, val_seg = make_dic_prev(val_list, val_gold_list, args.isize, view.string,max_shape=max_shape,flip=args.flip_images,out_ch = view.out_ch)
        print('train seg shape is: ',train_seg.shape)
        print('val seg shape is: ',val_seg.shape)
        #Create the tf record
        # The number of observations in the dataset.
        n_observations = len(train_dic)
        count = write_images_to_tfr_short(train_dic, train_seg, folder = weight_loc, filename=view.string+"train_data")
        count = write_images_to_tfr_short(val_dic, val_seg, folder = weight_loc, filename=view.string+"val_data")
        return

    parser = argparse.ArgumentParser('   ==========   Fetal U_Net segmentation script made by IBHM (11.28 ver.0)   ==========   ')
    parser.add_argument('-infol_MR', '--input_MR_folder',action='store',dest='in_fol_MR',type=str, required=True, help='input MR folder name for training')
    parser.add_argument('-infol_GT', '--input_GT_folder',action='store',dest='in_fol_GT',type=str, required=True, help='input GT folder name for training')
    parser.add_argument('-rl', '--result_save_locaiton', action='store',dest='result_loc', type=str, help='Output folder name, default: result/conv_style/')
    parser.add_argument('-wl', '--weight_save_location', action='store',dest='weight_loc', type=str, help='Output folder name, default: weights/conv_style/')
    parser.add_argument('-hl', '--history_save_location', action='store',dest='hist_loc', type=str, help='Output folder name, default: history/conv_style/')
    parser.add_argument('-f', '--num_fold',action='store',dest='num_fold',default=10, type=int, help='number of fold for training')
    parser.add_argument('-fi', '--stratified_info_file',action='store', dest='stratified_info',type=str, help='information for stratified fold')
    parser.add_argument('-fs', '--start_fold',action='store',dest='start_fold', type=int, help='number of fold for training')
    parser.add_argument('-fe', '--end_fold',action='store',dest='end_fold', type=int, help='number of fold for training')
    parser.add_argument('-is', '--input_shape',action='store', dest='isize',type=int, nargs='+',default=[192,192], help='Input size ex.-is 100 100')
    parser.add_argument('-bs', '--batch_size',action='store', dest='bsize',type=int, default=30, help='batch size')
    parser.add_argument('-e', '--epoch',action='store',dest='epoch',default=1500,  type=int, help='Number of epoch for training')
    parser.add_argument('-s', '--conv_style', choices=['basic','res','dense','RCL'], default='basic', action='store',dest='style', type=str, help='Conv block style')
    parser.add_argument('-n', '--conv_num', default=3, action='store',dest='ite', type=int, help='Number of convolution in block')
    parser.add_argument('-mnt', '--monitor', choices=['val_loss','loss','dice_coef','val_dice_coef'], default='val_dice_coef', action='store',dest='monitor', type=str, help='variable to monitor')
    parser.add_argument('-d', '--model_depth', default=4, action='store',dest='depth', type=int, help='Deep learning model detph')
    parser.add_argument('-c', '--n_channel', default=32, action='store',dest='dim', type=int, help='Start convolution channel size')
    parser.add_argument('-i', '--kernel_initial', choices=['he_normal', 'TruncatedNormal', 'RandomNormal'], default='he_normal', action='store',dest='init', type=str, help='Convolution weight initial method')
    parser.add_argument('-a', '--activation', choices=['elu', 'relu'], default='elu', action='store',dest='acti', type=str, help='Activation method')
    parser.add_argument('-lr', '--learning_rate', default=1e-4, action='store',dest='lr', type=float, help='Learning rate')

    parser.add_argument('-l', '--loss', choices=['hyb_loss', 'hyb_loss2', 'ori_dice_loss', 'dice_loss','dis_dice_loss'], default='dice_loss', action='store',dest='loss', type=str, help='Loss function')
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
    parser.add_argument('-uf','--use_folder', dest='use_folder', action='store_true')
    parser.add_argument('-ms','--mirrored_st', dest='mirrored_st', action='store_true')
    parser.add_argument('-mw','--multi_worker', dest='multi_worker', action='store_true')
    parser.add_argument('-aug', '--augment', dest='aug', action='store_true',help='augment images')
    parser.add_argument('-sm', '--split_method', choices=['skf', 'no_split','percent'], default=['percent'], action='store',dest='split_method', help='split method to use')

    
    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_usage()
        exit()

    #img_list = np.asarray(sorted(glob.glob(args.in_fol_MR+'/*nuc.nii')))
    img_list = np.asarray(sorted(glob.glob(args.in_fol_MR+'/*nuc_nonmasked.nii')))
    gold_list = np.asarray(sorted(glob.glob(args.in_fol_GT+'/*dilate.nii')))

    if args.mirrored_st == False:
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
    print('{0:37}  {1}'.format('Training image folder:',os.path.realpath(args.in_fol_MR)))
    print('{0:37}  {1}'.format('Training ground truth folder:',os.path.realpath(args.in_fol_GT)))
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
    print('{0:37}  {1}'.format('Use restart for SGD:',str(args.restart)))
    print('{0:37}  {1}'.format('Flip images for dictionaries:',str(args.flip_images)))
    print('{0:37}  {1}'.format('Not to use validation data:',str(args.no_validate)))

    max_shape = get_maxshape(img_list)
    print(max_shape)

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
        if args.split_method == 'percent':
            train_list, val_list, train_gold_list , val_gold_list = train_test_split(img_list,gold_list,test_size=0.20,random_state=42)
            print (img_list)
            print("train list contains:" +str(len(train_list)))
            print(train_list)
            print(train_gold_list)
            print ("si jalo")
            print("val list contains:" +str(len(val_list)))
            print(val_list)
            print(val_gold_list)

        elif args.split_method=='skf':
            fold_group = np.loadtxt(args.stratified_info).astype(int)
            #print(fold_group)
            #skf = StratifiedKFold(n_splits=args.num_fold, random_state=1,shuffle=True)
            skf = StratifiedKFold(n_splits=args.num_fold, random_state=1,shuffle=True)
            fold_info = list(skf.split(img_list, fold_group))

            tr2 = fold_info[ii][0]
            te = fold_info[ii][1]
            in_fold_info = list(skf.split(img_list[tr2],fold_group[tr2]))[0]
            #tr = tr2[in_fold_info[0]]
            va = tr2[in_fold_info[1]]
            tr  = np.concatenate((tr2[in_fold_info[0]],fold_info[ii][1]))

            train_list = img_list[tr]
            val_list = img_list[va]
            print("train list contains:" +str(len(train_list)))
            print(train_list)
            print("val list contains:" +str(len(val_list)))
            print(val_list)
            test_list = img_list[te]
            train_gold_list = gold_list[tr]
            val_gold_list = gold_list[va]
            test_gold_list = gold_list[te]
        else:
            train_list = img_list
            val_list = img_list
            print("train listcontains:" +str(len(train_list)))
            print(train_list)
            print("val list contains:" +str(len(val_list)))
            print(val_list)
            train_gold_list = gold_list
            val_gold_list = gold_list
            
        if args.no_validate:
            train_list = img_list
            train_gold_list = gold_list 

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
