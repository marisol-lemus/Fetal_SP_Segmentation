"""Console script for fetal_brain_segmentation."""
import argparse, tempfile
import sys
import numpy as np
import medpy
from medpy.io import load, save
import glob, os, time, pickle
sys.path.append(os.path.dirname(__file__))
from deep_util_sp import *


def main():
    parser = argparse.ArgumentParser('   ==========   Fetal U_Net segmentation script made by Marisol Lemus (November 16, 2021 ver.1)   ==========   ')
    parser.add_argument('-input', '--input_loc',action='store',dest='inp',type=str, required=True, help='input MR folder name for training')
    parser.add_argument('-output', '--output_loc',action='store',dest='out',type=str, required=True, help='Output path')
    parser.add_argument('-rl', '--result_save_location', action='store',dest='result_loc', type=str, help='Output folder name, default: result/conv_style/')
    parser.add_argument('-wl', '--weight_save_location', action='store',dest='weight_loc', default = os.path.dirname(os.path.abspath(__file__)), type=str, help='Output folder name, default: weights/conv_style/')
    parser.add_argument('-axi', '--axi_weight',action='store',dest='axi',default=os.path.dirname(os.path.abspath(__file__))+'/axi.h5',type=str, help='Axial weight file')
    parser.add_argument('-cor', '--cor_weight',action='store',dest='cor',default=os.path.dirname(os.path.abspath(__file__))+'/cor.h5',type=str, help='Coronal weight file')
    parser.add_argument('-sag', '--sag_weight',action='store',dest='sag',default=os.path.dirname(os.path.abspath(__file__))+'/sag.h5',type=str, help='Sagittal weight file')
    parser.add_argument('-hl', '--history_save_location', action='store',dest='hist_loc', type=str, default=os.path.dirname(os.path.abspath(__file__)), help='Output folder name, default: history/conv_style/')
    parser.add_argument('-is', '--input_shape',action='store', dest='isize',type=int, nargs='+', default= [192, 192], help='Input size ex.-is 100 100 1')
    parser.add_argument('-bs', '--batch_size',action='store', dest='bsize',type=int, default=30, help='batch size')
    parser.add_argument('-e', '--epoch',action='store',dest='epoch',default=1500,  type=int, help='Number of epoch for training')
    parser.add_argument('-s', '--conv_style', choices=['basic','res','dense','RCL'], default='basic', action='store',dest='style', type=str, help='Conv block style')
    parser.add_argument('-n', '--conv_num', default=3, action='store',dest='ite', type=int, help='Number of convolution in block')
    parser.add_argument('-d', '--model_depth', default=4, action='store',dest='depth', type=int, help='Deep learning model detph')
    parser.add_argument('-c', '--n_channel', default=32, action='store',dest='dim', type=int, help='Start convolution channel size')
    parser.add_argument('-i', '--kernel_initial', choices=['he_normal', 'TruncatedNormal', 'RandomNormal'], default='he_normal', action='store',dest='init', type=str, help='Convolution weight initial method')
    parser.add_argument('-a', '--activation', choices=['elu', 'relu'], default='elu', action='store',dest='acti', type=str, help='Activation method')
    parser.add_argument('-lr', '--learning_rate', default=1e-4, action='store',dest='lr', type=float, help='Learning rate')
    parser.add_argument('-l', '--loss', choices=['hyb_loss', 'hyb_loss2', 'ori_dice_loss', 'dice_loss','dis_dice_loss'], default='dice_loss', action='store',dest='loss', type=str, help='Loss function')
    parser.add_argument('-m', '--metric', choices=['dice_coef', 'dis_dice_coef'], default=['dice_coef'], nargs='*', action='store',dest='coef', help='Eval metric')
    parser.add_argument('-gpu', '--gpu_number',action='store',dest='gpu',type=str, default='-1',help='Select GPU')
    parser.add_argument('-il', '--input_label',action='store', dest='ilabel',type=int, nargs='+', default = [1,2,5,6,4,3], help='Input labels ex: 1 2 3 4 5 6')
    parser.add_argument('-ol', '--output_label',action='store', dest='olabel',type=int, nargs='+', default = [161,160,1,42,4,5], help='output labels ex: 160 161 160 161 0 0')
    parser.add_argument('-mr', '--merge', dest='merge', action='store_true',help='merge subplate with inner')
    parser.add_argument('-il2', '--input_axisdim',action='store', dest='iaxis',type=int, nargs='+', default=[7,7,4], help='Input size ex.-is 7 7 4')
    parser.add_argument('-cp', '--cp_seg',action='store', dest='cp_seg',help='use cp segmentation to define the CP on SP segmentation')

    args = parser.parse_args()
    #print("Arguments: " + str(args._))
    
    if len(sys.argv) < 2:
        parser.print_usage()
        exit()

    if os.path.isdir(args.inp):
        img_list = np.asarray(sorted(glob.glob(args.inp+'/*nuc.nii*')))
    elif os.path.isfile(args.inp):
        img_list = np.asarray(sorted(glob.glob(args.inp)))
    else:
        img_list = np.asarray(sorted(glob.glob(args.inp)))
    
    if len(img_list)==0:
        print('No such file or directory')
        exit()

    set_gpu(args.gpu)
    max_shape = get_maxshape(img_list)

    coef = str(args.coef).replace('[','').replace(']','').replace('\'','').replace(', ',',')
    if len(args.coef)>1:
        coef='both'

    if args.result_loc is None:
        result_loc='result/'+args.style+'_nc'+str(args.ite)\
        +'d'+str(args.depth)+'c'+str(args.dim)\
        +'_loss_'+args.loss+'_metric_'+coef+'/'
    else:
        result_loc=args.result_loc

    if args.weight_loc is None:
        weight_loc='weight/'+args.style+'_nc'+str(args.ite)\
        +'d'+str(args.depth)+'c'+str(args.dim)\
        +'_loss_'+args.loss+'_metric_'+coef+'/'
    else:
        weight_loc=args.weight_loc

    if args.hist_loc is None:
        hist_loc='history/'+args.style+'_nc'+str(args.ite)\
        +'d'+str(args.depth)+'c'+str(args.dim)\
        +'_loss_'+args.loss+'_metric_'+coef+'/'
    else:
        hist_loc=args.hist_loc

    if os.path.exists(result_loc)==False:
        os.makedirs(result_loc,exist_ok=True)
    if os.path.exists(hist_loc)==False:
        os.makedirs(hist_loc, exist_ok=True)


    test_dic, _ =make_dic(img_list, img_list, args.isize, 'axi',max_shape=max_shape)
    model = Unet_network([*args.isize,1], args.iaxis[0], loss=args.loss, metrics=args.coef, style=args.style, ite=args.ite, depth=args.depth, dim=args.dim, init=args.init, acti=args.acti, lr=args.lr).build()
    callbacks=make_callbacks(args.axi, hist_loc+'/fold'+str(0)+'axi.tsv')
    model.load_weights(args.axi)
    predic_axi = model.predict(test_dic, batch_size=args.bsize)
    predic_axif1 = model.predict(test_dic[:,::-1,:,:], batch_size=args.bsize)
    predic_axif1 = predic_axif1[:,::-1,:,:]
    predic_axif2 = model.predict(axfliper(test_dic), batch_size=args.bsize)
    predic_axif2 = axfliper(predic_axif2,1)
    predic_axif3 = model.predict(axfliper(test_dic[:,::-1,:,:]), batch_size=args.bsize)
    predic_axif3 = axfliper(predic_axif3[:,::-1,:,:],1)
    del model, test_dic, callbacks
    reset_gpu()

    test_dic, _ =make_dic(img_list, img_list, args.isize, 'cor',max_shape=max_shape)
    model = Unet_network([*args.isize,1], args.iaxis[1], loss=args.loss, metrics=args.coef, style=args.style, ite=args.ite, depth=args.depth, dim=args.dim, init=args.init, acti=args.acti, lr=args.lr).build()
    callbacks=make_callbacks(args.cor, hist_loc+'/fold'+str(0)+'cor.tsv')
    model.load_weights(args.cor)

    predic_cor = model.predict(test_dic, batch_size=args.bsize)
    predic_corf1 = model.predict(test_dic[:,:,::-1,:], batch_size=args.bsize)
    predic_corf1 = predic_corf1[:,:,::-1,:]
    predic_corf2 = model.predict(cofliper(test_dic), batch_size=args.bsize)
    predic_corf2 = cofliper(predic_corf2,1)
    predic_corf3 = model.predict(cofliper(test_dic[:,:,::-1,:]), batch_size=args.bsize)
    predic_corf3 = cofliper(predic_corf3[:,:,::-1,:],1)

    del model, test_dic, callbacks
    reset_gpu()


    test_dic, _ =make_dic(img_list, img_list, args.isize, 'sag',max_shape=max_shape)
    model = Unet_network([*args.isize, 1], args.iaxis[2], loss=args.loss, metrics=args.coef, style=args.style, ite=args.ite, depth=args.depth, dim=args.dim, init=args.init, acti=args.acti, lr=args.lr).build()
    callbacks=make_callbacks(args.sag, hist_loc+'/fold'+str(0)+'sag.tsv')
    model.load_weights(args.sag)
    predic_sag = model.predict(test_dic, batch_size=args.bsize)
    predic_sagf1 = model.predict(test_dic[:,::-1,:,:], batch_size=args.bsize)
    predic_sagf1 = predic_sagf1[:,::-1,:,:]
    predic_sagf2 = model.predict(test_dic[:,:,::-1,:], batch_size=args.bsize)
    predic_sagf2 = predic_sagf2[:,:,::-1,:]
    del model, test_dic, callbacks
    reset_gpu()

    predic_sag = np.stack((predic_sag[...,0], predic_sag[...,1], predic_sag[...,1],
                                predic_sag[...,2], predic_sag[...,2], predic_sag[...,3],
                                predic_sag[...,3]),axis=-1)
    predic_sagf1 = np.stack((predic_sagf1[...,0], predic_sagf1[...,1], predic_sagf1[...,1],
                                predic_sagf1[...,2], predic_sagf1[...,2], predic_sagf1[...,3],
                                predic_sagf1[...,3]),axis=-1)
    predic_sagf2 = np.stack((predic_sagf2[...,0], predic_sagf2[...,1], predic_sagf2[...,1],
                                predic_sagf2[...,2], predic_sagf2[...,2], predic_sagf2[...,3],
                                predic_sagf2[...,3]),axis=-1)

    import nibabel as nib
    predic_axi = return_shape(img_list, predic_axi, 'axi')
    predic_axif1 = return_shape(img_list, predic_axif1, 'axi')
    predic_axif2 = return_shape(img_list, predic_axif2, 'axi')
    predic_axif3 = return_shape(img_list, predic_axif3, 'axi')
    predic_cor = return_shape(img_list, predic_cor, 'cor')
    predic_corf1 = return_shape(img_list, predic_corf1, 'cor')
    predic_corf2 = return_shape(img_list, predic_corf2, 'cor')
    predic_corf3 = return_shape(img_list, predic_corf3, 'cor')
    predic_sag = return_shape(img_list, predic_sag, 'sag')
    predic_sagf1 = return_shape(img_list, predic_sagf1, 'sag')
    predic_sagf2 = return_shape(img_list, predic_sagf2, 'sag')
    argmax_sum(img_list, args.result_loc, '', predic_axi, predic_axif1, predic_axif2, predic_axif3, predic_cor, predic_corf1, predic_corf2, predic_corf3, predic_sag, predic_sagf1, predic_sagf2)
    predic_final = predic_axi+predic_axif1+predic_axif2+predic_axif3+predic_cor+predic_corf1+predic_corf2+predic_corf3+predic_sag+predic_sagf1+predic_sagf2

    
 
    if np.shape(img_list):
        
       for i in range(0,len(img_list)):
            img = nib.load(img_list[i])
            new_img = nib.Nifti1Image(np.argmax(predic_final[i],axis=-1), img.affine, img.header)
            filename=img_list[i].split('/')[-1:][0].split('.nii')[0]
            filename_complete=filename+'_deep_agg_sp.nii.gz'
            savedloc = args.out+str(filename_complete)
            nib.save(new_img, savedloc)
            relabel(savedloc,args.ilabel,args.olabel)
            if args.merge:
                filename_complete=filename+'_deep_agg_merged.nii.gz'
                savedloc = args.out+str(filename_complete)
                nib.save(new_img, savedloc)
                relabel(savedloc,[1,2,5,6,4,3,4,5],[161,160,1,42,4,5,161,160])
            make_verify(img_list[i],savedloc,args.out)  
            if args.cp_seg:   
                recon_cp, header =medpy.io.load(args.cp_seg)
                recon_sp, header =medpy.io.load(filename_complete)
                
                for i in range(recon_sp.shape[0]):
                    for j in range(recon_sp.shape[1]):
                        for k in range (recon_sp.shape[2]):
                            if recon_cp[i,j,k]==1.0 and recon_sp[i,j,k]!=1.0:
                                recon_sp[i,j,k]=1.0 
                            elif recon_cp[i,j,k]==42.0 and recon_sp[i,j,k]!=42.0:
                                recon_sp[i,j,k]=42.0   
                            elif recon_cp[i,j,k]==0.0 and recon_sp[i,j,k]!=0.0:
                                recon_sp[i,j,k]=0.0                                
                            elif recon_sp[i,j,k]==5.0 and recon_cp[i,j,k]==0.0:
                                recon_sp[i,j,k]=0.0
                            elif recon_cp[i,j,k]==160.0 and recon_sp[i,j,k]==161.0:
                                recon_sp[i,j,k]=160.0
                            elif recon_cp[i,j,k]==161.0 and recon_sp[i,j,k]==160.0:
                                recon_sp[i,j,k]=161.0   
                            elif recon_cp[i,j,k]==160.0 and recon_sp[i,j,k]!=160.0:
                                recon_sp[i,j,k]=4.0
                            elif recon_cp[i,j,k]==161.0 and recon_sp[i,j,k]!=161.0:
                                recon_sp[i,j,k]=5.0                      
                            elif recon_sp[i,j,k]==42.0 and recon_cp[i,j,k]==0.0:
                                recon_sp[i,j,k]=42.0   
                            elif recon_sp[i,j,k]==4.0 and recon_cp[i,j,k]==0:
                                recon_sp[i,j,k]=0.0
                            elif recon_sp[i,j,k]==4.0 and recon_cp[i,j,k]==160.0:
                                recon_sp[i,j,k]=160.0   
                            elif recon_sp[i,j,k]==5.0 and recon_cp[i,j,k]==161.0:
                                recon_sp[i,j,k]=161.0 
                                
                            else:
                                recon_sp[i,j,k]= recon_sp[i,j,k]
                
                new_sp = nib.Nifti1Image(recon_sp,img.affine, img.header)
                nib.save(new_sp, "_deep_agg_sp.nii.gz")       
                os.remove(savedloc)
                os.remove(args.out+str(filename)+str('_deep_agg_sp_verify.png'))     
    else:
        img = nib.load(img_list)
        new_img = nib.Nifti1Image(np.argmax(predic_final,axis=-1), img.affine, img.header)
        filename=img_list.split('/')[-1:][0].split('.nii')[0]
        savedloc=args.out+filename+'_deep_agg.nii.gz'
        nib.save(new_img, savedloc)
        relabel(savedloc,args.ilabel,args.olabel)
        if args.merge:
            filename_complete=filename+'_deep_agg_merged.nii.gz'
            savedloc = args.out+str(filename_complete)
            nib.save(new_img, savedloc)
            relabel(savedloc,[1,2,5,6,4,3,4,5],[160,161,42,1,5,4,160,161])
        make_verify(img_list, savedloc,args.out)
    return 0          


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
