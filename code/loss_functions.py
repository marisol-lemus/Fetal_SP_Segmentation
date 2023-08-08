from joblib import PrintTime
from turtle import shape
import imp
from builtins import print
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf

# Helper function to enable loss function to be flexibly used for 
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn
def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [1,2,3]
    # Two dimensional
    elif len(shape) == 4 : 
        #print("here shape=4")
        return [1,2]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')

################################
#           Dice loss          #
################################
def dice_loss(delta = 0.5, smooth = 0.000001):
    """Dice loss originates from Sørensen–Dice coefficient, which is a statistic developed in 1940s to gauge the similarity between two samples.
    
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.5
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """
    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1-y_pred), axis=axis)
        fp = K.sum((1-y_true) * y_pred, axis=axis)
        # Calculate Dice score
        dice_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Average class scores
        print('shape dice is : ' + str(dice_class.shape))
        dice_loss = K.mean(1-dice_class)
        return dice_loss
        
    return loss_function


################################
#         Tversky loss         #
################################
def tversky_loss(delta = 0.7, smooth = 0.000001):
    """Tversky loss function for image segmentation using 3D fully convolutional deep networks
	Link: https://arxiv.org/abs/1706.05721
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """
    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)   
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1-y_pred), axis=axis)
        fp = K.sum((1-y_true) * y_pred, axis=axis)
        tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Average class scores
        tversky_loss = K.mean(1-tversky_class)
        return tversky_loss

    return loss_function

################################
#       Dice coefficient       #
################################
def dice_coefficient(delta = 0.5, smooth = 0.000001):
    """The Dice similarity coefficient, also known as the Sørensen–Dice index or simply Dice coefficient, is a statistical tool which measures the similarity between two sets of data.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.5
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """
    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)   
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1-y_pred), axis=axis)
        fp = K.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Average class scores
        dice = K.mean(dice_class)
        return dice

    return loss_function

################################
#          Combo loss          #
################################
def combo_loss(alpha=0.5,beta=0.5):
    """Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation
    Link: https://arxiv.org/abs/1805.02798
    Parameters
    ----------
    alpha : float, optional
        controls weighting of dice and cross-entropy loss., by default 0.5
    beta : float, optional
        beta > 0.5 penalises false negatives more than false positives., by default 0.5
    """
    def loss_function(y_true,y_pred):
        dice = dice_coefficient()(y_true, y_pred)
        axis = identify_axis(y_true.get_shape())
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        if beta is not None:
            beta_weight = np.repeat([beta,1-beta],[1,y_pred.shape[-1]-1],axis = 0)
            print(beta_weight)
            cross_entropy = beta_weight * cross_entropy
        # sum over classes
        cross_entropy = K.mean(K.sum(cross_entropy, axis=axis))
        if alpha is not None:
            combo_loss = (alpha * cross_entropy) - ((1 - alpha) * dice)
        else:
            combo_loss = cross_entropy - dice
        return combo_loss
    return loss_function

################################
#      Focal Tversky loss      #
################################
def focal_tversky_loss(delta=0.7, gamma=0.75, smooth=0.000001):
    """A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation
    Link: https://arxiv.org/abs/1810.07842
    Parameters
    ----------
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """
    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon) 
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1-y_pred), axis=axis)
        fp = K.sum((1-y_true) * y_pred, axis=axis)
        tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        print('shape tversky class is: ' + str(tversky_class.shape))
        temp = K.pow((1-tversky_class), gamma)
        # Average class scores
        print('shape temp is : ' + str(temp.shape))
        focal_tversky_loss = K.mean(temp)
        print('shape tversky is : ' + str(focal_tversky_loss.shape))
        return focal_tversky_loss

    return loss_function

def focal_loss_2(gamma_f=0.15, alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma_f) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma_f) * K.log(1. - pt_0 + K.epsilon()))
	return focal_loss_fixed
    
def focal_loss_3(gamma=2., alpha=4.):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002
        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]
        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})
        Returns:
            [tensor] -- loss.
        """
        def CrossEntropy(yHat, y):
            if y == 1:
                return -K.log(yHat)
            else:
                return -K.log(1 - yHat)

        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon) 
        axis = identify_axis(y_true.get_shape())
        tmp = tf.multiply(y_true,K.log(y_pred))
        print(tmp.shape)
        ce = -K.sum(tmp,axis=-1)
        ce = K.mean(ce)
        return ce

    return focal_loss_fixed
################################
#          Focal loss          #
################################
def focal_loss(alpha=None, beta=None, gamma_f=2.):
    """Focal loss is used to address the issue of the class imbalance problem. A modulation term applied to the Cross-Entropy loss function.
    Parameters
    ----------
    alpha : float, optional
        controls relative weight of false positives and false negatives. Beta > 0.5 penalises false negatives more than false positives, by default None
    gamma_f : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 2.
    """
    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)

        if alpha is not None:
            alpha_weight = np.repeat([alpha,(1-alpha)/(y_pred.shape[-1]-1)],[1,y_pred.shape[-1]-1],axis = 0)
            focal_loss = alpha_weight * y_true * K.pow(1 - y_pred, gamma_f) * cross_entropy
        else:
            focal_loss = K.pow(1 - y_pred, gamma_f) * cross_entropy

        focal_loss = K.mean(tf.reduce_max(focal_loss, axis=axis))
        return focal_loss
        
    return loss_function


################################
#       Symmetric Focal loss      #
################################
def symmetric_focal_loss(delta=0.7, gamma=2.):
    """
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    """
    def loss_function(y_true, y_pred):

        axis = identify_axis(y_true.get_shape())  

        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        #calculate losses separately for each class
        back_ce = K.pow(1 - y_pred[:,:,:,0], gamma) * cross_entropy[:,:,:,0]
        back_ce =  (1 - delta) * back_ce
        print('back_ce: ',back_ce.shape)
        
        if y_true.shape[-1]==4:
            fore_ce1 = K.pow(1 - y_pred[:,:,:,1], gamma) * cross_entropy[:,:,:,1]
            fore_ce1 = delta * fore_ce1
            fore_ce2 = K.pow(1 - y_pred[:,:,:,2], gamma) * cross_entropy[:,:,:,2]
            fore_ce2 = delta * fore_ce2
            fore_ce3 = K.pow(1 - y_pred[:,:,:,3], gamma) * cross_entropy[:,:,:,3]
            fore_ce3 = delta * fore_ce3
            tmp = tf.stack([back_ce, fore_ce1,fore_ce2,fore_ce3],axis=-1)
            print('tmp base focal: ',tmp.shape)            

            tmp2 = K.sum(tmp,axis=-1)
            print('tmp2 base focal: ',tmp2.shape)            
            loss = K.mean(tmp)

            return loss

    return loss_function
SMOTH = 0.000001
#################################
# Symmetric Focal Tversky loss  #
#################################
def symmetric_focal_tversky_loss(delta=0.7, gamma=0.75):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """
    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1-y_pred), axis=axis)
        fp = K.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + SMOTH)/(tp + delta*fn + (1-delta)*fp + SMOTH)

        #calculate losses separately for each class, enhancing both classes
        back_dice = (1-dice_class[:,0]) * K.pow(1-dice_class[:,0], gamma)
        if y_true.shape[-1]==4:
            fore_dice1 = (1-dice_class[:,1]) * K.pow(1-dice_class[:,1], gamma) 
            fore_dice2 = (1-dice_class[:,2]) * K.pow(1-dice_class[:,2], gamma) 
            fore_dice3 = (1-dice_class[:,3]) * K.pow(1-dice_class[:,3], gamma) 
            # Average class scores
            tmp = tf.stack([back_dice,fore_dice1,fore_dice2,fore_dice3],axis=-1)
            print('tmp base symetric tversky: ',tmp.shape)            
            loss = K.mean(tmp)
            return loss

    return loss_function


################################
#     Asymmetric Focal loss    #
################################
def asymmetric_focal_loss(delta=0.7, gamma=2.):
    def loss_function(y_true, y_pred):
        """For Imbalanced datasets
        Parameters
        ----------
        delta : float, optional
            controls weight given to false positive and false negatives, by default 0.7
        gamma : float, optional
            Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
        """
        axis = identify_axis(y_true.get_shape())  

        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)

        #calculate losses separately for each class, only suppressing background class
        back_ce = K.pow(1 - y_pred[:,:,:,0], gamma) * cross_entropy[:,:,:,0]
        back_ce =  (1 - delta) * back_ce
        print('back_ce: ',back_ce.shape)
        if y_true.shape[-1]==4:
            fore_ce_full1 = delta * cross_entropy[:,:,:,1]
            fore_ce_full2 = delta * cross_entropy[:,:,:,2]
            fore_ce_full3 = delta * cross_entropy[:,:,:,3]
            temp = tf.stack([back_ce,fore_ce_full1,fore_ce_full2,fore_ce_full3],axis=-1)
            print('temp base: ',temp.shape)
            loss = K.mean(K.sum(temp,axis=-1))
            return loss
    return loss_function

#################################
# Asymmetric Focal Tversky loss #
#################################
def asymmetric_focal_tversky_loss(delta=0.7, gamma=0.75):
    """This is the implementation for multiclass segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """
    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1-y_pred), axis=axis)
        fp = K.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)

        #calculate losses separately for each class, only enhancing foreground class
        back_dice = (1-dice_class[:,0])
        if y_true.shape[-1]==7:
            fore_ce_full1 = (1-dice_class[:,1]) * K.pow(1-dice_class[:,1], gamma) 
            fore_ce_full2 = (1-dice_class[:,2]) * K.pow(1-dice_class[:,2], gamma) 
            fore_ce_full3 = (1-dice_class[:,3]) * K.pow(1-dice_class[:,3], gamma) 
            fore_ce_full4 = (1-dice_class[:,4]) * K.pow(1-dice_class[:,4], gamma) 
            fore_ce_full5 = (1-dice_class[:,5]) * K.pow(1-dice_class[:,5], gamma) 
            fore_ce_full6 = (1-dice_class[:,6]) * K.pow(1-dice_class[:,6], gamma) 
            temp = tf.stack([back_dice,fore_ce_full1,fore_ce_full2,fore_ce_full3,fore_ce_full4,fore_ce_full5,fore_ce_full6],axis=-1)
        if y_true.shape[-1]==4:
            fore_ce_full1 = (1-dice_class[:,1]) * K.pow(1-dice_class[:,1], gamma) 
            fore_ce_full2 = (1-dice_class[:,2]) * K.pow(1-dice_class[:,2], gamma) 
            fore_ce_full3 = (1-dice_class[:,3]) * K.pow(1-dice_class[:,3], gamma) 
            temp = tf.stack([back_dice,fore_ce_full1,fore_ce_full2,fore_ce_full3],axis=-1)
               # Average class scores
        loss = K.mean(temp,axis=-1)
        return loss

    return loss_function

###########################################
#      Symmetric Unified Focal loss       #
###########################################
def sym_unified_focal_loss(weight=0.5, delta=0.6, gamma=0.5):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to symmetric Focal Tversky loss and symmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    """
    def loss_function(y_true,y_pred):
      symmetric_ftl = symmetric_focal_tversky_loss(delta=delta, gamma=gamma)(y_true,y_pred)
      symmetric_fl = symmetric_focal_loss(delta=delta, gamma=gamma)(y_true,y_pred)
      if weight is not None:
        return (weight * symmetric_ftl) + ((1-weight) * symmetric_fl)  
      else:
        return symmetric_ftl + symmetric_fl

    return loss_function

###########################################
#      Asymmetric Unified Focal loss      #
###########################################
def asym_unified_focal_loss(weight=0.5, delta=0.6, gamma=0.5):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to asymmetric Focal Tversky loss and asymmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    """
    def loss_function(y_true,y_pred):
      asymmetric_ftl = asymmetric_focal_tversky_loss(delta=delta, gamma=gamma)(y_true,y_pred)
      asymmetric_fl = asymmetric_focal_loss(delta=delta, gamma=gamma)(y_true,y_pred)
      if weight is not None:
        return (weight * asymmetric_ftl) + ((1-weight) * asymmetric_fl)  
      else:
        return asymmetric_ftl + asymmetric_fl

    return loss_function
