from past.utils import old_div
import tensorflow as tf
import cv2
import numpy as np
import tensorflow_addons as tfa

def raise_red(image,power):
    
        
        #image = np.array(image*255,dtype=np.dtype('uint8'))
        identityB = np.arange(256, dtype=np.dtype('uint8'))
        
        identityR = np.array([((old_div(i, 255.0)) ** power) * 255
                              for i in np.arange(0, 256)]).astype("uint8")
        
        identityG = np.arange(256, dtype=np.dtype('uint8'))
        
        lut = np.dstack((identityB, identityG, identityR))
        
        
        # apply gamma correction using the lookup table
        
        image = cv2.LUT(image, lut)
        
        return(image)
        #return np.array(image/255,dtype=np.dtype('float32'))
    
def raise_blue(image,power):
    
   
        #image = np.array(image*255,dtype=np.dtype('uint8'))
        identityB = np.array([((old_div(i, 255.0)) ** power) * 255
                              for i in np.arange(0, 256)]).astype("uint8")
        
        identityR = np.arange(256, dtype=np.dtype('uint8'))
        
        identityG = np.arange(256, dtype=np.dtype('uint8'))
        
        lut = np.dstack((identityB, identityG, identityR))
        

        # apply gamma correction using the lookup table
        
        image = cv2.LUT(image, lut)
        
        return(image)
        #return np.array(image/255,dtype=np.dtype('float32'))
    
def raise_green(image,power):
    
        
        #image = np.array(image*255,dtype=np.dtype('uint8'))
        identityB = np.arange(256, dtype=np.dtype('uint8'))
        
        identityR = np.arange(256, dtype=np.dtype('uint8'))
        identityG = np.array([((old_div(i, 255.0)) ** power) * 255
                              for i in np.arange(0, 256)]).astype("uint8")
        
        lut = np.dstack((identityB, identityG, identityR))
        
        
        # apply gamma correction using the lookup table
        
        image = cv2.LUT(image, lut)
        
        return(image)
        #return np.array(image/255,dtype=np.dtype('float32'))

class CustomAugment(object):
    
    
    def _call_on_one_sample(self,sample):
        sample = self._random_apply(tf.image.flip_left_right, sample, p=0.5)
        
        sample = self._random_apply(self.augment_color,sample, p=0.8)
        sample = self._random_apply(self._color_jitter, sample, p=0.8)
        sample = self._random_apply(self.random_crop, sample, p=0.8)
        sample = tf.image.resize(sample,(224,224))
        sample = self._random_apply(self.random_blur, sample, p=0.5)
        sample = self._random_apply(self._color_drop, sample, p=0.2)

        return sample

    def __call__(self, samples):        
        # Random flips
        
        result = tf.map_fn(lambda sample: self._call_on_one_sample(sample), samples)

        return result
        

    def _color_jitter(self, x, s=1):
        # one can also shuffle the order of following augmentations
        # each time they are applied.
        x = tf.image.random_brightness(x, max_delta=0.1*s)
        x = tf.image.random_contrast(x, lower=1-0.2*s, upper=1+0.2*s)
        x = tf.image.random_saturation(x, lower=1-0.2*s, upper=1+0.2*s)
        x = tf.image.random_hue(x, max_delta=0.3*s)
        x = tf.clip_by_value(x, 0, 1)
        return x
    
    def _color_drop(self, x):
        x = tf.image.rgb_to_grayscale(x)
        x = tf.tile(x, [1, 1, 3])
        return x
    
    def _random_apply(self, func, x, p):
        return tf.cond(
          tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                  tf.cast(p, tf.float32)),
          lambda: func(x),
          lambda: x)
    
    def random_crop(self,x,size=(180,180,3)):
        
        x = tf.image.random_crop(x, size)
        return x
    
    def random_suffle(self,x):
        x = self.shuffler.augment(image=x.numpy())
        return tf.convert_to_tensor(x)
    
    def augment_color(self,img):
        blue = np.random.randint(8,13)/10
        red = np.random.randint(8,13)/10
        green = np.random.randint(8,13)/10

        img =  np.array(img*255,dtype=np.dtype('uint8'))
        img_color = raise_green(np.array(img),green)
        img_color = raise_blue(np.array(img_color),blue)
        img_color = raise_red(np.array(img_color),red)

        img_color =  np.array(img_color/255,dtype=np.dtype('float32'))
        return(img_color)
        
    def random_blur(self,x):
        x= tfa.image.gaussian_filter2d(x,(1,1))
        return(x)
