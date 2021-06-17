import tensorflow as tf
import numpy as np
from monk import BBox, Dataset


### DAMAGED

class DamagedDataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self,dataset, batch_size=32, dim=(128,128), n_channels=3,shuffle=True,to_keep=[],area_threshold=2000):
          
        self.dataset=dataset
        self.dim = dim ###
        self.batch_size = batch_size  ##
        self.list_IDs = np.arange(len(dataset)) ###
        self.n_channels = n_channels ##
        self.shuffle = shuffle ##
        self.to_keep=to_keep
        self.area_threshold=area_threshold

        
        self.get_map_id()
        self.on_epoch_end()
        
        #self.labels = labels
        #self.n_classes = n_classes
        
    def get_map_id(self):
        map_id ={}
        i=0

        for _,imds in enumerate(self.dataset):

            parts=[]
            to_keep = []

            for ind,poly in enumerate(imds.anns['polygons']):
                att = poly.attributes
                label =att["part_label"]
                if (not label in parts and label in self.to_keep and poly.area>self.area_threshold):
                    to_keep.append(ind)
                    parts.append(label)

            for ind in to_keep:
                map_id[i]=[imds.id,ind]
                i+=1
        
        self.map_id = map_id
        
    def load_image(self,ids):
        imds = self.dataset[ids[0]]
        ann = imds.anns["polygons"][ids[1]]
        att =ann.attributes
        label =att["part_label"]
       
        img_crop = imds.image.crop(BBox(xyxy=[att["x1_part"],att["y1_part"],att["x2_part"],att["y2_part"],])).resize(self.dim)
       
        return(img_crop.rgb)
        
    def __len__(self):
       
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
    
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels),dtype=int)
        #y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self.load_image(self.map_id[ID])

            # Store class
            #y[i] = self.labels[ID]

        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X


def get_damaged_generator(path, batch_size=3, dim=(128,128), to_keep=[]):
    dataset = Dataset.from_coco(path,"")
    gen = DamagedDataGenerator(dataset,batch_size=batch_size,dim=dim,to_keep=to_keep)

    return(gen)


## NON_DAMAGED



def my_to_bbox(polygon, allow_unsafe=False):
        """ Get the smallest BBox encompassing the polygon """
        xmin, ymin = np.inf, np.inf
        xmax, ymax = -np.inf, -np.inf
        for subpol in polygon.points:
            xmin = min(xmin, *subpol[:, 0])
            xmax = max(xmax, *subpol[:, 0])
            ymin = min(ymin, *subpol[:, 1])
            ymax = max(ymax, *subpol[:, 1])
            
        xmin = max(0,xmin)
        ymin = max(0,ymin)
        
        return BBox(
            label=polygon.label,
            image_size=polygon.image_size,
            xyxy=[xmin, ymin, xmax, ymax],
            allow_unsafe=allow_unsafe,
            attributes=polygon.attributes.copy(),
        )

class NonDamagedDataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self,dataset, batch_size=3, dim=(128,128), n_channels=3,shuffle=True):
          
        self.dataset=dataset
        self.dim = dim ###
        self.batch_size = batch_size  ##
        self.list_IDs = np.arange(len(dataset)) ###
        self.n_channels = n_channels ##
        self.shuffle = shuffle ##
        
        self.get_map_id()
        self.on_epoch_end()
        
        #self.labels = labels
        #self.n_classes = n_classes
        
    def get_map_id(self):
        i=0
        map_id ={}
        for _,imds in enumerate(self.dataset):
            for poly_id,poly in enumerate(imds.anns["polygons"]):
                
                if(poly.area>self.area_threshold):
                    map_id[i]=[0,imds.id,poly_id]
                    i+=1
               
        self.map_id = map_id
        
    def load_image(self,ids):
        imds = self.dataset[ids[0]]
        ann = imds.anns["polygons"][ids[1]]
        att =ann.attributes
        img_crop = imds.image.crop(my_to_bbox(ann)).resize(self.dim)
       
        return(img_crop.rgb)
        
    def __len__(self):
       
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
    
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels),dtype=int)
        #y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample

            X[i,] = self.load_image(self.map_id[ID])

            # Store class
            #y[i] = self.labels[ID]

        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X

def get_non_damaged_generator(path, batch_size=3, dim=(128,128), to_keep=[]):
    dataset_non_damaged = Dataset.from_coco(path)
    dataset_non_damaged = dataset_non_damaged.filter_images_with_cats(keep=to_keep).filter_cats(keep=to_keep)
    gen = NonDamagedDataGenerator(dataset_non_damaged,batch_size=batch_size,dim=dim)

    return(gen)

## MIXED

class MixedDataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self,dataset_damages,dataset_parts, batch_size=32, dim=(128,128), n_channels=3,shuffle=True,to_keep=[],area_threshold=2000):
          
        self.dataset_parts=dataset_parts
        self.dataset_damages=dataset_damages
        self.dim = dim 
        self.batch_size = batch_size  
        self.list_IDs = np.arange(len(dataset_damages)+len(dataset_parts)) 
        self.n_channels = n_channels 
        self.shuffle = shuffle 
        self.to_keep =to_keep
        self.area_threshold=area_threshold

    

        self.get_map_id()
        self.on_epoch_end()
        
        #self.labels = labels
        #self.n_classes = n_classes
        
    def get_map_id(self):
        i=0
        map_id ={}
        
        for _,imds in enumerate(self.dataset_damages):
            parts=[]
            to_keep = []

            for ind,poly in enumerate(imds.anns['polygons']):
                att = poly.attributes

                label =att["part_label"]
                if self.to_keep =='all':
                    if (not label in parts and poly.area>self.area_threshold):
                        to_keep.append(ind)
                        parts.append(label)
                else:
                    if (not label in parts and label in self.to_keep and poly.area>self.area_threshold):
                        to_keep.append(ind)
                        parts.append(label)

            for ind in to_keep:
                map_id[i]=[1,imds.id,ind]
                i+=1
                
        for _,imds in enumerate(self.dataset_parts):
            for poly_id,poly in enumerate(imds.anns["polygons"]):
                
                if(poly.area>self.area_threshold):
                    map_id[i]=[0,imds.id,poly_id]
                    i+=1
        
        self.map_id = map_id
        
    def load_image(self,ids):

        if ids[0]==1:
            

            
            imds = self.dataset_damages[ids[1]]
            poly = imds.anns["polygons"][ids[2]]
            
            att = poly.attributes
            label =att["part_label"]
            
    
            img_crop = imds.image.crop(BBox(xyxy=[att["x1_part"],att["y1_part"],att["x2_part"],att["y2_part"],])).resize(self.dim)

            
        
        if ids[0]==0:
          
            imds = self.dataset_parts[ids[1]]
            poly = imds.anns["polygons"][ids[2]]

            att =poly.attributes
     


            img_crop = imds.image.crop(my_to_bbox(poly)).resize(self.dim)
        
        
        
        img = np.array(img_crop.rgb/255,dtype=np.dtype('float32'))
        
        return ( img )
        
    def __len__(self):
       
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
    
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels),dtype=np.float32)
        #y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
   
            X[i,] = self.load_image(self.map_id[ID])

            # Store class
            #y[i] = self.labels[ID]

        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return tf.convert_to_tensor(X,dtype=tf.float32)

def get_mixed_generator(path_damaged,path_non_damaged, batch_size=3, dim=(128,128), to_keep='all',area_threshold=2000):
    dataset_damaged = Dataset.from_coco(path_damaged,"")
    dataset_non_damaged = Dataset.from_coco(path_non_damaged)
    if (to_keep!='all'):
        dataset_non_damaged = dataset_non_damaged.filter_images_with_cats(keep=to_keep).filter_cats(keep=to_keep)
    gen = MixedDataGenerator(dataset_damages=dataset_damaged,dataset_parts=dataset_non_damaged,batch_size=batch_size,dim=dim,to_keep=to_keep,area_threshold=area_threshold)
    return(gen)