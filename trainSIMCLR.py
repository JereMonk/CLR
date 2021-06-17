
import tensorflow as tf
import yaml
from trainer import train_simclr
from datasets import get_mixed_generator
from model import get_resnet_simclr
import sys 
from augmenter import CustomAugment
from losses import get_negative_mask

def main(arg):


    EXP_FOLDER= arg[0]
    print(arg[0])

    

    with open(EXP_FOLDER+"/custom.yaml", 'r') as stream:
        custom_data = yaml.safe_load(stream)

    IMS_PER_BATCH = int(custom_data['SOLVER']['IMS_PER_BATCH'])
    STEPS = int(custom_data['SOLVER']['STEPS'])
    MAX_ITER = int(custom_data['SOLVER']['MAX_ITER'])
    TEMPERATURE = int(custom_data['LOSS']['TEMPERATURE'])
    DATASET_DAMAGED = custom_data['DATASET']['DAMAGED']
    DATASET_NON_DAMAGED = custom_data['DATASET']['NON_DAMAGED']
    INPUT_DIM = int(custom_data['INPUT']['DIM'])
    AREA_THRESHOLD =  custom_data['DATASET']['AREA_THRESHOLD']
    SUBCATS = custom_data['SUBCATS']
    CHECKPOINT_PERIOD = int(custom_data['CHECKPOINT_PERIOD'])
    CROP_SIZE = int(custom_data['AUGMENTATION']['CROP_SIZE'])
    # GET DATA

    generator = get_mixed_generator(DATASET_DAMAGED,DATASET_NON_DAMAGED, batch_size=IMS_PER_BATCH, dim=(INPUT_DIM,INPUT_DIM), to_keep=SUBCATS,area_threshold=AREA_THRESHOLD)

    # GET MODEL

    model = get_resnet_simclr(hidden_1=256, hidden_2=128,input_size=(INPUT_DIM,INPUT_DIM,3))

    # GET AUGMENTATION

    data_augmentation = tf.keras.Sequential([tf.keras.layers.Lambda(CustomAugment(crop_size=CROP_SIZE,input_dim=INPUT_DIM))])

    # GET NEGATIVE MASKS

    negative_mask = get_negative_mask(IMS_PER_BATCH)

    # OPTIM
    lr  = 0.3*IMS_PER_BATCH/256
    print('learning rate', lr)
    lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=lr, decay_steps=STEPS)
    optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
    train_simclr(model=model, dataset=generator, optimizer=optimizer, criterion=criterion,data_augmentation=data_augmentation,temperature=TEMPERATURE,start_iter=0,max_iter=MAX_ITER,dir_path=EXP_FOLDER,ckpt_freq=CHECKPOINT_PERIOD,batch_size=IMS_PER_BATCH,negative_mask=negative_mask)

if __name__ == "__main__":
    
    main(sys.argv[1:])
