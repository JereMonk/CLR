import tensorflow as tf

from losses import _dot_simililarity_dim1 as sim_func_dim1
from losses import _dot_simililarity_dim2 as sim_func_dim2

#from tqdm import tqdm
import numpy as np


def get_negative_mask(batch_size):
    # return a mask that removes the similarity score of equal/similar images.
    # this function ensures that only distinct pair of images get their similarity scores
    # passed as negative examples
    negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0
    return tf.constant(negative_mask)


@tf.function
def train_step(xis, xjs, model, optimizer, criterion, temperature,BATCH_SIZE,negative_mask):
    with tf.GradientTape() as tape:
        zis = model(xis)
        zjs = model(xjs)

        # normalize projection feature vectors
        zis = tf.math.l2_normalize(zis, axis=1)
        zjs = tf.math.l2_normalize(zjs, axis=1)

        l_pos = sim_func_dim1(zis, zjs)
        l_pos = tf.reshape(l_pos, (BATCH_SIZE, 1))
        l_pos /= temperature

        negatives = tf.concat([zjs, zis], axis=0)

        loss = 0

        for positives in [zis, zjs]:
            l_neg = sim_func_dim2(positives, negatives)

            labels = tf.zeros(BATCH_SIZE, dtype=tf.int32)

            l_neg = tf.boolean_mask(l_neg, negative_mask)
            l_neg = tf.reshape(l_neg, (BATCH_SIZE, -1))
            l_neg /= temperature

            logits = tf.concat([l_pos, l_neg], axis=1) 
            loss += criterion(y_pred=logits, y_true=labels)

        loss = loss / (2 * BATCH_SIZE)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss



def train_simclr(model, dataset, optimizer, criterion,data_augmentation,temperature=0.1,start_iter=0,max_iter=50000,dir_path='',ckpt_freq=500,batch_size=2,negative_mask=[]):
    
    step_wise_loss = []
    #epoch_wise_loss = []
   
    step_counter=start_iter
    writer = tf.summary.create_file_writer(dir_path)

    while(step_counter<max_iter):
    
        print("\nStart of iter %d" % (step_counter,))
        for image_batch in dataset:
            step_counter+=1
            a = data_augmentation(image_batch)
            b = data_augmentation(image_batch)

            loss = train_step(a, b, model, optimizer, criterion, temperature,batch_size,negative_mask)
            step_wise_loss.append(loss)

            
            if step_counter%ckpt_freq ==0:
                    model.save_weights(dir_path+"/ckpt"+str(step_counter))

            if step_counter %1 == 0:
                final_loss =np.mean(step_wise_loss)
                print("step: {} loss: {:.3f}".format(step_counter + 1, final_loss))
                step_wise_loss=[]
                with writer.as_default():
                        tf.summary.scalar('loss', final_loss, step=step_counter)
                        tf.summary.scalar('learning rate', optimizer.lr, step=step_counter)
