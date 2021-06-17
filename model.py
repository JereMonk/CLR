import tensorflow as tf



def build_resnet_simclr(resnet,hidden_1, hidden_2,input_size=(224,224,3)):
    base_model = resnet
    base_model.trainable = True
    inputs = tf.keras.layers.Input(input_size)
    h = base_model(inputs, training=True)
    h = tf.keras.layers.GlobalAveragePooling2D()(h)

    projection_1 = tf.keras.layers.Dense(hidden_1)(h)
    projection_1 = tf.keras.layers.Activation("relu")(projection_1)
    projection_2 = tf.keras.layers.Dense(hidden_2)(projection_1)

    model = tf.keras.Model(inputs, projection_2)

    return model

def get_resnet_simclr(hidden_1=256, hidden_2=128,input_size=(224,224,3)):
    resnet = tf.keras.applications.ResNet50(include_top=False,weights="imagenet",)
    resnet_simclr = build_resnet_simclr(resnet,hidden_1, hidden_2,input_size=input_size)

    return(resnet_simclr)