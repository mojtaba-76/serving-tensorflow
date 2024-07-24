import tensorflow as tf

# Load the H5 model
model = tf.keras.models.load_model('20240710-033555model_N12.h5')

 
model.save('autoencoder/1', save_format='tf')