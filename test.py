from keras.models import load_model

# Load the model
model = load_model('music_final.h5')

# Display the summary of the model
model.summary()

# Get information about the input layer
input_layer = model.layers[0]  # Assuming the input layer is the first layer

# Print input layer information
print("Input Layer Information:")
print("Input Shape:", input_layer.input_shape)
print("Data Type:", input_layer.input.dtype)
