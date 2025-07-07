import tensorflow as tf
from tensorflow.keras.utils import plot_model

# Load your trained model
model = tf.keras.models.load_model("steel_defect_model.h5")

# Plot the model architecture
plot_model(
    model,
    to_file="cnn_architecture.png",
    show_shapes=True,
    show_layer_names=True,
    dpi=96,
    expand_nested=True,
)

print("✅ CNN model architecture image saved as cnn_architecture.png")
