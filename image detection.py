import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
from keras.models import load_model
import glob
import os
import pandas as pd
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


image_data = "C:\Users\dorot\OneDrive\Desktop\rodent"
data_files = [i for i in glob.glob(image_data + '//*//*')]
np.random.shuffle(data_files)
Rodent_labels = [os.path.dirname(i).split('/')[-1] for i in data_files]
Rodent_data = zip(data_files, Rodent_labels)
Rodent_df = pd.DataFrame(Rodent_data, columns=["Image", 'Label'])


train_df, test_df = train_test_split(Rodent_df, train_size=0.8, shuffle=True, random_state=123)

batch_size = 8
target_size = (224, 224)
validation_split = 0.2
test_split = 0.1

tr_gen = ImageDataGenerator()
ts_gen = ImageDataGenerator()

train_gen = tr_gen.flow_from_dataframe(train_df, x_col='Image', y_col='Label', target_size=target_size,
                                       class_mode='categorical',
                                       color_mode='rgb', shuffle=True, batch_size=batch_size)

test_gen = ts_gen.flow_from_dataframe(test_df, x_col='Image', y_col='Label', target_size=target_size,
                                       class_mode='categorical',
                                       color_mode='rgb', shuffle=True, batch_size=batch_size)


# load the trained model
MODEL_PATH = "C:\Users\dorot\Downloads\EfficientNet_B01.h5"  # Update this with your model's path
model = load_model(MODEL_PATH)
#
model.compile(optimizer = "Adam" , loss = "categorical_crossentropy" , metrics  = ["accuracy"])

# # Fitting model
#
History = model.fit(train_gen, epochs= 12, verbose= 1,
                    validation_data = test_gen, shuffle= False)

# Function to preprocess the uploaded image
def preprocess_image(img):
	img = img.resize((224, 224))  # Adjust size based on your model's input
	img_array = np.array(img)
	img_array = np.expand_dims(img_array, axis=0)
	img_array = img_array / 255.0  # Normalize
	return img_array


# Streamlit app
st.title("Deep Learning Image Classification")
st.write("Upload an image to classify using the trained deep learning model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
	# Display the uploaded image
	st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
	st.write("Processing...")
	
	# Preprocess and predict
	img = Image.open(uploaded_file)
	img_array = preprocess_image(img)
	
	prediction = model.predict(img_array)
	predicted_class = np.argmax(prediction, axis=1)[0]
	confidence = np.max(prediction)
	
	if predicted_class == 0:
		predicted_class = 'Mice'
	elif predicted_class == 1:
		predicted_class = 'Rat'
	else:
		predicted_class = 'Shrew'
	# Display prediction
	st.write(f"**Predicted Class:** {predicted_class}")
	st.write(f"**Confidence:** {confidence:.2f}")


#Run this on your terminal
# streamlit run "C:\Users\dorot\OneDrive\Desktop\image detection.py"