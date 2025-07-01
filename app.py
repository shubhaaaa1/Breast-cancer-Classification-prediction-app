import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import load_breast_cancer

model = pickle.load(open('model.pkl','rb'))

data = load_breast_cancer()
feature_names = data.feature_names
df = pd.DataFrame(data.data, columns=feature_names)
target_names = data.target_names


st.set_page_config(page_title="Classify your Breast Cancer !",layout="wide")
st.title("Binary Classifier Of Breast Cancer ")
st.markdown("Find Out whether U have a Malignant tumor or Benign Tumor")
st.markdown('---')


if st.checkbox('Show Dataset Preview'):
    st.write(df.head())
    st.write('Feature Statistics:')
    st.write(df.describe())

# User Guidance
st.info('Please enter the feature values based on the above dataset statistics.')

col,col1,col2 = st.columns(3)

with col:
    feature1 = st.number_input('Enter Feature 1 as per previewed dataset',min_value=0.0,step=0.1)
    feature2 = st.number_input('Enter Feature 2 as per previewed dataset',min_value=0.0,step=0.1)
    feature3 = st.number_input('Enter Feature 3 as per previewed dataset',min_value=0.0,step=0.1)
    feature4 = st.number_input('Enter Feature 4 as per previewed dataset',min_value=0.0,step=0.1)
    feature5 = st.number_input('Enter Feature 5 as per previewed dataset',min_value=0.0,step=0.1)
    feature6 = st.number_input('Enter Feature 6 as per previewed dataset',min_value=0.0,step=0.1)
    feature7 = st.number_input('Enter Feature 7 as per previewed dataset',min_value=0.0,step=0.1)
    feature8 = st.number_input('Enter Feature 8 as per previewed dataset',min_value=0.0,step=0.1)
    feature9 = st.number_input('Enter Feature 9 as per previewed dataset',min_value=0.0,step=0.1)
    feature10 = st.number_input('Enter Feature 10 as per previewed dataset',min_value=0.0,step=0.1)

with col1:
    feature11 = st.number_input('Enter Feature 11 as per previewed dataset',min_value=0.0,step=0.1)
    feature12 = st.number_input('Enter Feature 12 as per previewed dataset',min_value=0.0,step=0.1)
    feature13 = st.number_input('Enter Feature 13 as per previewed dataset',min_value=0.0,step=0.1)
    feature14 = st.number_input('Enter Feature 14 as per previewed dataset',min_value=0.0,step=0.1)
    feature15 = st.number_input('Enter Feature 15 as per previewed dataset',min_value=0.0,step=0.1)
    feature16 = st.number_input('Enter Feature 16 as per previewed dataset',min_value=0.0,step=0.1)
    feature17 = st.number_input('Enter Feature 17 as per previewed dataset',min_value=0.0,step=0.1)
    feature18 = st.number_input('Enter Feature 18 as per previewed dataset',min_value=0.0,step=0.1)
    feature19 = st.number_input('Enter Feature 19 as per previewed dataset',min_value=0.0,step=0.1)
    feature20 = st.number_input('Enter Feature 20 as per previewed dataset',min_value=0.0,step=0.1)

with col2:
    feature21 = st.number_input('Enter Feature 21 as per previewed dataset',min_value=0.0,step=0.1)
    feature22 = st.number_input('Enter Feature 22 as per previewed dataset',min_value=0.0,step=0.1)
    feature23 = st.number_input('Enter Feature 23 as per previewed dataset',min_value=0.0,step=0.1)
    feature24 = st.number_input('Enter Feature 24 as per previewed dataset',min_value=0.0,step=0.1)
    feature25 = st.number_input('Enter Feature 25 as per previewed dataset',min_value=0.0,step=0.1)
    feature26 = st.number_input('Enter Feature 26 as per previewed dataset',min_value=0.0,step=0.1)
    feature27 = st.number_input('Enter Feature 27 as per previewed dataset',min_value=0.0,step=0.1)
    feature28 = st.number_input('Enter Feature 28 as per previewed dataset',min_value=0.0,step=0.1)
    feature29 = st.number_input('Enter Feature 29 as per previewed dataset',min_value=0.0,step=0.1)
    feature30 = st.number_input('Enter Feature 30 as per previewed dataset',min_value=0.0,step=0.1)


if st.button('Predict'):
    features = (feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9,feature10,
                feature11,feature12,feature13,feature14,feature15,feature16,feature17,feature18,feature19,feature20,
                feature21,feature22,feature23,feature24,feature25,feature26,feature27,feature28,feature29,feature30)

    numpy_array = np.asarray(features)
    reshaped = numpy_array.reshape(1, -1)
    prediction = model.predict(reshaped)

    if prediction[0] == 1:
        st.success("The tumor is Benign.")
    else:
        st.error("The tumor is Malignant.")


st.markdown('---')
st.caption('Developed by Shubham Agnihotri | Breast Cancer Prediction System')

