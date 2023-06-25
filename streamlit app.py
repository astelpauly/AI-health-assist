import base64
import streamlit as st
import plotly.express as px

df = px.data.iris()

@st.experimental_memo
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("C:/Users/Astel Pauly/Desktop/multipple disease prediciton/mini/navigate.jpg")
bg = get_img_as_base64("C:/Users/Astel Pauly/Desktop/multipple disease prediciton/mini/x-ray.jpg")
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{bg}");
background-size: 160%;
background-position: top right;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: top left; 
background-size: 70%;

background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""


# Custom CSS styles for the title
title_style = """
    <style>
        .title {
            font-size: 64px;
            color: #92e4f1;
            text-align: center;
        }
    </style>
"""

# Render the title with custom styles
st.markdown(title_style, unsafe_allow_html=True)
st.markdown('<h1 class="title">AI HEALTH ASSIST</h1>', unsafe_allow_html=True)



st.markdown(page_bg_img, unsafe_allow_html=True)

import pickle
#from streamlit_option_menu import selectbox


#loading the saved models
diabetes_model = pickle.load(open("C:/Users/Astel Pauly/Desktop/multipple disease prediciton/mini/daibetes_pred/diabetes_model.sav",'rb'))
heart_disease_model = pickle.load(open("C:/Users/Astel Pauly/Desktop/multipple disease prediciton/mini/heart_disease/heart_disease_model.sav",'rb'))
#pneumonia_model = pickle.load(open("C:/Users/Astel Pauly/Desktop/multipple disease prediciton/mini/chest_xray/pneumonia_predictionn.h5"))
#sidebar for navigation
#st.sidebar.header("AI HEALTH ASSIST")
with st.sidebar:
    selected = st.sidebar.radio("AI HEALTH ASSIST",['Home','Diabetes prediction', 'Heart Disease prediction', 'Pneumonia prediction'])
    #default - which page is to be opened while browsing

if(selected=='Home'):

    with st.container():
        st.header("Diabetes Prediction")
        st.markdown(
            "Diabetes is a chronic metabolic disorder characterized by elevated levels of blood glucose (sugar). It occurs when the body either doesn't produce enough insulin (a hormone that regulates blood sugar) or cannot effectively use the insulin it produces. This leads to an imbalance in blood glucose levels, causing hyperglycemia. Diabetes can be classified into different types, including type 1 diabetes, type 2 diabetes, and gestational diabetes. Common symptoms include increased thirst, frequent urination, fatigue, and unexplained weight loss. Proper management of diabetes involves maintaining healthy blood sugar levels through medication, diet control, regular exercise, and monitoring of blood glucose levels.Diabetes prediction using machine learning (ML) involves utilizing algorithms to analyze various risk factors and clinical parameters to assess the likelihood of developing diabetes. ML models, such as decision trees, logistic regression, random forests, and neural networks, learn from patient data to identify patterns and correlations indicative of diabetes. This automated approach aids in early detection, facilitating proactive interventions and personalized preventive measures. However, medical expertise should be consulted for accurate evaluation and diagnosis. "   )
    with st.container():
        st.header("Heart Disease Prediction")
        st.markdown(
            "Heart disease refers to a range of conditions that affect the heart and blood vessels, leading to various complications. It encompasses conditions such as coronary artery disease, heart failure, arrhythmias, and heart valve problems. Heart disease can result from factors like high blood pressure, high cholesterol, smoking, obesity, diabetes, and a sedentary lifestyle. It is characterized by symptoms like chest pain, shortness of breath, fatigue, and palpitations. Heart disease can lead to life-threatening events like heart attacks and strokes. Early detection, risk factor management, and lifestyle modifications are essential in preventing and managing heart disease, along with medical interventions and treatments. Heart disease prediction using machine learning (ML) techniques aims to accurately assess the likelihood of heart disease by analyzing various risk factors and clinical parameters. By leveraging ML algorithms like decision trees, logistic regression, random forests, and neural networks, this system learns from patient data to identify patterns and correlations associated with heart disease. The system provides quick and reliable insights to healthcare professionals, aiding in proactive interventions and personalized treatment plans. However, medical expertise remains crucial in interpreting the system's predictions, as they are based on statistical patterns. The ML-based system serves as a valuable tool in enhancing heart disease prediction and supporting informed decision-making for improved patient outcomes."   )
    with st.container():
        st.header("Pneumonia Prediction")
        st.markdown(
            "Pneumonia is a prevalent and serious respiratory infection that requires accurate diagnosis for effective treatment. Manual examination of chest X-ray images by radiologists can be time-consuming and prone to errors. By leveraging deep learning algorithms such as CNN, ANN, and TL, a system can automate pneumonia prediction based on these images. The system learns to recognize patterns and extract relevant features from a large dataset of annotated chest X-ray images, offering speed, consistency, and potentially higher accuracy compared to manual examination. Healthcare professionals can benefit from the system's quick and reliable insights into the presence of pneumonia, enabling prompt interventions and appropriate treatment plans. However, it is crucial to note that the system should complement rather than replace medical expertise. Consulting healthcare professionals for accurate evaluation and diagnosis remains essential. The proposed system utilizing deep learning algorithms provides a promising solution for improving pneumonia diagnosis based on chest X-ray images, ultimately enhancing patient care and outcomes.    ")
    with st.container():
        st.header("Brain Tumor Prediction")
        st.markdown(
            "A Brain tumor is considered as one of the aggressive diseases, among children and adults. Brain tumors account for 85 to 90 percent of all primary Central Nervous System(CNS) tumors. Every year, around 11,700 people are diagnosed with a brain tumor. The 5-year survival rate for people with a cancerous brain or CNS tumor is approximately 34 percent for men and36 percent for women. Brain Tumors are classified as: Benign Tumor, Malignant Tumor, Pituitary Tumor, etc. Proper treatment, planning, and accurate diagnostics should be implemented to improve the life expectancy of the patients. The best technique to detect brain tumors is Magnetic Resonance Imaging (MRI). A huge amount of image data is generated through the scans. These images are examined by the radiologist. A manual examination can be error-prone due to the level of complexities involved in brain tumors and their properties.Application of automated classification techniques using Machine Learning(ML) and Artificial Intelligence(AI)has consistently shown higher accuracy than manual classification. Hence, proposing a system performing detection and classification by using Deep Learning Algorithms using Convolution-Neural Network (CNN), Artificial Neural Network (ANN), and Transfer-Learning (TL) would be helpful to doctors all around the world.   " )
        #st.plotly_chart(px.scatter(df, x="sepal_width", y="sepal_length", color="species"))

    
    
#diabetes preediction page
if(selected=='Diabetes prediction'):
    #page title
    st.title('Diabetes prediction using ML')
    
    #input fields
    col1,col2,col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input("Number of Pregnancies:")
    with col2:
        Glucose = st.text_input("Glucose Level:")
    with col3:
        BloodPressure = st.text_input("Blood Pressure Value:")
    with col1:
        SkinThickness = st.text_input("Skin Thickness value:")
    with col2:
        Insulin = st.text_input("Insulin Level:")
    with col3:
        BMI = st.text_input("BMI Value:")
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value:")
    with col2:
        Age = st.text_input("Age:")



    #code for prediction
    diabetes_diagnosis=""
    #creating button for getting result
    if st.button("Result"):
        diabetes_prediction = diabetes_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        
        if(diabetes_prediction[0]==1):
            diabetes_diagnosis = "The person is Diabetic"
            st.write(diabetes_diagnosis)
            
            st.write("If a person is suffering from diabetes, here are the key steps they should take:")
            st.write("1. Consult a healthcare professional for diagnosis and treatment.")
            st.write("2. Follow a healthy diet, focusing on whole foods and limiting sugary and processed foods.")
            st.write("3. Engage in regular physical activity, including aerobic and strength training exercises.")
            st.write("4. Monitor blood sugar levels as recommended by a healthcare professional.")
            st.write("5. Take prescribed medications or insulin as directed.")
            st.write("6. Educate oneself about diabetes and its management.")
            st.write("7. Maintain a healthy weight through diet and exercise.")
            st.write("8. Manage stress levels through relaxation techniques.")
            st.write("9. Regularly visit healthcare professionals for check-ups and adjustments to the treatment plan.")
            st.write("Remember, personalized guidance from a healthcare professional is crucial for effective diabetes management.") 
        else:
            diabetes_diagnosis = "The person is not Diabetic"
            st.write(diabetes_diagnosis)
            
            st.write("To prevent diabetes:")
            st.write("1. Maintain a healthy weight through a balanced diet and regular exercise.")
            st.write("2. Eat a nutritious diet rich in fruits, vegetables, whole grains, and lean proteins.")
            st.write("3. Limit the consumption of sugary and processed foods.")
            st.write("4. Engage in regular physical activity, aiming for at least 150 minutes of moderate-intensity exercise per week.")
            st.write("5. Avoid sedentary behavior and aim to be physically active throughout the day.")
            st.write("6. Stay hydrated and drink water as the primary beverage.")
            st.write("7. Avoid smoking and limit alcohol consumption.")
            st.write("8. Get regular check-ups and screenings for blood sugar levels and other health indicators.")
            st.write("9. Manage stress levels through healthy coping mechanisms such as exercise, meditation, or hobbies.")
            st.write("10. Maintain a healthy sleep routine, aiming for 7-8 hours of quality sleep per night.")
            
            st.write("Remember, these preventive measures can significantly reduce the risk of developing diabetes, but it's always recommended to consult with healthcare professionals for personalized advice and recommendations.")

        
    
#heart disease prediction
if(selected=='Heart Disease prediction'):
    st.title('Heart Disease prediction using ML')


    #input fields
    col1,col2,col3 = st.columns(3)
    
    with col1:
        age = st.text_input("Age:")
    with col2:
        sex = st.text_input("Sex:")
    with col3:
        cp = st.text_input("Chest Pain Types:")
    with col1:
        trestbps = st.text_input("Resting Blood Pressure:")
    with col2:
        chol = st.text_input("Serum Cholestral in mg/dl:")
    with col3:
        fbs = st.text_input("Fasting Blood Sugar>120 mg/dl:")
    with col1:
        restecg = st.text_input("Resting Electrocardiographic results:")
    with col2:
        thalach = st.text_input("Maximum Heart Rate achieved:")
    with col3:
        exang = st.text_input("Exercise Induced Angina:")
    with col1:
        oldpeak = st.text_input("ST depression induced by exercise:")
    with col2:
        slope = st.text_input("Slope of the peak exercise ST segment:")
    with col3:
        ca = st.text_input("Major vessels colored by Flourosopy:")
    with col1:
        thal = st.text_input("Thal:0=normal;1=fixed defect;2=reversable defect:")



    #code for prediction
    heart_diseases=""
    #creating button for getting result
    if st.button("Result"):
        heart_disease_prediction = heart_disease_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        
        if(heart_disease_prediction[0]==1):
            heart_diseases = "The person have heart disease"
            st.write(heart_diseases)
            st.write("If a person is suffering from heart disease, here are key steps to take:")
            st.write("1. Consult a healthcare professional for diagnosis and treatment.")
            st.write("2. Follow a heart-healthy diet and exercise regularly.")
            st.write("3. Take prescribed medications as directed.")
            st.write("4. Manage stress levels through relaxation techniques.")
            st.write("5. Quit smoking and limit alcohol consumption.")
            st.write("6. Maintain a healthy weight.")
            st.write("7. Monitor and manage other health conditions.")
            st.write("8. Follow up with healthcare professionals regularly.")
            st.write("Remember, personalized advice from healthcare professionals is important for specific recommendations.")
                        
        else:
            heart_diseases = "The person does not have heart disease"
            st.write(heart_diseases)
            st.write("To prevent heart disease:")
            st.write("1. Maintain a healthy diet.")
            st.write("2. Engage in regular physical activity.")
            st.write("3. Avoid smoking and limit alcohol consumption.")
            st.write("4. Maintain a healthy weight.")
            st.write("5. Manage stress effectively.")
            st.write("6. Monitor blood pressure and cholesterol levels.")
            st.write("7. Get regular check-ups.")
            st.write("8. Get sufficient sleep.")
            st.write("9. Stay informed about heart health.")
            st.write("Remember, personalized advice from healthcare professionals is important for specific recommendations.")
            

if(selected=='Pneumonia prediction'):
    st.title('Pneumonia Disease prediction')   

    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    import numpy as np
    
    # Load the trained model
    model = tf.keras.models.load_model("C:/Users/Astel Pauly/Desktop/multipple disease prediciton/mini/chest_xray/pneumonia_predictionn.h5")
    
    # Define the labels for prediction
    labels = ['Normal', 'Pneumonia']
    
    def preprocess_image(img):
        img = img.resize((120, 120))
        img = img.convert('RGB')
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    st.text("Upload an image and predict whether it is normal or pneumonia.")
    
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = image.load_img(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        processed_image = preprocess_image(img)
        prediction = model.predict(processed_image)
        predicted_class = labels[int(np.round(prediction))]
        st.write("Result:", predicted_class)
        if(predicted_class=="Pneumonia"):
                st.write("If a person develops pneumonia, here are the key steps they should take:")
                st.write("1. Seek medical attention for diagnosis and treatment.")
                st.write("2. Follow the prescribed treatment, including medications and rest.")
                st.write("3. Stay hydrated and get plenty of rest.")
                st.write("4. Manage symptoms with over-the-counter medications.")
                st.write("5. Practice good respiratory hygiene.")
                st.write("6. Follow a healthy lifestyle and avoid smoking.")
                st.write("7. Stay in touch with the healthcare professional for updates and follow-up.")
                st.write("Remember, personalized advice from a healthcare professional is important for proper care.")
        else:
                st.write("It's important to take certain steps to minimize the risk of infection. Here are key preventive measures:")
                st.write("1. Get vaccinated against pneumonia and influenza.")
                st.write("2. Practice good hygiene, including regular handwashing.")
                st.write("3. Maintain a strong immune system through a healthy lifestyle.")
                st.write("4. Promote respiratory health by covering coughs and avoiding smoking.")
                st.write("5. Stay up to date with routine healthcare.")
                st.write("6. Follow infection control measures in healthcare settings.")
                st.write("7. Consider at-risk populations and take extra precautions.")
                
                st.write("Remember, personalized advice from healthcare professionals is important for specific recommendations.")





