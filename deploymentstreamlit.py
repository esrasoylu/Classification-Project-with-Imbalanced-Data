import numpy as np
import streamlit as st
import pandas as pd


# Page Settings
st.set_page_config(
    page_title="Lung Cancer Probability Measrement",
    page_icon="🧪",
    menu_items={
        "Get help": "esrasoylu.bt@gmail.com",
        "About": "For More Information\n" + "https://github.com/esrasoylu"
   
  
    }
)

# Add a Title
st.title("Lung Cancer Probability Measurement")

# Creating Markdown
st.markdown("The effectiveness of :red[cancer prediction system] helps the people to know their cancer risk with low cost and it also helps the people to take the appropriate decision based on their cancer risk status. The data is collected from the website online lung cancer prediction system.")

# Adding Images
st.image("https://docsd.anadolusaglik.org/blog/crops/600x340/akciger_kanseri_nedir_belirtileri_ve_tedavisi_85867.jpg")

st.markdown("Lung cancer is the most common cause of cancer-related deaths in the world and in Turkey. Lung cancer is the most common cancer in men in the world and is the 3rd most common cancer type in women.")
st.markdown("It is estimated that 2.2 million new cases occurred worldwide in 2020 and 1.8 million deaths occurred due to lung cancer.")
st.markdown("*Measure your chances of getting lung cancer*")

#sidebar
gender = 1 if st.sidebar.selectbox("1. Cinsiyet (M(male) or F(female))", ("M", "F")) == "M" else 0
age = st.sidebar.number_input("2. Yaş", min_value=0, step=1)
smoking = 2 if st.sidebar.selectbox("3. Sigara Kullanımı (YES or NO)", ("YES", "NO")) == "YES" else 1
yellow_fingers = 2 if st.sidebar.selectbox("4. Sarı Parmaklar (YES or NO)", ("YES", "NO")) == "YES" else 1
anxiety = 2 if st.sidebar.selectbox("5. Anksiyete (YES or NO)", ("YES", "NO")) == "YES" else 1
peer_pressure = 2 if st.sidebar.selectbox("6. Arkadaş Baskısı (YES or NO)", ("YES", "NO")) == "YES" else 1
chronic_disease = 2 if st.sidebar.selectbox("7. Kronik Hastalık (YES or NO)", ("YES", "NO")) == "YES" else 1
fatigue = 2 if st.sidebar.selectbox("8. Yorgunluk (YES or NO)", ("YES", "NO")) == "YES" else 1
allergy = 2 if st.sidebar.selectbox("9. Alerji (YES or NO)", ("YES", "NO")) == "YES" else 1
wheezing = 2 if st.sidebar.selectbox("10. Hırıltı (YES or NO)", ("YES", "NO")) == "YES" else 1
alcohol = 2 if st.sidebar.selectbox("11. Alkol Kullanımı (YES or NO)", ("YES", "NO")) == "YES" else 1
coughing = 2 if st.sidebar.selectbox("12. Öksürük (YES or NO)", ("YES", "NO")) == "YES" else 1
shortness_of_breath = 2 if st.sidebar.selectbox("13. Nefes Darlığı (YES or NO)", ("YES", "NO")) == "YES" else 1
swallowing_difficulty = 2 if st.sidebar.selectbox("14. Yutma Zorluğu (YES or NO)", ("YES", "NO")) == "YES" else 1
chest_pain = 2 if st.sidebar.selectbox("15. Göğüs Ağrısı (YES or NO)", ("YES", "NO")) == "YES" else 1


#adding data

df = pd.read_csv("C:/Users/Esra SOYLU/Desktop/lungcancer.csv")

#---------------------------------------------------------------------------------------------------------------------

# Reusing the trained model using the Pickle library
from joblib import load

logreg_model = load('logreg_model.pkl')


input_df = pd.DataFrame({
    'Cinsiyet': [gender],
    'Yaş': [age],
    'Sigara Kullanımı': [smoking],
    'Sarı Parmak': [yellow_fingers],
    'Anksiyete': [anxiety],
    'Akran Baskısı': [peer_pressure],
    'Kronik Hastalık': [chronic_disease],
    'Yorgunluk': [fatigue],
    'Alerji': [allergy],
    'Hırıltı': [wheezing],
    'Alkol Kullanımı': [alcohol],
    'Öksürük': [coughing],
    'Nefes Darlığı': [shortness_of_breath],
    'Yutma Zorluğu': [swallowing_difficulty],
    'Gögüs Ağrısı': [chest_pain]

})

pred = logreg_model.predict(input_df.values)
pred_probability = np.round(logreg_model.predict_proba(input_df.values), 2)

#---------------------------------------------------------------------------------------------------------------------

st.header("Results")

# Result Screen
if st.sidebar.button("Submit"):

    # Creating info message
    st.info("You can find the result below.")

    # Obtaining information regarding query time
    from datetime import date, datetime

    today = date.today()
    time = datetime.now().strftime("%H:%M:%S")

    # DataFrame to Display Results
    results_df = pd.DataFrame({

    'Cinsiyet': [gender],
    'Yaş': [age],
    'Sigara Kullanımı': [smoking],
    'Sarı Parmak': [yellow_fingers],
    'Anksiyete': [anxiety],
    'Akran Baskısı': [peer_pressure],
    'Kronik Hastalık': [chronic_disease],
    'Yorgunluk': [fatigue],
    'Alerji': [allergy],
    'Hırıltı': [wheezing],
    'Alkol Kullanımı': [alcohol],
    'Öksürük': [coughing],
    'Nefes Darlığı': [shortness_of_breath],
    'Yutma Zorluğu': [swallowing_difficulty],
    'Gögüs Ağrısı': [chest_pain],
    'Prediction': [pred]

   
    })
 
    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("0"," low"))
    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("1","high"))

    st.table(results_df)
Finally, we can view the web interface we created by typing localhost:8501 in the browser. For detailed information on using Streamlit, you can take a look at https://docs.streamlit.io/

