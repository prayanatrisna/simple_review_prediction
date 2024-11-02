import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

model = joblib.load('model.pkl')

st.title('Sentiment Prediction')

col1, col2 = st.columns(2)

with col1:
	text = st.text_area(
		"Text to analyze:"
		)

prediction = model.predict([text])[0]
proba = model.predict_proba([text])[0]

if st.button("Predict"):
	with col1:
		st.markdown(
			"Predicted class: **{}**".format(prediction)
		)

	with col2:
		data_pie = pd.DataFrame()
		data_pie['class'] = ['Negative','Positive']
		data_pie['proba'] = proba
		pie = px.pie(data_pie, names='class', values='proba')
		st.plotly_chart(pie)