import streamlit as st
import pandas as pd
import uuid
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('Waiting_time.csv')
df.rename(columns={
    'People Count': 'people_count',
    'Waiting time(mins)': 'waiting_time',
    'Restaurant capacity': 'capacity'
}, inplace=True)

# Normalize
scaler = MinMaxScaler()
df[['people_count', 'capacity']] = scaler.fit_transform(df[['people_count', 'capacity']])

# Train model
X = df[['people_count', 'capacity']]
y = df['waiting_time']
model = LinearRegression()
model.fit(X, y)

# Predict function
def predict_wait_time(people_count, capacity=50):
    input_df = pd.DataFrame([[people_count, capacity]], columns=['people_count', 'capacity'])
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=['people_count', 'capacity'])
    prediction = model.predict(input_scaled_df)
    return round(prediction[0], 2)

# Streamlit App UI
st.title("🍽️ QueueLess - Restaurant Wait Time Estimator")

people_count = st.number_input("Enter your party size", min_value=1, max_value=20, value=2)

if st.button("Get Your Token"):
    token_id = str(uuid.uuid4())[:8]
    estimated_wait = predict_wait_time(people_count)
    expected_time = (datetime.now() + timedelta(minutes=estimated_wait)).strftime('%I:%M %p')

    st.success(f"🎟 Your Token: `{token_id}`")
    st.info(f"⏳ Estimated Wait Time: **{estimated_wait} minutes**")
    st.warning(f"📍 Be there by: **{expected_time}**")

    st.write("You'll be notified when your turn is near! 😊")
