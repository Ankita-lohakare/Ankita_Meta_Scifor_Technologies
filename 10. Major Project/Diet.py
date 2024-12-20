import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle



st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Diet Recommendation System! 🥗")

st.markdown(
    """
    A diet recommendation web application using content-based approach with Scikit-Learn, Python and Streamlit.
    """
)



def create_dataset():
    data = {
        "age": np.random.randint(18, 65, 1000),
        "gender": np.random.choice(["male", "female"], 1000),
        "weight": np.random.randint(50, 120, 1000),
        "height": np.random.randint(150, 200, 1000),
        "activity_level": np.random.choice(["low", "moderate", "high"], 1000),
        "dietary_preference": np.random.choice(["vegan", "vegetarian", "non-vegetarian"], 1000),
        "health_goal": np.random.choice(["weight loss", "weight maintenance", "weight gain"], 1000),
        "medical_condition": np.random.choice(["none", "diabetes", "hypertension", "gluten intolerance"], 1000),
        "calories": np.random.randint(1500, 3000, 1000),
    }
    return pd.DataFrame(data)

# Adjust the preprocessing for new features
df = create_dataset()

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=["gender", "activity_level", "dietary_preference", "health_goal", "medical_condition"], drop_first=True)


# Split data
X = df.drop("calories", axis=1)
y = df["calories"]

# Train the model as before
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Error fix
# Save the column names after one-hot encoding
with open("model_columns.pkl", "wb") as f:
    pickle.dump(X_train.columns.tolist(), f)

# till here

# Save the updated model
with open("diet_model.pkl", "wb") as f:
    pickle.dump(model, f)

#error fix

# Load model and column names
with open("diet_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# till here


import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load the trained model
with open("diet_model.pkl", "rb") as f:
    model = pickle.load(f)

# Food recommendation logic
def recommend_foods(calories, preference, condition):
    food_database = {
        "vegan": ["Quinoa Salad", "Avocado Toast", "Lentil Soup"],
        "vegetarian": ["Paneer Tikka", "Vegetable Stir Fry", "Cheese Sandwich"],
        "non-vegetarian": ["Grilled Chicken", "Salmon Salad", "Beef Steak"],
    }
    # Filter based on medical conditions
    condition_filter = {
        "diabetes": lambda x: "low sugar" in x.lower(),
        "hypertension": lambda x: "low salt" in x.lower(),
        "gluten intolerance": lambda x: "gluten-free" in x.lower(),
        "none": lambda x: True,
    }
    foods = food_database[preference]
    return [food for food in foods if condition_filter[condition](food)]

# Streamlit UI
st.title("Personalized Diet Recommendation")

# User Inputs
age = st.slider("Age", 18, 65, 25)
gender = st.selectbox("Gender", ["male", "female"])
weight = st.slider("Weight (kg)", 40, 150, 70)
height = st.slider("Height (cm)", 140, 210, 170)
activity_level = st.selectbox("Activity Level", ["low", "moderate", "high"])
dietary_preference = st.selectbox("Dietary Preference", ["vegan", "vegetarian", "non-vegetarian"])
health_goal = st.selectbox("Health Goal", ["weight loss", "weight maintenance", "weight gain"])
medical_condition = st.selectbox("Medical Condition", ["none", "diabetes", "hypertension", "gluten intolerance"])

# Preprocess input
input_data = pd.DataFrame({
    "age": [age],
    "weight": [weight],
    "height": [height],
    "gender_male": [1 if gender == "male" else 0],
    "activity_level_moderate": [1 if activity_level == "moderate" else 0],
    "activity_level_high": [1 if activity_level == "high" else 0],
    "dietary_preference_vegetarian": [1 if dietary_preference == "vegetarian" else 0],
    "dietary_preference_non-vegetarian": [1 if dietary_preference == "non-vegetarian" else 0],
    "health_goal_weight maintenance": [1 if health_goal == "weight maintenance" else 0],
    "health_goal_weight gain": [1 if health_goal == "weight gain" else 0],
    "medical_condition_diabetes": [1 if medical_condition == "diabetes" else 0],
    "medical_condition_hypertension": [1 if medical_condition == "hypertension" else 0],
    "medical_condition_gluten intolerance": [1 if medical_condition == "gluten intolerance" else 0],
})

# Align input data with model columns
for col in model_columns:
    if col not in input_data.columns:
        input_data[col] = 0  # Add missing columns with default value

# Ensure column order matches the training data
input_data = input_data[model_columns]

# Predict calorie needs
predicted_calories = model.predict(input_data)[0]
st.subheader(f"Recommended Calorie Intake: {int(predicted_calories)} kcal")

# Recommend foods
recommended_foods = recommend_foods(predicted_calories, dietary_preference, medical_condition)
st.subheader("Recommended Foods:")
for food in recommended_foods:
    st.write(f"- {food}")

# Visualize calorie distribution
st.subheader("Calorie Distribution")
calorie_distribution = {
    "Breakfast": predicted_calories * 0.3,
    "Lunch": predicted_calories * 0.4,
    "Dinner": predicted_calories * 0.3,
}
fig, ax = plt.subplots()
ax.bar(calorie_distribution.keys(), calorie_distribution.values(), color=["blue", "green", "orange"])
st.pyplot(fig)

