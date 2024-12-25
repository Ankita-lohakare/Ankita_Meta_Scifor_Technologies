import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import requests
import matplotlib.pyplot as plt
from streamlit_echarts import st_echarts


# Set Streamlit page configuration
st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Diet Recommendation System! 🥗")
st.markdown(
    """
    A diet recommendation web application using a content-based approach with Scikit-Learn, Python, and Streamlit.
    """
)

# Dataset creation
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

# Adjust preprocessing for new features
df = create_dataset()
df = pd.get_dummies(df, columns=["gender", "activity_level", "dietary_preference", "health_goal", "medical_condition"], drop_first=True)

# Split data
X = df.drop("calories", axis=1)
y = df["calories"]

# Train and save the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save the column names
with open("model_columns.pkl", "wb") as f:
    pickle.dump(X_train.columns.tolist(), f)

# Save the trained model
with open("diet_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load the model and column names
with open("diet_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# Spoonacular API Configuration
API_KEY = "142cf34ee433423fafd74acb9779858a"  # Replace with your actual API key
BASE_URL = "https://api.spoonacular.com/recipes/complexSearch"

# Food recommendation logic
def recommend_foods(calories, preference, condition):
    """
    Fetch food recommendations from the Spoonacular API and filter based on medical conditions.

    Args:
        calories (int): Calorie constraint for the meal.
        preference (str): Dietary preference (e.g., "vegan", "vegetarian", "non-vegetarian").
        condition (str): Health condition or additional keyword (e.g., "diabetes").

    Returns:
        list: List of recipe names with their URLs.
    """
    # Map preference to Spoonacular diet filters
    diet_map = {
        "vegan": "vegan",
        "vegetarian": "vegetarian",
        "non-vegetarian": None  # Non-veg is default in Spoonacular
    }

    # Condition filter for medical needs
    condition_keywords = {
        "diabetes": "low sugar",
        "hypertension": "low salt",
        "gluten intolerance": "gluten-free",
        "none": ""
    }
    condition_query = condition_keywords.get(condition, "")

    # Prepare query parameters
    params = {
        "apiKey": API_KEY,
        "diet": diet_map.get(preference),
        "maxCalories": calories,
        "number": 5,  # Number of recipes to fetch
        "query": condition_query  # Add keyword for the condition
    }

    # Make API request
    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        recipes = []
        for recipe in data.get("results", []):
            recipes.append({
                "title": recipe["title"],
                "url": f"https://spoonacular.com/recipes/{recipe['id']}"
            })
        return recipes
    else:
        st.error(f"Error fetching recipes: {response.status_code}")
        return []

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





# BMI Calculation
st.subheader("Calculate Your BMI")
bmi = weight / ((height / 100) ** 2)
st.write(f"Your BMI is: **{bmi:.2f}**")

# BMI Category
if bmi < 18.5:
    st.warning("You are underweight. Consider a diet to help gain weight healthily.")
elif 18.5 <= bmi < 24.9:
    st.success("You are in the normal weight range. Maintain your current lifestyle for good health.")
elif 25 <= bmi < 29.9:
    st.warning("You are overweight. Consider a diet and exercise plan to reduce weight.")
else:
    st.error("You are in the obese category. It's recommended to consult a healthcare provider for advice.")







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
if recommended_foods:
    for food in recommended_foods:
        st.markdown(f"- [{food['title']}]({food['url']})")
else:
    st.write("No recipes found for your criteria.")




# Nutrition Values Section
st.subheader("Nutritional Analysis of Your Chosen Recipes")

# List of nutrition attributes
nutritions_values = [
    "Calories", "FatContent", "SaturatedFatContent", "CholesterolContent",
    "SodiumContent", "CarbohydrateContent", "FiberContent", "SugarContent", "ProteinContent"
]

# Placeholder for demonstration: Replace `choices` and `recommendations` with actual user selections and API recommendations
choices = ["Recipe 1", "Recipe 2"]  # Example choices made by the user
recommendations = [
    [
        {"Name": "Recipe 1", "Calories": 250, "FatContent": 10, "SaturatedFatContent": 3, "CholesterolContent": 30, "SodiumContent": 200, "CarbohydrateContent": 30, "FiberContent": 5, "SugarContent": 8, "ProteinContent": 20},
        {"Name": "Recipe 2", "Calories": 300, "FatContent": 15, "SaturatedFatContent": 4, "CholesterolContent": 40, "SodiumContent": 250, "CarbohydrateContent": 40, "FiberContent": 6, "SugarContent": 10, "ProteinContent": 25}
    ]
]

# Calculate total nutritional values
total_nutrition_values = {nutrition_value: 0 for nutrition_value in nutritions_values}
for choice, meals_ in zip(choices, recommendations):
    for meal in meals_:
        if meal['Name'] == choice:
            for nutrition_value in nutritions_values:
                total_nutrition_values[nutrition_value] += meal[nutrition_value]

# Total and target calorie calculations
total_calories_chose = total_nutrition_values['Calories']
loss_calories_chose = round(predicted_calories * 0.8)  # Adjust based on weight loss or other goals






# Section to display categorized recipes
st.subheader("Recommended Meals")

# Dummy logic for dividing recipes (for demonstration purposes)
# You can enhance this by adding conditions or additional filtering logic
def categorize_meals(recommended_foods):
    breakfast = recommended_foods[:len(recommended_foods) // 3]
    lunch = recommended_foods[len(recommended_foods) // 3: 2 * len(recommended_foods) // 3]
    dinner = recommended_foods[2 * len(recommended_foods) // 3:]
    return breakfast, lunch, dinner

# Fetch food recommendations
recommended_foods = recommend_foods(predicted_calories, dietary_preference, medical_condition)

# Categorize the meals
breakfast, lunch, dinner = categorize_meals(recommended_foods)

# Display recipes categorized by meal type
def display_meal_category(category_name, recipes):
    st.markdown(f"### {category_name}")
    if recipes:
        for recipe in recipes:
            st.markdown(f"- [{recipe['title']}]({recipe['url']})")
    else:
        st.write("No recipes available for this category.")

# Display categorized meals
display_meal_category("Breakfast", breakfast)
display_meal_category("Lunch", lunch)
display_meal_category("Dinner", dinner)

























# Display Nutritional Values Pie Chart
st.markdown('<h5 style="text-align: center;font-family:sans-serif;">Nutritional Values Distribution:</h5>', unsafe_allow_html=True)
nutritions_graph_options = {
    "tooltip": {"trigger": "item"},
    "legend": {"top": "5%", "left": "center"},
    "series": [
        {
            "name": "Nutritional Values",
            "type": "pie",
            "radius": ["40%", "70%"],
            "avoidLabelOverlap": False,
            "itemStyle": {
                "borderRadius": 10,
                "borderColor": "#fff",
                "borderWidth": 2,
            },
            "label": {"show": False, "position": "center"},
            "emphasis": {
                "label": {"show": True, "fontSize": "20", "fontWeight": "bold"}
            },
            "labelLine": {"show": False},
            "data": [
                {"value": round(total_nutrition_values[nutrition_value]), "name": nutrition_value}
                for nutrition_value in nutritions_values
            ],
        }
    ],
}
st_echarts(options=nutritions_graph_options, height="500px")







# Visualize calorie distribution with enhancements
st.subheader("Calorie Distribution")

calorie_distribution = {
    "Breakfast": predicted_calories * 0.3,
    "Lunch": predicted_calories * 0.4,
    "Dinner": predicted_calories * 0.3,
}

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))


# Bar plot with custom colors
bars = ax.bar(
    calorie_distribution.keys(),
    calorie_distribution.values(),
    color=["#FF9999", "#66B3FF", "#99FF99"],
    edgecolor="black",
    linewidth=1.2,
)

# Add title and labels
ax.set_title("Calorie Distribution by Meals", fontsize=16, fontweight="bold")
ax.set_xlabel("Meal", fontsize=14)
ax.set_ylabel("Calories", fontsize=14)

# Add grid lines
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Add value annotations on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 10,  # Position slightly above the bar
        f"{int(height)} kcal",
        ha="center",
        va="bottom",
        fontsize=12,
        color="black",
    )

# Improve spacing and layout
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)
