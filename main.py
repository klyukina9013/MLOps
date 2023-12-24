import joblib
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import json

st.header('Расчет стоимости квартиры')

PATH_DATA = 'data/all_v2.csv'
PATH_UNIQUE_VALUES = 'data/unique_values.json'
PATH_MODEL = 'models/lr_pipeline.sav'

@st.cache_data
def load_data(path):
    """Load data from path"""
    data = pd.read_csv(path)
    # для демонстрации
    data = data.sample(5000)
    return data

@st.cache_data
def load_model(path):
    """Load model from path"""
    model = joblib.load(PATH_MODEL)
    return model

@st.cache_data
def transform(data):
    """Transform data"""
    colors = sns.color_palette("coolwarm").as_hex()
    n_colors = len(colors)

    data = data.reset_index(drop=True)
    data['norm_price'] = data['price']/data['area'] #цена за м кв

    data['label_colors'] = pd.qcut(data['norm_price'], n_colors, labels=colors)
    data['label_colors'] = data['label_colors'].astype('str')
    return data

df = load_data(PATH_DATA)
df = transform(df)
st.write(df[:5])

st.map(data=df, latitude='geo_lat', longitude='geo_lon')

with open(PATH_UNIQUE_VALUES) as file:
    dict_unique = json.load(file)

building_type = st.sidebar.selectbox('Тип здания', (dict_unique['building_type']))
object_type = st.sidebar.selectbox('Тип объекта', (dict_unique['object_type']))
level = st.sidebar.slider(
    "Этаж", min_value=min(dict_unique['level']), max_value=max(dict_unique['level'])
)
levels = st.sidebar.slider(
    "Количество этажей", min_value=min(dict_unique['levels']), max_value=max(dict_unique['levels'])
)
rooms = st.sidebar.selectbox('Количество комнат', (dict_unique['rooms']))
area = st.sidebar.slider(
    "Локация", min_value=min(dict_unique['area']), max_value=max(dict_unique['area'])
)
kitchen_area = st.sidebar.slider(
    "Кухня", min_value=min(dict_unique['kitchen_area']), max_value=max(dict_unique['kitchen_area'])
)


dict_data = {
    'building_type': building_type,
    'object_type': object_type,
    'level': level,
    'levels': levels,
    'rooms': rooms,
    'area': area,
    'kitchen_area': kitchen_area,
}

coefficients = {
    'building_type': 0.01713991,
    'object_type': -0.24630437,
    'level': 0.01686385,
    'levels': 0.19029443,
    'rooms': 0.1732788,
    'area': 0.39313966,
    'kitchen_area': 0.13014464,
}

data_predict = pd.DataFrame([dict_data])
model = load_model(PATH_MODEL)

button = st.button('Предсказать цену')

if button:
    output = model.predict(data_predict)[0]
    st.success(f"{round(output)} рублей")
   #st.write(model.predict(data_predict)[0])


button_two = st.button('Предсказать цену с помощью модели линейной регрессии')

# coefficients = np.array([-0.04758898, -0.10148621,  0.01713991,  0.01686385,  0.19029443,  0.1732788, 0.39313966,  0.13014464, -0.24630437, -0.31429068])

intercept = -0.1822784074880021

if button_two:
    output_two = sum([coefficients[k] * dict_data[k] for k in coefficients.keys()]) + intercept
    st.success(f"{round(output_two)} рублей")

st.markdown(
    """
    ### Описание полей
        - Тип здания. 0 - Другой. 1 - Панельный. 2 - Монолитный. 3 - Кирпичный. 4 - Блочный. 5 - Деревянный
        - Тип объекта. 1 - Вторичка; 2 - Новостройка;
        - Этаж.
        - Количество этажей.
        - Количество комнат. * -1 - студия 
        - Локация. 
        - Кухня.
        - Цена. - в рублях
    """
)