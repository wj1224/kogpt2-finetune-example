import streamlit as st
import requests

url = "http://localhost:8000/translate"
headers = {'Content-type': 'application/json', 'accept': 'application/json'}

def translate(text):
    data = '{"text":%s}' % f'"{text}"'
    with st.spinner("Translating ..."):
        r = requests.post(url, headers=headers, data=data.encode('utf-8')).text[1:-1]
    st.success("Complete")
    st.write("Standard form")
    st.write(r)

    return r

st.title("Dialect Translator - KoGPT2")
    
text_input = st.text_input("Dialect form")
button = st.button("Translate to standard form")

if button:
    translated = translate(text_input)