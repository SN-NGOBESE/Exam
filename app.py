import streamlit as st, sys

st.title("Streamlit sanity check ✅")
st.write("Python:", sys.version)
st.write("If you see this, your app is running.")

st.subheader("Built-in chart (no extra packages)")
st.line_chart({"y":[1,3,2,5,4,6]})
