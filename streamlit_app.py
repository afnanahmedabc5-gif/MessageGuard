
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="MessageGuard", page_icon="🛡️")

@st.cache_resource
def load_model():
    return pipeline("text-classification", model="telecomsxchange/OpenTextShield")

st.title("🛡️ MessageGuard")
st.subheader("AI-Powered Message Safety Checker")
st.write("Analyze SMS, emails, WhatsApp messages, and social media text for potential phishing and spam.")

message = st.text_area("Paste your message here:", max_chars=10000, height=180)

if st.button("Analyze Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        try:
            with st.spinner("Analyzing message..."):
                result = load_model()(message, truncation=True)[0]
            label = result["label"].lower()

            if label == "phishing":
                st.error("⚠️ Potential Phishing Detected")
                st.write("The model identified patterns associated with phishing.")
                st.write("**Advice:** Avoid suspicious links and never share passwords or verification codes.")
            elif label == "spam":
                st.warning("⚠️ Potential Spam Detected")
                st.write("The model identified patterns associated with spam.")
                st.write("**Advice:** Be cautious with unexpected offers and links.")
            elif label == "ham":
                st.success("✅ Likely Legitimate")
                st.write("The model did not classify this message as phishing or spam.")
                st.write("**Advice:** This is not a guarantee of safety. Verify unusual requests independently.")
            else:
                st.info("Uncertain result")
                st.write("Verify the message through a trusted channel.")
        except Exception:
            st.error("Analysis failed. Please try again.")

st.divider()
st.caption("MessageGuard — AI predictions are not guarantees of safety. Avoid entering private messages.")
