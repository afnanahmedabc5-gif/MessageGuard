
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="MessageGuard", page_icon="🛡️")


@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="telecomsxchange/OpenTextShield"
    )


st.title("🛡️ MessageGuard")
st.subheader("AI-Powered Message Safety Checker")
st.write(
    "Analyze SMS, emails, WhatsApp messages, and social media "
    "text for potential phishing and spam."
)

message = st.text_area(
    "Paste your message here:",
    max_chars=10000,
    height=180
)

if st.button("Analyze Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        try:
            with st.spinner("Analyzing message..."):
                classifier = load_model()
                raw_results = classifier(
                    message,
                    truncation=True,
                    top_k=None
                )

            # Handle the possible output formats from Transformers.
            if raw_results and isinstance(raw_results[0], list):
                scores = raw_results[0]
            else:
                scores = raw_results

            if not scores:
                st.error("The model did not return a result.")
            else:
                best_result = max(
                    scores,
                    key=lambda item: item["score"]
                )

                label = best_result["label"].lower()

                # Preserve the existing classification behavior.
                if label == "phishing":
                    st.error("⚠️ Potential Phishing Detected")
                    st.write(
                        "The model identified patterns associated "
                        "with phishing."
                    )
                    st.write(
                        "**Advice:** Avoid suspicious links and "
                        "never share passwords or verification codes."
                    )

                elif label == "spam":
                    st.warning("⚠️ Potential Spam Detected")
                    st.write(
                        "The model identified patterns associated "
                        "with spam."
                    )
                    st.write(
                        "**Advice:** Be cautious with unexpected "
                        "offers and links."
                    )

                elif label == "ham":
                    st.success("✅ Likely Legitimate")
                    st.write(
                        "The model did not classify this message "
                        "as phishing or spam."
                    )
                    st.write(
                        "**Advice:** This is not a guarantee of "
                        "safety. Verify unusual requests independently."
                    )

                else:
                    st.info("Uncertain result")
                    st.write(
                        "Verify the message through a trusted channel."
                    )

                # Temporary diagnostic information for development.
                with st.expander("Model scores (testing only)"):
                    st.write(
                        "These are model scores, not guaranteed "
                        "probabilities of correctness."
                    )

                    for item in sorted(
                        scores,
                        key=lambda result: result["score"],
                        reverse=True
                    ):
                        st.write(
                            f'{item["label"]}: '
                            f'{item["score"]:.4f}'
                        )

        except Exception:
            st.error(
                "Analysis failed. Please try again. "
                "If the problem continues, check the app logs."
            )

st.divider()
st.caption(
    "MessageGuard — AI predictions are not guarantees of safety. "
    "Avoid entering private messages."
)
