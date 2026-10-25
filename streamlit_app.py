
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="MessageGuard", page_icon="🛡️")


@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="telecomsxchange/OpenTextShield"
    )


def classify_message(text):
    results = load_model()(text, truncation=True, top_k=None)

    if results and isinstance(results[0], list):
        scores = results[0]
    else:
        scores = results

    scores = sorted(
        scores,
        key=lambda item: item["score"],
        reverse=True
    )

    if not scores:
        raise ValueError("The model returned no scores.")

    return scores



   

        

   
def get_security_warnings(message):
    text = message.lower()
    warnings = []

    if "enable macros" in text or "enable macro" in text:
        warnings.append(
            "This message asks you to enable macros. "
            "Unexpected macro-enabled documents can execute "
            "dangerous code. Verify the document and its source "
            "independently before opening or enabling macros."
        )

    credential_terms = [
        "password",
        "passcode",
        "one-time password",
        "otp",
        "verification code",
        "authentication code",
        "cvv",
        "security code",
        "card number",
        "expiry date",
    ]

    if any(term in text for term in credential_terms):
        warnings.append(
            "This message mentions passwords, verification codes, "
            "or payment-card details. Do not share sensitive details "
            "in response to an unexpected message. Verify requests "
            "through the organization's official channel."
        )

    return warnings





def needs_more_information(message, scores):
    top_score = scores[0]["score"]
    second_score = scores[1]["score"] if len(scores) > 1 else 0.0
    margin = top_score - second_score

    return top_score < 0.75 or margin < 0.20



def show_assessment(scores):
    label = scores[0]["label"].lower()

    if label == "phishing":
        st.error("⚠️ Potential Phishing Detected")
        st.write(
            "The model identified patterns associated with phishing."
        )
        st.write(
            "**Advice:** Avoid suspicious links and never share "
            "passwords or verification codes."
        )

    elif label == "spam":
        st.warning("⚠️ Potential Spam Detected")
        st.write(
            "The model identified patterns associated with spam."
        )
        st.write(
            "**Advice:** Be cautious with unexpected offers and links."
        )

    elif label == "ham":
        st.success("✅ Likely Legitimate")
        st.write(
            "The model did not classify this message as phishing or spam."
        )
        st.write(
            "**Advice:** This prediction is not a guarantee of safety. "
            "Verify unusual requests independently."
        )

    else:
        st.info("Unable to classify this message reliably.")


def show_scores(scores):
    with st.expander("Model scores (testing only)"):
        st.write(
            "These scores show the model's relative output, "
            "not guaranteed probabilities of correctness."
        )
        for item in scores:
            st.write(f'{item["label"]}: {item["score"]:.4f}')

def show_security_warnings(message):
    warnings = get_security_warnings(message)

    for warning in warnings:
        st.warning(f"🛡️ Security warning: {warning}")


st.title("🛡️ MessageGuard")
st.subheader("AI-Powered Message Safety Checker")
st.write(
    "Analyze SMS, emails, WhatsApp messages, and social media "
    "text for potential phishing and spam."
)

if "original_message" not in st.session_state:
    st.session_state.original_message = ""

if "initial_scores" not in st.session_state:
    st.session_state.initial_scores = None

if "request_context" not in st.session_state:
    st.session_state.request_context = False

if "final_scores" not in st.session_state:
    st.session_state.final_scores = None

if "reassessed" not in st.session_state:
    st.session_state.reassessed = False

message = st.text_area(
    "Paste your message here:",
    max_chars=10000,
    height=180,
    key="message_input"
)

if st.button("Analyze Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        try:
            with st.spinner("Analyzing message..."):
                scores = classify_message(message.strip())

            st.session_state.original_message = message.strip()
            st.session_state.initial_scores = scores
            st.session_state.request_context = needs_more_information(
                message.strip(), scores
            )
            st.session_state.final_scores = None
            st.session_state.reassessed = False

        except Exception:
            st.error(
                "Analysis failed. Please try again. "
                "If the problem continues, check the app logs."
            )


if st.session_state.initial_scores is not None:
    initial_scores = st.session_state.initial_scores

    st.caption("Message currently being assessed:")
    st.code(st.session_state.original_message)

    show_security_warnings(st.session_state.original_message)

    if st.session_state.request_context:

        st.warning(
            "We need more information to assess this message reliably."
        )
        st.write(
            "The prediction may be uncertain, or the message may contain "
            "warning signs that deserve closer review."
        )

        additional_info = st.text_area(
            "Additional information (optional)",
            placeholder=(
                "For example: where you received it, when it arrived, "
                "who it claims to be from, or other relevant details."
            ),
            max_chars=5000,
            key="additional_info"
        )

        if st.button("Reassess with Additional Information"):
            if not additional_info.strip():
                st.warning(
                    "Please provide some additional information first."
                )
            else:
                combined_text = (
                    "Original message:\n"
                    + st.session_state.original_message
                    + "\n\nAdditional user-provided context:\n"
                    + additional_info.strip()
                )

                try:
                    with st.spinner("Reassessing the message..."):
                        st.session_state.final_scores = (
                            classify_message(combined_text)
                        )
                    st.session_state.reassessed = True

                except Exception:
                    st.error(
                        "Reassessment failed. Please try again."
                    )

        if st.session_state.reassessed:
            st.subheader("Updated Assessment")
            show_assessment(st.session_state.final_scores)
            show_scores(st.session_state.final_scores)

            st.info(
                "This is a best-effort AI reassessment using your "
                "additional information. The information has not "
                "necessarily been independently verified."
            )

    else:
        show_assessment(initial_scores)
        show_scores(initial_scores)

st.divider()
st.caption(
    "MessageGuard — AI predictions are not guarantees of safety. "
    "Avoid entering private or sensitive information."
)
