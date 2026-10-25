
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

    if "enable macro" in text:
        warnings.append(
            "This message asks you to enable macros. Unexpected "
            "macro-enabled documents can execute dangerous code. "
            "Verify the document and its source independently."
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
            "or payment-card details. Never share sensitive details "
            "in response to an unexpected request. Verify through "
            "the organization's official channel."
        )

    suspicious_offer_terms = [
        "guaranteed income",
        "guaranteed return",
        "receive $",
        "earn money instantly",
        "double your money",
    ]

    if any(term in text for term in suspicious_offer_terms):
        warnings.append(
            "This message may be promoting a suspicious financial "
            "offer. Guaranteed or unusually high returns are a warning "
            "sign. Verify the offer independently before sending money."
        )

    return warnings


def needs_more_information(message, scores):
    top_score = scores[0]["score"]
    second_score = scores[1]["score"] if len(scores) > 1 else 0.0
    margin = top_score - second_score

    return (
        bool(get_security_warnings(message))
        or top_score < 0.75
        or margin < 0.20
    )


def get_label(scores):
    return scores[0]["label"].lower()


def show_assessment(scores):
    label = get_label(scores)

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
            "**Advice:** Be cautious with unsolicited offers and links."
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
        st.info("The model returned an unrecognized classification.")


def show_scores(scores):
    with st.expander("Model scores (testing only)"):
        st.write(
            "These are relative model output scores, not verified "
            "probabilities that the classification is correct."
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


# Initialize session state.
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

if "analysis_version" not in st.session_state:
    st.session_state.analysis_version = 0


message = st.text_area(
    "Paste your message here:",
    max_chars=10000,
    height=180,
    key="message_input"
)


if st.button("Analyze Message", type="primary"):
    # Clear all results from the previous analysis.
    st.session_state.original_message = ""
    st.session_state.initial_scores = None
    st.session_state.request_context = False
    st.session_state.final_scores = None
    st.session_state.reassessed = False
    st.session_state.analysis_version += 1

    cleaned_message = message.strip()

    if not cleaned_message:
        st.warning("Please enter a message.")
    else:
        try:
            with st.spinner("Analyzing message..."):
                scores = classify_message(cleaned_message)

            st.session_state.original_message = cleaned_message
            st.session_state.initial_scores = scores
            st.session_state.request_context = needs_more_information(
                cleaned_message, scores
            )

        except Exception:
            st.error(
                "Analysis failed. Please try again. "
                "If the problem continues, check the app logs."
            )


if st.session_state.initial_scores is not None:
    original_message = st.session_state.original_message
    initial_scores = st.session_state.initial_scores

    st.caption("Message currently being assessed:")
    st.code(original_message)

    # Security warnings are independent of the AI classification.
    show_security_warnings(original_message)

    # Always display the initial prediction and scores.
    st.subheader("Initial Assessment")
    show_assessment(initial_scores)
    show_scores(initial_scores)

    if st.session_state.request_context:
        st.warning(
            "Additional context is recommended because the prediction "
            "may be uncertain or the message contains warning signs."
        )

        st.write(
            "Only provide non-sensitive context, such as whether you "
            "expected the message or independently verified the sender. "
            "Never enter passwords, verification codes, or card details."
        )

        additional_info = st.text_area(
            "Additional information (optional)",
            placeholder=(
                "Where did the message arrive? Were you expecting it? "
                "Did you verify the sender through an official channel?"
            ),
            max_chars=5000,
            key=f"additional_info_{st.session_state.analysis_version}"
        )

        if st.button("Reassess with Additional Information"):
            if not additional_info.strip():
                st.warning(
                    "Please provide relevant non-sensitive context first."
                )
            else:
                combined_text = (
                    "Original message:\n"
                    + original_message
                    + "\n\nAdditional user-provided context:\n"
                    + additional_info.strip()
                )

                try:
                    with st.spinner("Reassessing the message..."):
                        new_scores = classify_message(combined_text)

                    st.session_state.final_scores = new_scores
                    st.session_state.reassessed = True

                except Exception:
                    st.error(
                        "Reassessment failed. Please try again."
                    )

        if st.session_state.reassessed:
            st.subheader("Updated Assessment")
            show_assessment(st.session_state.final_scores)
            show_scores(st.session_state.final_scores)

            initial_label = get_label(initial_scores)
            updated_label = get_label(st.session_state.final_scores)

            if initial_label == updated_label:
                st.info(
                    f"The model returned the same label both times: "
                    f"{initial_label}. This does not independently "
                    "confirm that the label is correct."
                )
            else:
                st.info(
                    f"The model's label changed from {initial_label} "
                    f"to {updated_label} after context was added. "
                    "A changed prediction is not proof that the updated "
                    "prediction is correct."
                )

            st.caption(
                "Both assessments use the same AI model. The added "
                "context and the sender have not been independently "
                "verified by MessageGuard."
            )


st.divider()
st.caption(
    "MessageGuard — AI predictions are not guarantees of safety. "
    "Avoid entering private or sensitive information."
)
