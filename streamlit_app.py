
import re
import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="MessageGuard",
    page_icon="🛡️"
)


@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="telecomsxchange/OpenTextShield"
    )


def classify_message(text):
    results = load_model()(
        text,
        truncation=True,
        top_k=None
    )

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


CREDENTIAL_TERMS = [
    "password",
    "passcode",
    "one-time password",
    "otp",
    "verification code",
    "authentication code",
    "security code",
    "cvv",
    "card number",
    "expiry date",
    "expiration date",
]

DISCLOSURE_ACTIONS = [
    "reply with",
    "give us",
    "give me",
    "send",
    "share",
    "provide",
    "enter",
    "submit",
    "disclose",
    "type",
]

MACRO_TERMS = [
    "enable macro",
    "enable macros",
    "turn on macros",
    "allow macros",
]

SUSPICIOUS_OFFER_TERMS = [
    "guaranteed income",
    "guaranteed return",
    "receive $",
    "earn money instantly",
    "double your money",
]


def contains_term(text, terms):
    for term in terms:
        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def split_clauses(text):
    return [
        part.strip()
        for part in re.split(r"[.!?;\n]+", text.lower())
        if part.strip()
    ]


def is_negated_action(clause, action_start):
    """
    Check whether an action is preceded by wording such as
    'never send', 'do not share', or 'will never ask you to share'.
    """
    prefix = clause[:action_start]

    # Only examine the words immediately before the action.
    prefix_words = prefix.split()[-9:]

    if not prefix_words:
        return False

    recent_prefix = " ".join(prefix_words)

    negation_pattern = (
        r"\b(?:"
        r"never|"
        r"not|"
        r"don't|"
        r"do not|"
        r"doesn't|"
        r"does not|"
        r"didn't|"
        r"did not|"
        r"shouldn't|"
        r"should not|"
        r"mustn't|"
        r"must not|"
        r"cannot|"
        r"can't|"
        r"avoid"
        r")\b"
    )

    if not re.search(negation_pattern, recent_prefix):
        return False

    # Handle constructions such as:
    # "Your bank will never ask you to share your code."
    if re.search(
        r"\b(?:never|not|don't|do not|doesn't|does not|"
        r"didn't|did not|shouldn't|should not|mustn't|"
        r"must not|cannot|can't|avoid)\b",
        recent_prefix,
    ):
        return True

    return False


def find_positive_actions(clause):
    """Return disclosure actions that are not locally negated."""
    actions = []

    for action in DISCLOSURE_ACTIONS:
        pattern = r"(?<!\w)" + re.escape(action) + r"(?!\w)"

        for match in re.finditer(pattern, clause):
            if not is_negated_action(clause, match.start()):
                actions.append(match.group())

    return actions


def requests_sensitive_information(message):
    """
    Detect a likely request to disclose sensitive information.
    A sensitive term alone, or an unrelated action alone, is
    not enough to trigger this finding.
    """
    clauses = split_clauses(message)

    sensitive_clauses = [
        clause
        for clause in clauses
        if contains_term(clause, CREDENTIAL_TERMS)
    ]

    if not sensitive_clauses:
        return False

    # First check whether a sensitive term and a positive
    # disclosure action appear in the same clause.
    for clause in sensitive_clauses:
        if find_positive_actions(clause):
            return True

    # Then check for a follow-up instruction referring to
    # information mentioned in an earlier clause.
    reference_terms = [
        "the code",
        "your code",
        "this code",
        "the password",
        "your password",
        "the passcode",
        "your passcode",
        "the cvv",
        "your cvv",
        "the card number",
        "your card number",
        "the details",
        "these details",
        "this information",
        "it",
        "them",
    ]

    for index, clause in enumerate(clauses):
        if not contains_term(clause, CREDENTIAL_TERMS):
            continue

        # Only inspect the next two clauses for a related
        # instruction, rather than unrelated text anywhere.
        for follow_up in clauses[index + 1:index + 3]:
            if not find_positive_actions(follow_up):
                continue

            if contains_term(follow_up, reference_terms):
                return True

    return False


def has_positive_macro_instruction(message):
    for clause in split_clauses(message):
        if not contains_term(clause, MACRO_TERMS):
            continue

        # Treat direct warnings such as "Do not enable macros"
        # as warnings, not as instructions to enable them.
        macro_actions = [
            "enable macro",
            "enable macros",
            "turn on macros",
            "allow macros",
        ]

        for action in macro_actions:
            pattern = r"(?<!\w)" + re.escape(action) + r"(?!\w)"

            for match in re.finditer(pattern, clause):
                if not is_negated_action(clause, match.start()):
                    return True

    return False


def get_security_warnings(message):
    text = message.lower()
    warnings = []

    if has_positive_macro_instruction(text):
        warnings.append(
            "This message appears to instruct you to enable macros. "
            "Unexpected macro-enabled documents can execute dangerous "
            "code. Verify the document and its source independently."
        )

    if requests_sensitive_information(text):
        warnings.append(
            "This message may ask you to disclose sensitive information. "
            "Never share passwords, authentication codes, or payment-card "
            "security details in response to an unexpected request. "
            "Verify through the organization's official channel."
        )
    elif contains_term(text, CREDENTIAL_TERMS):
        warnings.append(
            "This message mentions passwords, verification codes, "
            "or payment-card details. A mention alone does not mean "
            "the sender is requesting them. Never disclose sensitive "
            "information in response to an unexpected request."
        )

    if contains_term(text, SUSPICIOUS_OFFER_TERMS):
        warnings.append(
            "This message contains wording associated with potentially "
            "suspicious financial offers. Guaranteed or unusually high "
            "returns are warning signs. Verify the offer independently "
            "before sending money."
        )

    return warnings


def get_risk_assessment(message):
    text = message.lower()
    findings = []

    if requests_sensitive_information(text):
        findings.append(
            {
                "level": "high",
                "title": "Sensitive-information request",
                "detail": (
                    "The message appears to ask for sensitive information. "
                    "Do not send passwords, authentication codes, or "
                    "payment-card security details through an unexpected "
                    "message. Verify using an official channel you access "
                    "independently."
                ),
            }
        )

    if has_positive_macro_instruction(text):
        findings.append(
            {
                "level": "high",
                "title": "Potentially dangerous document instruction",
                "detail": (
                    "Enabling macros in an unexpected document can "
                    "execute malicious code. Do not enable them unless "
                    "the document and its source have been verified."
                ),
            }
        )

    if contains_term(text, SUSPICIOUS_OFFER_TERMS):
        findings.append(
            {
                "level": "elevated",
                "title": "Suspicious financial offer",
                "detail": (
                    "The message contains language associated with "
                    "potentially deceptive financial offers. Check the "
                    "claims and organization independently before "
                    "sending money."
                ),
            }
        )

    if any(item["level"] == "high" for item in findings):
        overall_level = "high"
    elif findings:
        overall_level = "elevated"
    else:
        overall_level = "no_rule_triggered"

    return overall_level, findings


def needs_more_information(message, scores):
    top_score = scores[0]["score"]
    second_score = (
        scores[1]["score"] if len(scores) > 1 else 0.0
    )
    margin = top_score - second_score

    _, risk_findings = get_risk_assessment(message)

    return (
        bool(get_security_warnings(message))
        or bool(risk_findings)
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
            "The AI model identified patterns associated with phishing."
        )
        st.write(
            "**Advice:** Avoid suspicious links and never share "
            "passwords or verification codes."
        )

    elif label == "spam":
        st.warning("⚠️ Potential Spam Detected")
        st.write(
            "The AI model identified patterns associated with spam."
        )
        st.write(
            "**Advice:** Be cautious with unsolicited offers and links."
        )

    elif label == "ham":
        st.success("✅ Likely Legitimate — AI Prediction")
        st.write(
            "The AI model did not classify this message as phishing "
            "or spam. This does not establish that the message is safe."
        )
        st.write(
            "**Advice:** Verify unusual requests independently."
        )

    else:
        st.info(
            "The AI model returned an unrecognized classification."
        )


def show_scores(scores):
    with st.expander("AI model scores (testing only)"):
        st.write(
            "These are relative model output scores, not verified "
            "probabilities that the classification is correct."
        )

        for item in scores:
            st.write(
                f'{item["label"]}: {item["score"]:.4f}'
            )


def show_security_warnings(message):
    warnings = get_security_warnings(message)

    for warning in warnings:
        st.warning(f"🛡️ Security warning: {warning}")


def show_risk_assessment(message):
    level, findings = get_risk_assessment(message)

    st.subheader("Independent Rule-Based Risk Checks")

    st.caption(
        "These checks use message wording and predefined rules. "
        "They do not verify the sender or prove that a message "
        "is malicious."
    )

    if level == "high":
        st.error(
            "High-risk warning: the message contains a pattern "
            "that warrants particular caution."
        )
    elif level == "elevated":
        st.warning(
            "Caution: the message contains a potentially suspicious "
            "pattern that should be checked."
        )
    else:
        st.info(
            "No high-risk pattern was detected by the current rules. "
            "This does not mean the message is safe."
        )

    for finding in findings:
        st.markdown(f'**{finding["title"]}**')
        st.write(finding["detail"])


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
    # Clear previous results.
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
            st.session_state.request_context = (
                needs_more_information(cleaned_message, scores)
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

    st.subheader("Initial AI Assessment")
    show_assessment(initial_scores)
    show_scores(initial_scores)

    # Keep rule-based checks separate from the AI prediction.
    show_security_warnings(original_message)
    show_risk_assessment(original_message)

    if st.session_state.request_context:
        st.warning(
            "Additional context is recommended because the AI "
            "prediction may be uncertain or the message contains "
            "warning signs."
        )

        st.write(
            "Provide only non-sensitive context, such as whether "
            "you expected the message or verified the sender through "
            "an official channel. Never enter passwords, verification "
            "codes, or payment-card details."
        )

        additional_info = st.text_area(
            "Additional information (optional)",
            placeholder=(
                "Where did the message arrive? Were you expecting it? "
                "Did you verify the sender through an official channel?"
            ),
            max_chars=5000,
            key=(
                f"additional_info_"
                f"{st.session_state.analysis_version}"
            )
        )

        if st.button("Reassess with Additional Information"):
            if not additional_info.strip():
                st.warning(
                    "Please provide relevant, non-sensitive context first."
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
            final_scores = st.session_state.final_scores

            st.subheader("Updated AI Assessment")
            show_assessment(final_scores)
            show_scores(final_scores)

            initial_label = get_label(initial_scores)
            updated_label = get_label(final_scores)

            if initial_label == updated_label:
                st.info(
                    f"The AI model returned the same label both times: "
                    f"{initial_label}. This does not independently "
                    "confirm that the label is correct."
                )
            else:
                st.info(
                    f"The AI model's label changed from {initial_label} "
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
    "Rule-based checks can miss threats or flag harmless messages. "
    "Avoid entering private or sensitive information."
)
