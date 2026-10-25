
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
    "passwords",
    "passcode",
    "passcodes",
    "one-time password",
    "one time password",
    "otp",
    "verification code",
    "verification codes",
    "authentication code",
    "authentication codes",
    "security code",
    "security codes",
    "verify code",
    "verification number",
    "recovery code",
    "recovery codes",
    "login code",
    "login codes",
    "pin",
    "pin code",
    "cvv",
    "cvc",
    "card number",
    "credit card number",
    "debit card number",
    "expiry date",
    "expiration date",
    "card security details",
]

DISCLOSURE_ACTIONS = [
    r"send(?:ing|s)?",
    r"share(?:d|s|ing)?",
    r"provid(?:e|es|ed|ing)",
    r"enter(?:s|ed|ing)?",
    r"submit(?:s|ted|ting)?",
    r"type(?:s|d|ing)?",
    r"disclos(?:e|es|ed|ing)",
    r"giv(?:e|es|en|ing)",
    r"tell(?:s|ing|t)?",
    r"upload(?:s|ed|ing)?",
    r"reveal(?:s|ed|ing)?",
    r"supply(?:s|ied|ying)?",
    r"divulg(?:e|es|ed|ing)",
    r"forward(?:s|ed|ing)?",
    r"reply(?:ing|ies|ied)?",
    r"confirm(?:s|ed|ing)?",
]

MACRO_TERMS = [
    "enable macro",
    "enable macros",
    "enabling macro",
    "enabling macros",
    "turn on macro",
    "turn on macros",
    "allow macro",
    "allow macros",
]

SUSPICIOUS_OFFER_TERMS = [
    "guaranteed income",
    "guaranteed return",
    "receive $",
    "earn money instantly",
    "double your money",
]

NEGATION_WORDING = (
    r"(?:never|not|don't|do not|doesn't|does not|"
    r"didn't|did not|shouldn't|should not|mustn't|"
    r"must not|cannot|can't|avoid|refrain from)"
)

DISCLOSURE_VERB_PATTERN = (
    r"(?:send(?:ing|s)?|share(?:d|s|ing)?|"
    r"provid(?:e|es|ed|ing)|enter(?:s|ed|ing)?|"
    r"submit(?:s|ted|ting)?|type(?:s|d|ing)?|"
    r"disclos(?:e|es|ed|ing)|giv(?:e|es|en|ing)|"
    r"tell(?:s|ing|t)?|upload(?:s|ed|ing)?|"
    r"reveal(?:s|ed|ing)?|supply(?:s|ied|ying)?|"
    r"divulg(?:e|es|ed|ing)|forward(?:s|ed|ing)?|"
    r"reply(?:ing|ies|ied)?|confirm(?:s|ed|ing)?)"
)


def contains_term(text, terms):
    for term in terms:
        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"

        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def split_clauses(text):
    return [
        part.strip()
        for part in re.split(r"[.!?;\n,]+", text.lower())
        if part.strip()
    ]


def is_negated_action(clause, action_start):
    prefix = clause[:action_start]

    if re.search(
        NEGATION_WORDING + r"\s+(?:\w+\s+){0,4}$",
        prefix
    ):
        return True

    if re.search(
        r"\b(?:dangerous|unsafe|risky|unwise)\s+to\s+"
        r"(?:\w+\s+){0,2}$",
        prefix
    ):
        return True

    return False


def find_positive_actions(clause):
    actions = []

    for action in DISCLOSURE_ACTIONS:
        pattern = r"(?<!\w)" + action + r"(?!\w)"

        for match in re.finditer(pattern, clause):
            if not is_negated_action(clause, match.start()):
                actions.append(match.group())

    return actions


def has_sensitive_term(clause):
    return contains_term(clause, CREDENTIAL_TERMS)


def requests_sensitive_information(message):
    clauses = split_clauses(message)

    sensitive_indexes = [
        index
        for index, clause in enumerate(clauses)
        if has_sensitive_term(clause)
    ]

    if not sensitive_indexes:
        return False

    for index in sensitive_indexes:
        clause = clauses[index]

        if find_positive_actions(clause):
            return True

        # Catch common direct requests, including:
        # "enter your password"
        # "entering your password"
        # "reply with your verification code"
        # "provide your card number"
        direct_request_patterns = [
            r"\b(?:ask|asks|asked|asking)\b.{0,45}\b"
            + DISCLOSURE_VERB_PATTERN + r"\b",

            r"\b(?:need|needs|needed|needing|want|wants|"
            r"wanted|require|requires|required|request|"
            r"requests|requested)\b.{0,45}\b"
            + DISCLOSURE_VERB_PATTERN + r"\b",

            r"\b(?:reply|respond)\b.{0,20}\bwith\b",
        ]

        for pattern in direct_request_patterns:
            for match in re.finditer(pattern, clause):
                action_match = re.search(
                    DISCLOSURE_VERB_PATTERN,
                    match.group(),
                )

                if action_match:
                    absolute_start = match.start() + action_match.start()

                    if not is_negated_action(clause, absolute_start):
                        return True

    # Handle a sensitive item mentioned in one clause and a
    # related instruction in the following clause.
    reference_terms = [
        "the code",
        "your code",
        "this code",
        "the password",
        "your password",
        "the passcode",
        "your passcode",
        "the pin",
        "your pin",
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

    for index in sensitive_indexes:
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

        # A warning about enabling macros is not an instruction
        # to enable them.
        warning_patterns = [
            r"\bdo not\b.{0,30}\benabl(?:e|ing)\s+macros?\b",
            r"\bdon't\b.{0,30}\benabl(?:e|ing)\s+macros?\b",
            r"\bnever\b.{0,30}\benabl(?:e|ing)\s+macros?\b",
            r"\bshould not\b.{0,30}\benabl(?:e|ing)\s+macros?\b",
            r"\bshouldn't\b.{0,30}\benabl(?:e|ing)\s+macros?\b",
            r"\bmust not\b.{0,30}\benabl(?:e|ing)\s+macros?\b",
            r"\bmustn't\b.{0,30}\benabl(?:e|ing)\s+macros?\b",
            r"\bavoid\b.{0,30}\benabl(?:e|ing)\s+macros?\b",
            r"\b(?:dangerous|unsafe|risky)\s+to\s+"
            r"enabl(?:e|ing)\s+macros?\b",
        ]

        if any(
            re.search(pattern, clause)
            for pattern in warning_patterns
        ):
            continue

        for action in [
            "enable macro",
            "enable macros",
            "enabling macro",
            "enabling macros",
            "turn on macro",
            "turn on macros",
            "allow macro",
            "allow macros",
        ]:
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
    label = get_label(scores)

    _, risk_findings = get_risk_assessment(message)

    # Recommend context for rule-based findings, suspicious AI
    # classifications, or a comparatively uncertain prediction.
    # A credential mention alone does not trigger this prompt.
    return (
        bool(risk_findings)
        or label in ("phishing", "spam")
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

    show_security_warnings(original_message)
    show_risk_assessment(original_message)

    if st.session_state.request_context:
        st.warning(
            "Additional context may help interpret this result. "
            "The prediction and rule-based findings should be "
            "considered alongside independently verified context."
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
