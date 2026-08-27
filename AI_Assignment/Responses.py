import random

responses_data = {
    "greeting": [
        "Hello! How can I assist you with your TARUMT enquiry today?",
        "Hi! What TARUMT information are you looking for?",
        "Hello! You may ask me about admissions, timetables, examinations, fees, scholarships, programmes, or campus facilities.",
    ],

    "admission": [
        "You may submit your application through the TARUMT Online Application system and upload the required supporting documents.",
        "You may check your application status by logging in to your TARUMT Online Application account. The application result will also be sent by email.",
        "Please refer to the TARUMT Online Application system for application procedures, required documents, processing fees, and application status.",
    ],

    "timetable": [
        "You may check your class timetable through the Class Timetable section in the TARC App.",
        "Please open the TARC App and select Class Timetable to view your lecture and tutorial schedule.",
        "Your latest class schedule is available under Class Timetable in the TARC App.",
    ],

    "examination": [
        "You may use the TARC App to check your examination timetable, including the date, time, and venue, as well as your examination results.",
        "Please refer to the examination section in the TARC App for your exam schedule, venue, and results.",
        "Your examination timetable and examination results are available through the relevant sections in the TARC App.",
    ],

    "fees": [
        "You may check your current billing and outstanding fees through Billing → Current Billing in the TARC App.",
        "Please open the TARC App and go to Billing → Current Billing to view your fee balance and payment information.",
        "Your latest billing details are available under Billing → Current Billing in the TARC App.",
    ],

    "scholarship": [
        "Please refer to the TARUMT official website for the latest scholarship and financial aid information. Merit Scholarship details are available under the Merit Scholarship page.",
        "You may check the TARUMT official website for scholarship eligibility, financial assistance, application details, and deadlines.",
        "For the latest scholarship and financial aid information, please refer to the relevant scholarship pages on the TARUMT official website.",
    ],

    "programme": [
        "You may view your programme structure through Student Intranet by selecting Programme and then Programme Structure.",
        "Please log in to Student Intranet and go to Programme → Programme Structure to view your modules and course structure.",
        "Your programme modules, semesters, and course structure are available under Programme Structure in Student Intranet.",
    ],

    "campus_facility": [
        "You may use the Campus Map in the TARC App to locate campus facilities. For the latest operating hours, please check the relevant facility information or contact the campus directly.",
        "Please refer to the Campus Map in the TARC App for facility locations. Operating hours may vary, so please check the relevant facility information or contact the campus directly.",
        "Campus facilities such as computer laboratories, libraries, cafeterias, and study areas can be located through the Campus Map in the TARC App. For operating hours, please check the latest facility information.",
    ],

    "goodbye": [
        "Goodbye! Feel free to ask again whenever you need help.",
        "Thank you for using the TARUMT Student Assistance Chatbot.",
        "You are welcome. Have a nice day!",
    ],

    "unknown": [
        (
            "Sorry, that question is outside the current scope of this chatbot. "
            "I can assist with common TARUMT student enquiries."
        ),
        (
            "I'm unable to assist with that topic. "
            "Please try one of the supported TARUMT topics below."
        ),
        (
            "I'm not sure how to answer that question. "
            "This chatbot is designed for common TARUMT student enquiries."
        ),
    ],
}

specific_responses = {
    "campus_facility": {
        "dk x": (
            "lecture block DK X are located nearby library.",
            "or refer to the campus map from this link: https://www.scribd.com/document/339707399/TAR-UC-Map"
        ),

        "dk y": (
            "lecture block DK Y are located nearby library."
            "or refer to the campus map from this link: https://www.scribd.com/document/339707399/TAR-UC-Map"
        ),

        "dk z": (
            "lecture block DK Z are located nearby library."
            "or refer to the campus map from this link: https://www.scribd.com/document/339707399/TAR-UC-Map"
        ),

        "dk aba": (
            "lecture block DK ABA are located at east campus."
            "or refer to the campus map from this link: https://www.scribd.com/document/339707399/TAR-UC-Map"
        ),

        "dk abb": (
            "lecture block DK ABB are located at east campus."
            "or refer to the campus map from this link: https://www.scribd.com/document/339707399/TAR-UC-Map"
        ),

        "dk abc": (
            "lecture block DK ABC are located at east campus."
            "or refer to the campus map from this link: https://www.scribd.com/document/339707399/TAR-UC-Map"
        ),

        "dk abd": (
            "lecture block DK ABD are located at east campus."
            "or refer to the campus map from this link: https://www.scribd.com/document/339707399/TAR-UC-Map"
        ),

        "dk abe": (
            "lecture block DK ABE are located at east campus."
            "or refer to the campus map from this link: https://www.scribd.com/document/339707399/TAR-UC-Map"
        ),

        "dk abf": (
            "lecture block DK ABF are located at east campus."
            "or refer to the campus map from this link: https://www.scribd.com/document/339707399/TAR-UC-Map"
        ),

        "sg 1": (
            "You can locate SG 1 using the Campus Map in the TARC App."
            "or refer to the campus map from this link: https://www.scribd.com/document/339707399/TAR-UC-Map"
        ),

        "sg 2": (
            "You can locate SG 2 using the Campus Map in the TARC App."
            "or refer to the campus map from this link: https://www.scribd.com/document/339707399/TAR-UC-Map"
        ),

        "sg 3": (
            "You can locate SG 3 using the Campus Map in the TARC App."
            "or refer to the campus map from this link: https://www.scribd.com/document/339707399/TAR-UC-Map"
        ),

        "sg 4": (
            "You can locate SG 4 using the Campus Map in the TARC App."
            "or refer to the campus map from this link: https://www.scribd.com/document/339707399/TAR-UC-Map"
        ),

        "library": (
            "You can locate the library using the Campus Map in the TARC App."
            "or refer to the campus map from this link: https://www.scribd.com/document/339707399/TAR-UC-Map"
        ),

        "computer lab": (
            "You can locate computer laboratories using the Campus Map in the TARC App."
            "or refer to the campus map from this link: https://www.scribd.com/document/339707399/TAR-UC-Map"
        ),
    },

    "fees": {
        ("current billing", "bill", "fees now", "fees"): (
            "Open the TARC App and go to "
            "Billing → Current Billing to view "
            "your current charges."
        ),

        "outstanding": (
            "You can check your outstanding fees "
            "under Billing → Current Billing "
            "in the TARC App."
        ),

        ("it", "arts", "foundation", "diploma", "degree", "business"): (
            "You can check fees "
            "from this link https://www.tarc.edu.my/bursary/malaysian-student-fees-guide/ "
            "under the fees guide"
        )

    },

    "programme": {
        ("programme", "structure"): (
            "Log in to Student Intranet using this link https://web.tarc.edu.my/portal/login.jsp and go to "
            "Programme → Programme Structure."
        ),
    },
}

def get_response(message, intent):
    message_lower = message.lower()

    intent_responses = specific_responses.get(
        intent,
        {}
    )

    # Try specific responses first
    for keywords, response in intent_responses.items():

        if isinstance(keywords, str):
            keywords = (keywords,)

        if any(
            keyword in message_lower
            for keyword in keywords
        ):
            return response

     # If not use a general response
    return random.choice(
        responses_data.get(
            intent,
            responses_data["unknown"],
        )
    )