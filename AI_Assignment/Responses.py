import random
import re

# general responses
responses_data = {

    "greeting": [
        "Hi! What can I help you with today?",
        "Hello! What would you like to know about TARUMT?",
        (
            "Hey! I can help with things like admissions, programmes, "
            "fees, scholarships, class timetables, examinations, "
            "and campus facilities. What do you need?"
        ),
    ],


    "admission": [
        (
            "Sure! Are you looking for entry requirements, "
            "application procedures, required documents, intake "
            "information, or your application status?"
        ),
        (
            "I can help with TARUMT admission information. "
            "Tell me whether you're asking about applying, entry "
            "requirements, documents, intake dates, or application status."
        ),
    ],


    "timetable": [
        (
            "You can check your latest class schedule in the TARC App "
            "under Class Timetable."
        ),
        (
            "For your class schedule, open the TARC App and check "
            "Class Timetable. You should be able to see your lectures "
            "and tutorials there."
        ),
    ],


    "examination": [
        (
            "Sure! Are you looking for your examination timetable, "
            "exam venue, exam date and time, or examination results?"
        ),
        (
            "You can check examination-related information through "
            "the relevant examination section in the TARC App."
        ),
    ],


    "fees": [
        (
            "Sure! Are you asking about programme tuition fees, "
            "your current billing, outstanding fees, or a payment?"
        ),
        (
            "I can help with that. If you mean your own current bill, "
            "check Billing → Current Billing in the TARC App. "
            "If you're asking about programme tuition fees, I can point "
            "you to the fee guide instead."
        ),
    ],


    "scholarship": [
        (
            "Sure! Are you looking for available scholarships, "
            "eligibility requirements, application information, "
            "financial aid, or your scholarship status?"
        ),
        (
            "TARUMT has different scholarship and financial assistance "
            "information. Tell me what part you're looking for and "
            "I'll point you in the right direction."
        ),
    ],


    "programme": [
        (
            "Sure! Are you looking for your programme structure, "
            "modules, curriculum, or information about a programme?"
        ),
        (
            "You can check your own Programme Structure through "
            "Student Intranet. If you're looking for information about "
            "a particular TARUMT programme, tell me the programme name."
        ),
    ],


    "campus_facility": [
        (
            "Sure! Which building or facility are you trying to find? "
            "For example, you can ask me about a lecture block, "
            "library, computer lab, cafeteria, or study area."
        ),
        (
            "I can help you find your way around the KL Main Campus. "
            "Tell me the building code or facility you're looking for."
        ),
    ],


    "goodbye": [
        "You're welcome! Have a great day.",
        "No problem! Glad I could help.",
        "Anytime! All the best with your studies.",
        "You're welcome. Feel free to come back if you need anything else.",
    ],


    "unknown": [
        (
            "I'm not quite sure what you mean. Could you rephrase "
            "the question or give me a little more detail?"
        ),
        (
            "I don't have enough information to answer that reliably. "
            "Could you tell me a little more about what you're looking for?"
        ),
        (
            "I'm not sure about that one. If it's related to TARUMT, "
            "try giving me the specific service, programme, or issue "
            "you're asking about."
        ),
    ],
}

# specific responses
specific_responses = {

    "admission": {

        (
            "entry requirement",
            "entry requirements",
            "minimum requirement",
            "minimum requirements",
            "academic requirement",
            "academic requirements",
            "qualification required",
            "qualifications required",
        ): (
            "Entry requirements depend on the programme you're applying for. You can check the requirements for your chosen programme on TARUMT's official programme or admission information page.\n\n"
            "Programme / admission link: https://www.tarc.edu.my/admissions/programmes/programme-offered-a-z/undergraduate-programme/"
        ),


        (
            "how to apply",
            "apply to tarumt",
            "apply for tarumt",
            "application procedure",
            "application procedures",
            "online application",
            "submit application",
            "start application",
        ): (
            "You can apply through TARUMT's Online Application system. "
            "I'd recommend checking the required documents before submitting your application so you don't miss anything.\n\n"
            "Online Application: https://www.tarc.edu.my/account/login.jsp?fappcode=online-app"
        ),


        (
            "application status",
            "check application",
            "application result",
            "application outcome",
            "offer status",
            "admission status",
        ): (
            "If you've already applied, log in to your TARUMT Online Application account to check your application status. "
            "Keep an eye on your email as well, since application updates may also be sent there.\n\n"
            "Online Application: https://www.tarc.edu.my/account/login.jsp?fappcode=online-app"
        ),


        (
            "required document",
            "required documents",
            "supporting document",
            "supporting documents",
            "upload document",
            "upload documents",
            "certificate",
            "certificates",
        ): (
            "The supporting documents depend on your application. Check the document requirements in the TARUMT Online Application system before submitting them.\n\n"
            "Online Application: https://www.tarc.edu.my/account/login.jsp?fappcode=online-app"
        ),


        (
            "intake",
            "intake date",
            "intake dates",
            "next intake",
            "when can i apply",
        ): (
            "You can check TARUMT's latest intake and application information on the official admissions page.\n\n"
            "Intake information: https://dace.tarc.edu.my/programmes/intakes"
        ),
    },

    "timetable": {

        (
            "class timetable",
            "class schedule",
            "lecture schedule",
            "tutorial schedule",
            "class time",
            "lecture time",
            "tutorial time",
            "next class",
            "today's class",
            "todays class",
        ): (
            "You can check that in the TARC App. Open Class Timetable to see your latest lecture and tutorial schedule."
        ),


        (
            "rescheduled class",
            "class rescheduled",
            "class changed",
            "lecture changed",
            "tutorial changed",
            "replacement class",
            "new class time",
        ): (
            "If your class has been rescheduled, check the latest Class Timetable in the TARC App to confirm the updated time and venue."
        ),
    },

    "examination": {

        (
            "exam timetable",
            "examination timetable",
            "exam schedule",
            "examination schedule",
            "exam date",
            "examination date",
            "exam time",
            "examination time",
        ): (
            "You can check your examination timetable in the TARC App. It should show the date, time, and venue for your papers."
        ),


        (
            "exam venue",
            "examination venue",
            "exam room",
            "examination room",
            "where is my exam",
            "where is my examination",
        ): (
            "Your exam venue should be listed in your examination timetable. Check the latest version in the TARC App before heading to the venue."
        ),


        (
            "exam result",
            "exam results",
            "examination result",
            "examination results",
            "final result",
            "final results",
            "result release",
        ): (
            "You can check your examination results through the relevant results section in the TARC App once they have been released."
        ),
    },

    "fees": {
        (
            "diploma fee",
            "diploma fees",
            "diploma tuition",
            "fees for diploma",
            "fee for diploma",
            "diploma programme fee",
        ): (
            "If you're checking Diploma tuition fees, the latest amounts are listed in TARUMT's Malaysian Student Fees Guide. You can check the Diploma section there.\n\n"
            "Fees Guide: https://www.tarc.edu.my/bursary/malaysian-student-fees-guide/"
        ),


        (
            "foundation fee",
            "foundation fees",
            "foundation tuition",
            "fees for foundation",
            "fee for foundation",
        ): (
            "For Foundation programme fees, check TARUMT's latest Malaysian Student Fees Guide and look under the Foundation section.\n\n"
            "Fees Guide: https://www.tarc.edu.my/bursary/malaysian-student-fees-guide/"
        ),


        (
            "degree fee",
            "degree fees",
            "degree tuition",
            "bachelor fee",
            "bachelor fees",
            "bachelor tuition",
            "fees for degree",
            "fee for degree",
        ): (
            "You can find the latest Degree tuition fees in TARUMT's Malaysian Student Fees Guide. Check the section for the programme you're interested in.\n\n"
            "Fees Guide: https://www.tarc.edu.my/bursary/malaysian-student-fees-guide/"
        ),


        (
            "programme fee",
            "programme fees",
            "course fee",
            "course fees",
            "tuition fee",
            "tuition fees",
            "study fee",
            "study fees",
        ): (
            "For programme tuition fees, the best place to check is TARUMT's latest Student Fees Guide, since the amount varies by programme.\n\n"
            "Fees Guide: https://www.tarc.edu.my/bursary/malaysian-student-fees-guide/"
        ),


        (
            "current billing",
            "current bill",
            "my bill",
            "my billing",
            "fee balance",
            "fees balance",
            "amount due",
            "amount i owe",
            "how much i owe",
        ): (
            "If you're checking your own bill, open the TARC App and go to Billing → Current Billing. You'll be able to see your latest charges and balance there."
        ),


        (
            "outstanding fee",
            "outstanding fees",
            "outstanding balance",
            "unpaid fee",
            "unpaid fees",
            "remaining fee",
            "remaining fees",
        ): (
            "You can check any outstanding amount in the TARC App under Billing → Current Billing. Your remaining balance should be shown there."
        ),


        (
            "payment status",
            "payment pending",
            "pending payment",
            "payment recorded",
            "payment successful",
            "payment failed",
            "paid already",
            "already paid",
        ): (
            "If you've already made a payment, check Billing → Current Billing in the TARC App to see whether it has been reflected in your account. If it still looks wrong, you may need to contact the relevant TARUMT office."
        ),


        (
            "payment history",
            "payment record",
            "payment records",
            "receipt",
            "transaction",
        ): (
            "If you're looking for a previous payment or transaction, check the billing information available in the TARC App. "
            "For an official payment record or an unresolved transaction, you may need to contact the relevant TARUMT office."
        ),
    },

    "scholarship": {

        (
            "merit scholarship",
            "merit scholarships",
        ): (
            "If you're looking for the Merit Scholarship, you can check the latest eligibility requirements and details on TARUMT's official scholarship page.\n\n"
            "Merit Scholarship: https://www.tarc.edu.my/admissions/a/merit-scholarship/"
        ),


        (
            "scholarship eligibility",
            "eligible for scholarship",
            "qualify for scholarship",
            "scholarship requirement",
            "scholarship requirements",
            "minimum cgpa",
        ): (
            "Scholarship requirements can differ depending on the award. Check the eligibility criteria for the scholarship you're interested in before applying.\n\n"
            "Scholarship information: https://www.tarc.edu.my/admissions/a/merit-scholarship/"
        ),


        (
            "financial aid",
            "financial assistance",
            "financial support",
            "study assistance",
            "funding assistance",
        ): (
            "If you need financial assistance, check TARUMT's latest scholarship and financial aid information to see which options you may be eligible for.\n\n"
            "Financial aid information: https://www.tarc.edu.my/dsa/financial-aid/financial-aid/"
        ),


        (
            "scholarship deadline",
            "scholarship closing date",
            "application deadline",
        ): (
            "Scholarship application periods can vary, so I'd recommend checking the latest deadline on TARUMT's official scholarship page before applying.\n\n"
            "Scholarship information: https://www.tarc.edu.my/admissions/a/merit-scholarship/"
        ),


        (
            "scholarship status",
            "scholarship application status",
            "funding status",
            "financial aid status",
        ): (
            "If you've already applied for a scholarship or financial aid, "
            "check the application channel or instructions provided for that particular award. If no status is shown, contact the relevant TARUMT office for an update."
        ),
    },

    "programme": {

        (
            "programme structure",
            "program structure",
            "course structure",
            "curriculum",
            "module structure",
            "subject structure",
        ): (
            "You can check your Programme Structure in Student Intranet. "
            "After logging in, go to Programme → Programme Structure to see your modules and semester arrangement.\n\n"
            "Student Intranet: https://web.tarc.edu.my/portal/login.jsp"
        ),


        (
            "my modules",
            "my subjects",
            "module list",
            "subject list",
            "modules this semester",
            "subjects this semester",
        ): (
            "You can check the modules in your programme through Student Intranet under Programme → Programme Structure. It shows how your subjects are arranged across semesters."
        ),


        (
            "elective",
            "electives",
            "compulsory module",
            "compulsory modules",
            "compulsory subject",
            "compulsory subjects",
        ): (
            "Check your Programme Structure in Student Intranet to see the modules included in your programme. For detailed information about compulsory or elective requirements, refer to your programme information or faculty."
        ),


        (
            "programme duration",
            "course duration",
            "how many years",
            "how many semesters",
        ): (
            "Programme duration depends on the course you're interested in. "
            "You can check the official programme page for the latest duration and structure.\n\n"
            "Programme information: https://www.tarc.edu.my/admissions/programmes/programme-offered-a-z/undergraduate-programme/"
        ),
    },

    "campus_facility": {
        (
            "dk aba",
            "dk abb",
            "dk abc",
            "dk abd",
            "dk abe",
            "dk abf",
        ): (
            "DK ABA-ABF are lecture blocks at the KL Main Campus. They're located around the East Campus area. You can use the campus map to locate the exact block.\n\n"
            "Campus Map: https://www.scribd.com/document/339707399/TAR-UC-Map"
        ),


        (
            "dk x",
            "dk y",
            "dk z",
        ): (
            "DK X, DK Y and DK Z are lecture blocks around the library area. Check the campus map for the exact block and route.\n\n"
            "Campus Map: https://www.scribd.com/document/339707399/TAR-UC-Map"
        ),


        (
            "sg a",
            "sg b",
            "sg c",
            "sg d",
            "sg e",
            "sg f",
        ): (
            "That's one of the SG A-F blocks. You can check its exact location on the KL Main Campus map.\n\n"
            "Campus Map: https://www.scribd.com/document/339707399/TAR-UC-Map"
        ),


        (
            "sg 1",
            "sg 2",
            "sg 3",
            "sg 4",
        ): (
            "That's one of the SG 1-4 blocks. The easiest way to find the exact location is through the KL Main Campus map.\n\n"
            "Campus Map: https://www.scribd.com/document/339707399/TAR-UC-Map"
        ),


        (
            "block a",
            "block b",
            "block c",
            "block d",
            "block e",
            "block f",
            "block g",
            "block h",
            "block i",
            "block j",
            "block k",
            "block l",
        ): (
            "You can find Blocks A-L on the KL Main Campus map. "
            "Check the block shown on your timetable and use the map to locate it.\n\n"
            "Campus Map: https://www.scribd.com/document/339707399/TAR-UC-Map"
        ),


        (
            "library",
            "perpustakaan",
        ): (
            "Looking for the library? You can use the KL Main Campus map to locate it and plan your route from your current block.\n\n"
            "Campus Map: https://www.scribd.com/document/339707399/TAR-UC-Map"
        ),


        (
            "computer lab",
            "computer laboratory",
            "computer laboratories",
            "computer room",
        ): (
            "If you need a computer lab, check the campus map or the relevant faculty/facility information for the nearest available computer laboratory.\n\n"
            "Campus Map: https://www.scribd.com/document/339707399/TAR-UC-Map"
        ),


        (
            "cafeteria",
            "canteen",
            "food",
            "eat",
            "lunch",
        ): (
            "Looking for somewhere to eat? You can check the campus map for cafeterias and food facilities around the KL Main Campus.\n\n"
            "Campus Map: https://www.scribd.com/document/339707399/TAR-UC-Map"
        ),


        (
            "study area",
            "study space",
            "study room",
            "quiet place",
            "quiet area",
        ): (
            "If you're looking for somewhere to study, check the campus map for the library and available student study areas. "
            "Facility availability may vary."
        ),


        (
            "printing",
            "printer",
            "print assignment",
            "photocopy",
            "photocopying",
        ): (
            "If you need printing or photocopying, check the relevant student facilities around campus or use the campus map to find a nearby facility."
        ),
    },
}


def contains_keyword(message: str, keyword: str) -> bool:
# Match a keyword as a word/phrase rather than accidentally matching
# tiny strings inside unrelated words.
# Ex: it should not match facility

    pattern = r"\b" + re.escape(keyword.lower()) + r"\b"

    return re.search(
        pattern,
        message.lower(),
    ) is not None


def get_response(message: str, intent: str,) -> str:

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
            contains_keyword(message, keyword)
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

