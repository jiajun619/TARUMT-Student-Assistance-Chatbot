# Training and unseen questions for the unknown intent.

unknown_data = [
    {
        "question": 'What is the weather today?',
        "intent": 'unknown',
    },
    {
        "question": 'Will it rain tomorrow?',
        "intent": 'unknown',
    },
    {
        "question": 'What is the temperature outside?',
        "intent": 'unknown',
    },
    {
        "question": 'Who is the president?',
        "intent": 'unknown',
    },
    {
        "question": 'Tell me a joke',
        "intent": 'unknown',
    },
    {
        "question": 'What is your favourite movie?',
        "intent": 'unknown',
    },
    {
        "question": 'Can you play music?',
        "intent": 'unknown',
    },
    {
        "question": 'What is the capital of Japan?',
        "intent": 'unknown',
    },
    {
        "question": 'How do I cook fried rice?',
        "intent": 'unknown',
    },
    {
        "question": 'What time is the football match?',
        "intent": 'unknown',
    },
    {
        "question": 'Who won the game yesterday?',
        "intent": 'unknown',
    },
    {
        "question": 'Can you order food for me?',
        "intent": 'unknown',
    },
    {
        "question": 'What is two plus two?',
        "intent": 'unknown',
    },
    {
        "question": 'Can you recommend a restaurant?',
        "intent": 'unknown',
    },
    {
        "question": 'How old are you?',
        "intent": 'unknown',
    },
    {
        "question": 'What is your name?',
        "intent": 'unknown',
    },
    {
        "question": 'Can you sing a song?',
        "intent": 'unknown',
    },
    {
        "question": 'Where should I go for vacation?',
        "intent": 'unknown',
    },
    {
        "question": 'What is the latest news?',
        "intent": 'unknown',
    },
    {
        "question": 'Can you translate this sentence?',
        "intent": 'unknown',
    },
    {
        "question": 'How do I repair my computer?',
        "intent": 'unknown',
    },
    {
        "question": 'What phone should I buy?',
        "intent": 'unknown',
    },
    {
        "question": 'Can you book a hotel?',
        "intent": 'unknown',
    },
    {
        "question": 'What is Bitcoin?',
        "intent": 'unknown',
    },
    {
        "question": 'Tell me about dinosaurs',
        "intent": 'unknown',
    },
    {
        "question": 'How do I reset my student email password?',
        "intent": 'unknown',
    },
    {
        "question": 'How can I join a student club?',
        "intent": 'unknown',
    },
    {
        "question": 'Where can I apply for an internship?',
        "intent": 'unknown',
    },
    {
        "question": 'How do I get my student ID card?',
        "intent": 'unknown',
    },
    {
        "question": 'How can I book counselling?',
        "intent": 'unknown',
    },
    {
        "question": "Where can I find my lecturer's email?",
        "intent": 'unknown',
    },
    {
        "question": 'How do I apply for hostel accommodation?',
        "intent": 'unknown',
    },
    {
        "question": 'How do I connect to campus WiFi?',
        "intent": 'unknown',
    },
    {
        "question": 'How can I borrow sports equipment?',
        "intent": 'unknown',
    },
    {
        "question": 'Where do I report a lost student card?',
        "intent": 'unknown',
    },
    {
        "question": 'How do I access my student email?',
        "intent": 'unknown',
    },
    {
        "question": 'Can I change my student portal password?',
        "intent": 'unknown',
    },
    {
        "question": 'Where can I ask about internship placement?',
        "intent": 'unknown',
    },
    {
        "question": 'How do I join a university society?',
        "intent": 'unknown',
    },
    {
        "question": 'Is counselling available for students?',
        "intent": 'unknown',
    },
    {
        "question": 'Where can I park my car on campus?',
        "intent": 'unknown',
    },
    {
        "question": 'How do I appeal a disciplinary decision?',
        "intent": 'unknown',
    },
    {
        "question": 'Can I book a sports court?',
        "intent": 'unknown',
    },
    {
        "question": 'How do I request an official transcript?',
        "intent": 'unknown',
    },
    {
        "question": 'Where can I collect my graduation robe?',
        "intent": 'unknown',
    },
    {
        "question": 'Where can I change my TARC App password?',
        "intent": 'unknown',
    },
    {
        "question": 'How do I activate my student email account?',
        "intent": 'unknown',
    },
    {
        "question": 'Can I reserve a parking space on campus?',
        "intent": 'unknown',
    },
    {
        "question": 'How do I join orientation activities?',
        "intent": 'unknown',
    },
    {
        "question": 'Where can I submit a complaint about a lecturer?',
        "intent": 'unknown',
    },
    {
        "question": 'How do I request a replacement student card?',
        "intent": 'unknown',
    },
    {
        "question": 'Can I apply for an internship through the university?',
        "intent": 'unknown',
    },
    {
        "question": 'Where can I get help with campus WiFi problems?',
        "intent": 'unknown',
    },
    {
        "question": 'How do I register for a student society?',
        "intent": 'unknown',
    },
    {
        "question": 'Where can I find career counselling services?',
        "intent": 'unknown',
    },
    {
        "question": 'How do I apply for graduation?',
        "intent": 'unknown',
    },
    {
        "question": 'Can I request an official confirmation letter?',
        "intent": 'unknown',
    },
    {
        "question": 'Where do I collect my student ID?',
        "intent": 'unknown',
    },
    {
        "question": 'How do I make an appointment with a counsellor?',
        "intent": 'unknown',
    },
    {
        "question": 'Can I reserve a discussion room online?',
        "intent": 'unknown',
    },
    {
        "question": 'How do I change my emergency contact details?',
        "intent": 'unknown',
    },
    {
        "question": 'Where can I get information about graduation ceremony dates?',
        "intent": 'unknown',
    },
    {
        "question": 'How do I apply for a student exchange programme?',
        "intent": 'unknown',
    },
    {
        "question": 'Where can I report a technical problem with the student portal?',
        "intent": 'unknown',
    },
    {
        "question": 'How do I update my home address in university records?',
        "intent": 'unknown',
    },
    {
        "question": 'Can I request a letter confirming I am a student?',
        "intent": 'unknown',
    },
    {
        "question": 'Where can I ask about bus or transport services?',
        "intent": 'unknown',
    },
    {
        "question": 'How do I register for co-curricular activities?',
        "intent": 'unknown',
    },
    {
        "question": 'Where can I get help with internship documents?',
        "intent": 'unknown',
    },
    {
        "question": 'How can I contact the IT helpdesk?',
        "intent": 'unknown',
    },
]


unknown_unseen_data = [
    {
        "question": 'What is the weather forecast for this weekend?',
        "intent": 'unknown',
    },
    {
        "question": 'Can you recommend a movie to watch?',
        "intent": 'unknown',
    },
    {
        "question": 'Who won the football competition yesterday?',
        "intent": 'unknown',
    },
    {
        "question": 'How do I cook chicken curry?',
        "intent": 'unknown',
    },
    {
        "question": 'What is the latest world news?',
        "intent": 'unknown',
    },
    {
        "question": 'I forgot the password for my university email account',
        "intent": 'unknown',
    },
    {
        "question": 'Where do students sign up for clubs and societies?',
        "intent": 'unknown',
    },
    {
        "question": 'Who should I contact about industrial training placement?',
        "intent": 'unknown',
    },
    {
        "question": 'I lost my student card, what should I do?',
        "intent": 'unknown',
    },
    {
        "question": 'How can I reserve a university sports facility?',
        "intent": 'unknown',
    },
    {
        "question": 'Where do I request an academic transcript?',
        "intent": 'unknown',
    },
    {
        "question": 'Is there student counselling and how do I make an appointment?',
        "intent": 'unknown',
    },
    {
        "question": 'I need help connecting my laptop to campus wireless internet',
        "intent": 'unknown',
    },
    {
        "question": 'My student email account is locked, who can help me?',
        "intent": 'unknown',
    },
    {
        "question": 'Where can I replace a lost student ID card?',
        "intent": 'unknown',
    },
    {
        "question": 'How can I sign up for a campus club?',
        "intent": 'unknown',
    },
    {
        "question": 'I need help with university WiFi, where should I go?',
        "intent": 'unknown',
    },
    {
        "question": 'How do I register for graduation?',
        "intent": 'unknown',
    },
    {
        "question": 'Where do I report a problem with my student portal account?',
        "intent": 'unknown',
    },
    {
        "question": 'How can I request a student status confirmation letter?',
        "intent": 'unknown',
    },
    {
        "question": 'Who can help me update my personal address?',
        "intent": 'unknown',
    },
]
