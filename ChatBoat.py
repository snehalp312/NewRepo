import nltk
from nltk.chat.util import Chat, reflections

# here i've define the Chatboat replies.
patterns = [
    (r'hi|hello|hey', ['Hello!', 'Hi there!']),
    (r'how are you', ['I am just a chatbot.', 'I am doing fine, thank you.']),
    (r'what is your name', ['I am a chatbot.', 'My name is Chatbot.']),
    (r'(.*) weather (.*)', ['The weather is always nice here!', 'I am not a weather bot.']),
    (r'Recommend a good book to read.', ['To Kill a Mockingbird']),
    (r'tell me a joke', ['Why dont scientists trust atoms? Because they make up everything!']),
    (r'Whats the capital of India',['Delhi']),
    (r'recommend me best temple in Pune',['Swaminarayan temple.','iskon temple is also best.']),
    (r'bye|goodbye', ['Goodbye!', 'See you later!']),
]

# Create a chatbot
chatbot = Chat(patterns, reflections)

print("Hello! I'm your chatbot. You can start a conversation. Type 'exit' to end.")

# Interaction loop
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    response = chatbot.respond(user_input)
    print("Chatbot:", response)
