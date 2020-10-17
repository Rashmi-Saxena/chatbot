import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

from flask import Flask,render_template,request

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r',encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "SupBot"
@app.route("/")
def index():
    return render_template("index.html")#to send context to html file
#print("Let's chat! (type 'quit' to exit)")

@app.route("/get")
def get_bot_response():
    #userText =request.args.get("msg")
    while True:
        # sentence = "do you use credit cards?"
        #userText = input("You: ")
        userText =request.args.get("msg")
        if userText == "quit":
            break

        userText = tokenize(userText)
        X = bag_of_words(userText, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    bot_choice = random.choice(intent['responses'])
                    #print(f"{bot_name}:", bot_choice)
                    #return str(bot_choice.get_response(userText))
                    return str(bot_choice)
                    
                    chat_saved = {
                            "chats": [
                                {
                                "tag": tag,
                                "patterns": userText,
                                "responses": bot_choice
                                }]
                            }
                    with open('saved_chat.json', 'a') as json_file:
                        json.dump(chat_saved, json_file)
        else:
            #print(f"{bot_name}: I do not understand...")
            return str("I do not understand")

if __name__=="__main__":
    app.run()
