from train import create_vectorize_layer, create_model, vectorize_text
import gradio as gr
import random
import time
import pandas as pd
import json

with gr.Blocks() as demo:
    with open("data/respuestas.json", "r", encoding='utf-8') as f:
        respuestas = json.load(f)
    columns = ['text', 'label']
    data = pd.read_csv('data.csv', encoding='utf-8', header=None, names=columns)
    X = data['text'].values
    vectorizer_layer = create_vectorize_layer(X)
    model = create_model()
    model.load_weights('data/model_cpu.keras')
    df_clases = pd.read_csv('data/clases.csv')
    clases = df_clases['clases'].values
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    

    def respond(message, chat_history):
        vectorized_message = vectorize_text([message], vectorizer_layer)
        prediction = model.predict(vectorized_message)
        print(prediction)
        class_index = prediction.argmax(axis=1)[0]
        class_name = clases[class_index]
        print(class_name)
        bot_message = random.choice(respuestas[class_name])
        chat_history.append((message, bot_message))
        time.sleep(2)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()