from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import numpy as np
import torch

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

# Initialize the model and tokenizer

app = Flask(__name__)

# We use the route decorator (@app.route('/')) to specify the URL that should trigger the execution of the home function.
@app.route('/')

# The home function renders the 'home.html' file in the templates folder
def home():
	return render_template('home.html')

@app.route('/generate',methods=['POST'])
def generate():
    model_class=GPT2LMHeadModel
    tokenizer_class=GPT2Tokenizer
    model_name_or_path='gpt2'
    prompt="In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
    length=40
    stop_token=None
    temperature=0.9
    k=40
    p=0.9
    device=torch.device("cpu")
    repetition_penalty=1.0

    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    model = model_class.from_pretrained(model_name_or_path)
    model.to(device)

    length = adjust_length_to_model(length, max_sequence_length=model.config.max_position_embeddings)


    if request.method == 'POST':
        prompt_text = request.form['message'] 
        model_class = request.form['model']

        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
        model = model_class.from_pretrained(args.model_name_or_path)
        model.to(args.device)

        temperature = 
        prompt_text = request.form['message'] 
        length = 
        k=
        p=


        encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(device)

        output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=length,
            temperature=temperature,
            top_k=k,
            top_p=p,
            repetition_penalty=repetition_penalty,
        )
		# Batch size == 1. to add more examples please use num_return_sequences > 1
        generated_sequence = output_sequences[0].tolist()
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        text = text[: text.find(stop_token) if stop_token else None]
    return render_template('result.html',generatedText = text)



if __name__ == '__main__':
	app.run(debug=True)
