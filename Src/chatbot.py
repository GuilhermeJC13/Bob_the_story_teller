import speech_recognition as sr
import pyttsx3

import re
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, pipeline

class ChatBot:

    def __init__(self):
        self.listener = sr.Recognizer()
        self.engine = pyttsx3.init()
 
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[2].id)

        self.tokenizer = GPT2Tokenizer.from_pretrained(
            'gpt2', bos_token='<start>', eos_token='<end>')

        self.model_path = "Data\checkpoint-23915"

        self.model = GPT2LMHeadModel.from_pretrained(
            self.model_path, eos_token_id=self.tokenizer.eos_token_id, 
            bos_token_id=self.tokenizer.bos_token_id)
        
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model = self.model.to('cpu')


    
    def train(self):
        pass

    def generate_text(self):
        pass

    def talk(self, text, window):

        if text != "":

            text = "<start> " + text

            print("processando : " + text)

            inputs = self.tokenizer(text, return_tensors="pt")

            generation_output = self.model.generate(**inputs, return_dict_in_generate=True, output_scores=True, do_sample=True, min_length = 70,
                                                    max_length=80, top_k=40, pad_token_id=self.tokenizer.eos_token_id,
                                                    temperature=1.0)

            print("finish")

            suggestion = " ".join(self.tokenizer.decode(
            generation_output[0][0]).split()[:])

            suggestion = suggestion.replace('<end>', '\n')
            suggestion = suggestion.replace('<start>', '')
            
            print("generation_output : " + suggestion)

            window.speaking_flag = True

            self.engine.say(suggestion)
            self.engine.runAndWait()

            window.speaking_flag = False
            print(text)
    
    def listen(self, window):
        while True:
            try:
                with sr.Microphone() as source:
                    command = ""
                    self.listener.adjust_for_ambient_noise(source)
                    print('listening...')
                    voice = self.listener.listen(source)
                    command = self.listener.recognize_google(voice)
                    command = command.lower()

                    self.talk(command, window)
            
            except:
                pass