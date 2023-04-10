import pyttsx3, time
import sys
import os

text = sys.argv[1]
engine = pyttsx3.init()
engine.say(text)
engine.runAndWait()