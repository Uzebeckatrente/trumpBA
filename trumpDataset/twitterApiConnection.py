import tweepy
import json
from termcolor import colored

import os

from dotenv import load_dotenv
load_dotenv()
SECRET_KEY = os.getenv("EMAIL")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD")

consumer_key = os.getenv("consumer_key")
consumer_secret = os.getenv("consumer_secret")
access_token = os.getenv("access_token")
acces_token_secret = os.getenv("acces_token_secret")

def createAuthorize():
	try:
		autho = tweepy.OAuthHandler(consumer_key,consumer_secret);
		autho.set_access_token(access_token,acces_token_secret)
		return autho
	except:
		return None

###Start:setup

oAuth = createAuthorize()
api = tweepy.API(oAuth);

