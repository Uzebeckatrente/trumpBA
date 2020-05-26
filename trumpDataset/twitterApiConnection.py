import tweepy
import json
from termcolor import colored

consumer_key = "2JPVwVz9OYaCSAbJuSRC75DHY"
consumer_secret = "NZqEGGRlMPKLkWnSCButQw1KfUpPKLwzcvhRSqDchLSQ1zgoQE"
access_token = "325189589-BAlPOcBAa1MC8nZXnqE2bF2FxZ6q218mQwTWebeF"
acces_token_secret = "NbSuOOD6HOUqnSm31YENKHLvmE1xNtPHa91TboMHOFpye"

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

