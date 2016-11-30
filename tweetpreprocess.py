import re
import string


def processTweet(tweet):
    # process the tweets

    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@americanair','',tweet)
    tweet = re.sub('@virginamerica','',tweet)
    tweet = re.sub('@united','',tweet)
    tweet = re.sub('@usairways','',tweet)
    tweet = re.sub('@southwestair','',tweet)
    tweet = re.sub('@jetblue','',tweet)

#    tweet = re.sub('@jetblue','',tweet)

    tweet = re.sub('@([^\s]+)',r'\1',tweet)
   # tweet = re.sub('@[^\s]+','',tweet)
 
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    remove = string.punctuation
    remove = remove.replace("-", "") # don't remove hyphens
    pattern = r"[{}]".format(remove) # create the pattern

    txt = ")*^%{}[]thi's - is - @@#!a !%%!!%- test."
    tweet=re.sub(pattern, "", tweet) 
    tweet = tweet.strip('\'"')
    return tweet
#end




import pandas as pd
prec = pd.read_csv("airline.csv")
tweets =  prec['text'].values
sentiment = prec['airline_sentiment'].values
#tweets[2]="ho"
#print tweets[2]
#Read the tweets one by one and process it
#fp = open('tweets.txt', 'r')
#line = fp.readline()
positiveFp = open('tweetspos.txt','w')
negativeFp = open('tweetsneg.txt','w')
neutralFp = open('tweetsneutral.txt','w')
prev= ' '

for i  in range(len(tweets)):
	line = tweets[i]
	if(line == prev):
		#fp2.write('----------------------------------+++++++++++++++++++REPEAT' +'\n')	
			
		#line = fp.readline()
		continue
	else:
		prev = line
		processedTweet = processTweet(line)
	  #  print processedTweet
		sentiments = sentiment[i]
		if(sentiments == 'neutral'):
			neutralFp.write(processedTweet+'\n')
		if(sentiments == 'positive'):
			positiveFp.write(processedTweet+'\n')
		if(sentiments == 'negative'):
			negativeFp.write(processedTweet+'\n')
		#line = fp.readline()
#fp2.close()
#end loop
#	fp.close()

