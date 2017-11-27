import tweepy
from tweepy import OAuthHandler
import pickle as p
import re
import json
from collections import defaultdict
import random
from numpy import arange, where, isin, mean, array, exp, sqrt, dot, pi, diag, concatenate
from bs4 import BeautifulSoup
import requests
from scipy.optimize import curve_fit
from random import uniform
from numpy.random import normal
from matplotlib.pyplot import hist, show, plot, xlim, legend
from datetime.datetime import now

#load twitter keys
twitter_keys=p.load(open('tk.pkl','rb'))

auth = OAuthHandler(twitter_keys['consumer_key'], twitter_keys['consumer_secret'])
auth.set_access_token(twitter_keys['access_token'], twitter_keys['access_secret'])
 
api = tweepy.API(auth)

#grab the time
today = str(now())[:10]

del twitter_keys, auth
while True:
    
    #At midnight start a new day of tweeting
	if today!=str(now())[:10]:
        
        #Grab Trumps tweets and store them in a json dictionary
		hold=[]
        with open('test.txt','w') as f:
            json.dump([], f)
        for i,status in enumerate(limit_handled(tweepy.Cursor(api.user_timeline,'realDonaldTrump').items(10000))):
            hold.append(status._json)
            if i%100==0:
                with open('test.txt','r') as f:
                    a=json.load(f)+hold
                with open('test.txt','w') as f:
                    json.dump(a, f)
                hold=[]

        trump_tweets=json.load(open('test.txt','r'))
        
        #count the number of times Trump tweets each day 
        dt = tweets_per_day(trump_tweets)
        
        #fit a guassian to the number of times trump tweets each day
        params = fit_pdf([i for v in dt.values() for i in v],100,gauss,(10,4))
        
        #draw from distribution to determine the number of times to tweet (num_of_tweets), minimum of once per day.
        num_of_tweets=0
        while num_of_tweets <=0:
            num_of_tweets=int(round(normal(params[0],params[1])))
        num_of_tweets
        
        # determine when Trump usually tweets. It appears to be a bimodal distribution if we look at a 24 hour
        # cycle and start at 7am
        times=tweet_times(trump_tweets)-7 
        for i,t in enumerate(times):
            if t<0:
                times[i]+=24
       
        #fit the tweet time distribution as a bimodal 
        params = fit_pdf(times,arange(0,24,.25),bimodal,(5,1,.1,15,5))
        
        #draw from the tweet time distribution, bot will tweet at each value of times
        times=[]
        for i in arange(num_of_tweets):
            time=25
            while (time<0 or time>24):
                if uniform(0,1) >= params[2]:
                    time = normal(params[3],params[4])
                else:
                    time = normal(params[0],params[1])
                time+=7
                if time>24: time-=24
            times.append(time)
            print(times)

        # free memory, better to reload than leave it since this will run continuously
        del trump_tweets, dt, params, num_of_tweets, time, hold, a, i, status

        for t in sorted(times):
        	if t >= convert_time(str(datetime.now())[11:19]):
                gen_tweet=['.','.','.','.','.']
                while gen_tweet.count('.')>4:
                    try:
                        gen_tweet=nxm_grams(3,1,tweets_to_list(json.load(open('test.txt','r'))))
                    except IndexError:
                        gen_tweet[:3]==['.','.','.','.','.']
                print(gen_tweet)
                #api.update_status(gen_tweet)
            else:
            	time.sleep((convert_time(str(datetime.now())[11:19])-t)*60**2)
        today = str(now())[:10]

    else:
    	time.sleep((24-(convert_time(str(datetime.now())[11:19])))*60**2)


def fit_pdf(data,bins,pdf,expected):
	counts,bounds,_ = hist(data,bins=bins,normed=True)
    bin_c=(bounds[1:]+bounds[:-1])/2
    params,_ =curve_fit(pdf,bin_c,counts,expected)
    return params
        

def limit_handled(cursor):
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError:
            time.sleep(15 * 60)

def tweets_per_day(tweets):
    daily_tweets=defaultdict(list)
    j=0
    day = tweets[0]['created_at'][:3]
    for i,t in enumerate(tweets[:-1]):
        next_day=tweets[i+1]['created_at'][:3]
        j+=1
        if day!=next_day:
            daily_tweets[day].append(j)
            day=next_day
            j=0
    if j!=0:
        j+=1
        daily_tweets[day].append(j)
    else:
        daily_tweets[next_day].append(1)
    return daily_tweets

def gauss(x,mu,sigma):
    return exp(-(x-mu)**2/2/sigma**2)/sqrt(2*pi*sigma**2)

def bimodal(x,mu1,sigma1,w1,mu2,sigma2):
    return w1*gauss(x,mu1,sigma1)+(1-w1)*gauss(x,mu2,sigma2)

def convert_time(t):
    return int(t[:2])+int(t[3:5])/60+int(t[6:8])/3600
    

def tweet_times(tweets):
    times=[]
    for t in tweets:
        created_at = t['created_at']
        col_loc=where([v==':' for v in created_at])[0][0]
        times.append(convert_time(created_at[col_loc-2:col_loc+6]))
    return array(times)

def drop_links(tweet_words):
    try:
        cutoff=where([x=='https' for x in tweet_words])[0][0]
        return tweet_words[:cutoff]
    except IndexError:
        return tweet_words

def grab_links(tweet_list):
    if 'http' in tweet_list[-1]:
        return tweet_list[-1]
    else:
        return ''
    
def fix_symbols(words):
    for i,w in enumerate(words):
        if w =='amp':
            words[i]='&'
            del words[i+1]
        if w =='U' and words[i+1:i+6]==['.','S','.','A','.']:
            words[i]='U.S.A.'
            del words[i+1:i+6]
    return words

def tweets_to_list(tweets):
    tweet_words=[]
    for tweet in tweets:
        if tweet['text'][:2]!='RT':# and not tweet['truncated']:
            link=grab_links(tweet['text'].split())
            tweet_words.extend(drop_links(re.findall(r"[\w']+|[.,!?;]",tweet['text']))+[link])
    return fix_symbols(tweet_words)

def nxm_grams(n,m,words):
    '''
    Generate a sentence using the ngrams method. 
    
    Params:
    n: Words is broken into seqences of length n, n-1 words are then used to predict the next word.
    words: A list of words that generated sentences will be based on.
    '''
    
    if not n>1:raise ValueError("n must be at least 2")
        
    nxmgrams=zip(*[words[i:] for i in arange(n+m)])
    transitions=defaultdict(list)
    starts=[]
    
    for nxmgram in nxmgrams:
        if nxmgram[0] in ['.','!','?']:
            starts.append(nxmgram[1:-m])
        transitions[nxmgram[:-m]].append(nxmgram[-m:])
    
    current= random.choice(starts)
    prev='.'
    result=[*current]
    while True:
        next_candidates=transitions[(prev,*current)]
        next = random.choice(next_candidates)
        prev = current[m-n]
        if n-1-m==0:
            current = (*next,)
        else:
            current =  current[m+1-n:]+(*next,)
        for c in current[-m:]: result.append(c)
        if current[-1] in ['.','!','?'] and len(result)>5: return ' '.join(result)