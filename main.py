# from flask import Flask
from flask import Flask, request, render_template
import os
import sqlite3 as sql
import pandas as pd
import time
import redis
import _pickle as cPickle
import random


app = Flask(__name__)

port = int(os.getenv('PORT', 7000))


myHostname = "anushreeazure.redis.cache.windows.net"
myPassword = "Iq4h8ZDl7kigFTOkH0njINd9LmUAfZFawLjJkB3Gnqw="

r = redis.StrictRedis(host=myHostname,port=6380, db=0, password=myPassword, ssl=True)


@app.route('/cachecheck')
def cachecheck():
    cacheName = 'anushreeazure'

    if r.exists(cacheName):
        isCache = 'with Cache'
        start_time = time.time()
        rows = cPickle.loads(r.get(cacheName))
        end_time = time.time()-start_time
        r.delete(cacheName)
    else:
        isCache = 'without Cache'
        start_time = time.time()
        con = sql.connect("database.db")
        cur = con.cursor()
        cur.execute("select * from Earthquake")
        rows = cur.fetchall();
        end_time = time.time() - start_time
        con.close()
        r.set(cacheName, cPickle.dumps(rows))
    return render_template('results.html', data=rows, time=end_time, isCache=isCache)

@app.route('/randomcache')
def randomcache():
    cacheName = 'anushreeazure'

    if r.exists(cacheName):
        isCache = 'with Cache'
        start_time = time.time()
        rows = cPickle.loads(r.get(cacheName))
        end_time = time.time()-start_time
        r.delete(cacheName)
    else:
        isCache = 'without Cache'
        start_time = time.time()
        con = sql.connect("database.db")
        cur = con.cursor()

        for i in range(1000):
            mag1 = str(random.uniform(1, 8))
            mag2 = str(random.uniform(1, 8))
            cur.execute("select * from Earthquake where mag between "+mag1+ " and " +mag2)
            rows = cur.fetchall();

        end_time = time.time() - start_time
        con.close()
        r.set(cacheName, cPickle.dumps(rows))
    return render_template('results.html', data=rows, time=end_time, isCache=isCache)


@app.route('/randomrange')
def randomrange():
    j = 0
    start_time = time.time()
    countCache = 0
    countwithoutCache = 0
    for i in range(100):
        # mag = "{:.2f}".format(random.uniform(1, 8))
        mag = round(random.uniform(1,8), 2)
        if r.exists(mag):
            isCache = 'with Cache'
            print(isCache,mag)
            countCache += 1
            print(countCache)
            rows = cPickle.loads(r.get(mag))
            # end_time = time.time() - start_time
            # r.delete(mag)
        else:

            isCache = 'without Cache'
            countwithoutCache += 1
            #start_time = time.time()
            con = sql.connect("database.db")
            cur = con.cursor()
            mag1 = str(random.uniform(1, 8))
            print(isCache, mag1)
            cur.execute("select * from Earthquake where mag>="+mag1)
            rows = cur.fetchall();
            con.close()
            r.set(mag, cPickle.dumps(rows))
        j = j+1
        end_time = time.time() - start_time
    print(j)
    return render_template('results.html', data=rows, time=end_time, isCache=isCache, cc= countCache, cc1=countwithoutCache )






# r.set('anu','kasal')
# text=r.get('anu')
# print(text)

# @app.route('/')
# def my_form():
#     return render_template('my-form.html')
#
# @app.route('/', methods=['POST'])
# def my_form_post():
#     text = request.form['text']
#     processed_text = text.upper()
#     return processed_text
#
# def hello_world():
#   return 'Hello, World!\n This looks just amazing within 5 minutes'

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/enternew')
def upload_csv():
    return render_template('upload.html')



@app.route('/addrec', methods=['POST', 'GET'])
def addrec():
    if request.method == 'POST':

        con = sql.connect("database.db")
        csv = request.files['myfile']
        file = pd.read_csv(csv)
        file.to_sql('Earthquake', con, schema=None, if_exists='replace', index=True, index_label=None, chunksize=None,
                    dtype=None)
        con.close()
        return render_template("result.html", msg="Record inserted successfully")


@app.route('/list')
def list():
    start_time = time.time()
    con = sql.connect("database.db")
    cur = con.cursor()
    cur.execute("select * from quakes")
    rows = cur.fetchall();
    end_time=time.time()-start_time
    con.close()
    return render_template("list.html", rows=rows, time=end_time, count=(len(rows)))

@app.route('/addnewrec')
def addnewrec():
    return render_template('uploads.html')

@app.route('/addnewrec1', methods=['POST', 'GET'])
def addnewrec1():
    if request.method == 'POST':
        con = sql.connect("database.db")
        csv = request.files['myfile']
        file = pd.read_csv(csv)
        file.to_sql('quakes', con, schema=None, if_exists='replace', index=True, index_label=None, chunksize=None,
                    dtype=None)
        con.close()
        return render_template("result.html", msg="Record inserted successfully")

@app.route('/delete')
def delete():
    con = sql.connect("database.db")
    cur = con.cursor()
    cur.execute("delete from quakes  where rms <=0.25")
    con.close()
    return render_template("list.html")



if __name__ == '__main__':
  app.run(host='127.0.0.1', port=port,debug=True)
