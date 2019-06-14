# from flask import Flask
from flask import Flask, request, render_template
import os
import sqlite3 as sql
import pandas as pd
import time
import redis
import _pickle as cPickle


app = Flask(__name__)

port = int(os.getenv('PORT', 6000))


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
    cur.execute("select * from Earthquake")
    rows = cur.fetchall();
    end_time=time.time()-start_time
    con.close()
    return render_template("list.html", rows=rows, time=end_time)

if __name__ == '__main__':
  app.run(host='127.0.0.1', port=port,debug=True)
