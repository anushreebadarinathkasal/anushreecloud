# from flask import Flask
from flask import Flask, request, render_template
import os
import sqlite3 as sql
import pandas as pd
import time
import redis
import _pickle as cPickle
import random
import matplotlib
from matplotlib import pyplot as plt
# from matplotlib import *
from sklearn.cluster import KMeans
from scipy.spatial import distance
from numpy import sin, cos, arctan2, sqrt, cross, pi, radians

app = Flask(__name__)

port = int(os.getenv('PORT', 7000))


myHostname = "anushreeazure.redis.cache.windows.net"
myPassword = "Iq4h8ZDl7kigFTOkH0njINd9LmUAfZFawLjJkB3Gnqw="

r = redis.StrictRedis(host=myHostname,port=6380, db=0, password=myPassword, ssl=True)


@app.route('/question6')
def question6():
    con = sql.connect("database.db")
    cur = con.cursor()
    cur.execute("select distinct net from Earthquake where net like 'n%'")
    rows = cur.fetchall()
    start_time = time.time()
    for i in range(100):
        print(len(rows))
        val = random.randint(0, len(rows)-1)
        str1 = str(rows[val])
        cur = con.cursor()
        sqlquery= "select * from Earthquake where net='"+str1[2:4]+"'"
        print(sqlquery)
        cur.execute(sqlquery)
        rows1 = cur.fetchall();
        r.set(sqlquery, cPickle.dumps(rows1))
    end_time=time.time()-start_time
    con.close()
    return render_template("list1.html",data=rows1, time=end_time)

@app.route('/question7')
def question7():
    con = sql.connect("database.db")
    cur = con.cursor()
    cur.execute("select distinct net from Earthquake where net like 'n%'")
    rows = cur.fetchall()
    start_time = time.time()
    for i in range(100):
        print(len(rows))
        val = random.randint(0, len(rows)-1)
        str1 = str(rows[val])
        sqlquery= "select * from Earthquake where net='"+str1[2:4]+"'"
        if(r.get(sqlquery)):
             print("cached data")
    end_time=time.time()-start_time
    con.close()
    return render_template("list1.html", time=end_time)



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


@app.route('/clustering')
def clustering():
    query = "SELECT latitude,longitude FROM Earthquake "
    con = sql.connect("database.db")
    cur = con.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    y = pd.DataFrame(rows)
    k = KMeans(n_clusters=5,random_state=0).fit(y)
    X = y.dropna()
    fig = plt.figure()
    c = k.cluster_centers_
    l = k.labels_
    # plt.scatter(X[0],X[1])
    plt.scatter(X[0], X[1], c = l)
    plt.scatter(c[:, 0], c[:, 1], c='y', s=100, marker='x')
    plt.title('Clusters based on latitude and longitude')
    plt.xlabel('latitude')
    plt.ylabel('longitude')
    plot = convert_fig_to_html(fig)
    return render_template("clus.html",data=plot.decode('utf8'))


def convert_fig_to_html(fig):
    from io import BytesIO
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    import base64
    #figdata_png = base64.b64encode(figfile.read())
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png


@app.route('/clusteringtrial')
def clusteringtrial():
    query = "SELECT latitude,longitude FROM Earthquake "
    con = sql.connect("database.db")
    cur = con.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    y = pd.DataFrame(rows)
    k = KMeans(n_clusters=5, random_state=0).fit(y)
    centroids = k.cluster_centers_
    X = y.dropna()
    print(X[0])
    fig = plt.figure()
    plt.scatter(X[0], X[1])
    plt.scatter(centroids[:,0],centroids[:,1],color='black')
    # print(X[:,0])
    #display popup
    plt.show()
    fig.savefig('static/img.png')
    # print(k.cluster_centers_)
    return render_template("clus.html", data=rows, kmeansCentroid = centroids)




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
    con.commit()
    con.close()
    return render_template("list.html")



if __name__ == '__main__':
  app.run(host='127.0.0.1', port=port,debug=True)
