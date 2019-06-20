# from flask import Flask
from flask import Flask, request, render_template
import os
import sqlite3 as sql
import pandas as pd
import numpy as np
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
from geopy import distance

app = Flask(__name__)

port = int(os.getenv('PORT', 7000))


myHostname = "anushreeazure.redis.cache.windows.net"
myPassword = "Iq4h8ZDl7kigFTOkH0njINd9LmUAfZFawLjJkB3Gnqw="

r = redis.StrictRedis(host=myHostname,port=6380, db=0, password=myPassword, ssl=True)

# correct
# @app.route('/question6')
# def question6():
#     con = sql.connect("database.db")
#     cur = con.cursor()
#     cur.execute("select distinct net from Earthquake where net like 'n%'")
#     rows = cur.fetchall()
#     start_time = time.time()
#     for i in range(100):
#         print(len(rows))
#         val = random.randint(0, len(rows)-1)
#         str1 = str(rows[val])
#         cur = con.cursor()
#         sqlquery= "select * from Earthquake where net='"+str1[2:4]+"'"
#         print(sqlquery)
#         cur.execute(sqlquery)
#         rows1 = cur.fetchall();
#         r.set(sqlquery, cPickle.dumps(rows1))
#     end_time=time.time()-start_time
#     con.close()
#     return render_template("list1.html",data=rows1, time=end_time)


@app.route('/question6', methods=['POST', 'GET'])
def question6():
    con = sql.connect("database.db")
    cur = con.cursor()
    latstart = float(request.form['lat1'])
    latend = float(request.form['lat2'])
    count = int(request.form['count'])
    lat1 = round(random.uniform(latstart, latend), 2)
    lat2 = round(random.uniform(latstart, latend), 2)
    start_time = time.time()
    cur.execute("select * from Earthquake where mag between " + str(lat1) + " and " + str(lat2))
    rows = cur.fetchall()

    for i in range(count):

        sqlquery = "select * from Earthquake where mag between " + str(lat1) + " and " + str(lat2)
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
    # cur.execute("select * from quakes where latitude between " + str(lat1) + " and " + str(lat2))
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

# correct
# @app.route('/randomrange')
# def randomrange():
#     j = 0
#     start_time = time.time()
#     countCache = 0
#     countwithoutCache = 0
#     for i in range(100):
#         # mag = "{:.2f}".format(random.uniform(1, 8))
#         mag = round(random.uniform(1,8), 2)
#         if r.exists(mag):
#             isCache = 'with Cache'
#             print(isCache,mag)
#             countCache += 1
#             print(countCache)
#             rows = cPickle.loads(r.get(mag))
#             # end_time = time.time() - start_time
#             # r.delete(mag)
#         else:
#
#             isCache = 'without Cache'
#             countwithoutCache += 1
#             print(countwithoutCache)
#             #start_time = time.time()
#             con = sql.connect("database.db")
#             cur = con.cursor()
#             mag1 = str(random.uniform(1, 8))
#             print(isCache, mag1)
#             cur.execute("select * from Earthquake where mag>="+mag1)
#             rows = cur.fetchall();
#             con.close()
#             r.set(mag, cPickle.dumps(rows))
#         j = j+1
#         end_time = time.time() - start_time
#     print(j)
#     return render_template('results.html', data=rows, time=end_time, isCache=isCache, cc= countCache, cc1=countwithoutCache )


@app.route('/randomrange')
def randomrange():
    j = 0
    countCache = 0
    countwithoutCache = 0
    withcachetime = 0
    withoutcachetime = 0
    for i in range(100):
        # mag = "{:.2f}".format(random.uniform(1, 8))
        mag = round(random.uniform(1,8), 2)
        if r.exists(mag):
            isCache = 'with Cache'
            print(isCache, mag)
            countCache += 1
            print(countCache)
            start_time = time.time()
            # rows = cPickle.loads(r.get(mag))
            r.get(mag)
            end_time_withcache = time.time() - start_time
            withcachetime += end_time_withcache
            # r.delete(mag)
        else:

            isCache = 'without Cache'
            countwithoutCache += 1
            print(countwithoutCache)
            #start_time = time.time()
            con = sql.connect("database.db")
            cur = con.cursor()
            mag1 = str(random.uniform(1, 8))
            print(isCache, mag1)
            start_time = time.time()
            cur.execute("select * from Earthquake where mag>="+mag1)
            # rows = cur.fetchall();
            end_timewithoutcache = time.time() - start_time
            withoutcachetime += end_timewithoutcache
            con.close()
            # r.set(mag, cPickle.dumps(rows))
            r.set(mag,1)
        j = j+1
        print(j)
    return render_template('results.html',time1 =withcachetime/countCache, time=withoutcachetime/countwithoutCache, cc= countCache, cc1=countwithoutCache )

@app.route('/randomranges')
def randomranges():
    val = 0
    rows = []
    start_t = time.time()
    for i in range(100):
        val = val + 0.001
        cache_name = 'result' + str(i)
        query = "select * from Earthquake where mag>='"+str(val)+"'"
        if r.get(cache_name):
            t = "with cache"
            store = r.get(cache_name)
        else:
            t = "without cache"
            con = sql.connect("database.db")
            cur = con.cursor()
            cur.execute(query)
            # rows = cur.fetchall()
            con.close()
            # r.set(cache_name, cPickle.dumps(rows))
            r.set(cache_name, 1)
        end_t = time.time() - start_t
        print(end_t)
    return render_template("list.html", time=end_t,cache= t)

#correct
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


@app.route('/clusteringbargraph')
def clusteringbargraph():
    list =[]
    for i in range(0,10,2):
        result = []
        query = "SELECT count(*) FROM Earthquake where mag between " + str(i) + " and " + str(i+2)
        print(query)
        con = sql.connect("database.db")
        cur = con.cursor()
        cur.execute(query)
        rows = cur.fetchone()
        mag_range = str(i)+ "-" +str(i+2)
        result.append(mag_range)
        result.append(rows[0])
        list.append(result)
    y = pd.DataFrame(list)
    X = y.dropna()
    fig = plt.figure()
    # display the values on top of graph
    for i,v in enumerate(X[1]):
        plt.text(i, v, str(v), color='blue', fontweight='bold', horizontalalignment='center')
    # display the legend
    color =['red','green','gold','blue','black']
    for i in range(len(X[0])):
        plt.bar(X[0][i], X[1][i], color=color[i], width=0.2, align='center', label=X[0][i])
    plt.legend()
    # plt.bar(X[0],X[1])
    plt.title('Clusters based on NumberOfEarthquakes and magnitude')
    plt.xlabel('magnitude')
    plt.ylabel('NumberOfEarthquakes')
    plot = convert_fig_to_html(fig)
    return render_template("clus.html",data=plot.decode('utf8'))

@app.route('/barhorizontalgraph')
def barhorizontalgraph():
    list =[]
    for i in range(0,10,2):
        result = []
        query = "SELECT count(*) FROM Earthquake where mag between " + str(i) + " and " + str(i+2)
        print(query)
        con = sql.connect("database.db")
        cur = con.cursor()
        cur.execute(query)
        rows = cur.fetchone()
        mag_range = str(i)+ "-" +str(i+2)
        result.append(mag_range)
        result.append(rows[0])
        list.append(result)
    y = pd.DataFrame(list)
    X = y.dropna()
    fig = plt.figure()
    for i, v in enumerate(X[1]):
        plt.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')

    color = ['red', 'green', 'gold', 'blue', 'black']
    for i in range(len(X[0])):
        plt.bar(X[0][i], X[1][i], color=color[i], width=0.2, align='center', label=X[0][i])
    plt.legend()
    # plt.barh(X[0],X[1], color =["red","green","blue","yellow"])
    plt.title('Clusters based on NumberOfEarthquakes and magnitude')
    # plt.yticks(np.arange(0, 10, step=1))
    plt.xlabel('magnitude')
    plt.ylabel('NumberOfEarthquakes')
    # plt.xlim(0,10000)
    plot = convert_fig_to_html(fig)
    return render_template("clus.html",data=plot.decode('utf8'))


@app.route('/clusteringpiegraph')
def clusteringpiegraph():
    list =[]
    lables = []
    result = []
    for i in range(0,10,2):
        query = "SELECT count(*) FROM Earthquake where mag between " + str(i) + " and " + str(i+2)
        print(query)
        con = sql.connect("database.db")
        cur = con.cursor()
        cur.execute(query)
        rows = cur.fetchone()
        mag_range = str(i)+ "-" +str(i+2)
        lables.append(mag_range)
        # result.append(mag_range)
        result.append(rows[0])
            # fig = plt.figure()
            # plt.pie(result, labels=lables)
            # plt.show()
        # list.append(result)
    # y = pd.DataFrame(list)
    # X = y.dropna()
    print(len(lables),len(result))
    fig = plt.figure()
    plt.pie(result,labels=lables, autopct='%1.0f%%')
    plt.legend()
    plt.title('Clusters based on NumberOfEarthquakes and magnitude')
    plot = convert_fig_to_html(fig)
    return render_template("clus.html",data=plot.decode('utf8'))

@app.route('/linegraph')
def linegraph():
    query = "SELECT latitude,longitude FROM Earthquake "
    con = sql.connect("database.db")
    cur = con.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    y = pd.DataFrame(rows)
    X = y.dropna()
    fig = plt.figure()
    # plt.scatter(X[0],X[1])
    plt.plot(X[0], X[1],marker ='o',markeredgecolor='red', color='purple')
    # X.plot(X[0], X[1],style='o')
    plt.title('Clusters based on latitude and longitude')
    plt.xlabel('latitude')
    plt.ylabel('longitude')
    plot = convert_fig_to_html(fig)
    return render_template("clus.html",data=plot.decode('utf8'))


@app.route('/clusteringhistgraph')
def clusteringhistgraph():
    res = []
    con = sql.connect("database.db")
    cur = con.cursor()
    cur.execute("select mag from Earthquake")
    rows = cur.fetchall()
    row = pd.DataFrame(rows)
    row = row.dropna()
    fig = plt.figure()
    plt.hist(row[0], bins=5)
    plt.title('Based on mag range')
    plt.xlabel('mag range')
    plt.ylabel('frequency')
    plot = convert_fig_to_html(fig)
    print(res)
    return render_template('hist.html', data1=plot.decode('utf8'))














@app.route('/latlong')
def latlong():
    return render_template('latlongin.html')

@app.route('/latlongout', methods=['POST', 'GET'])
def latlongout():
    start_t = time.time()
    lat = request.form['lat']
    lon = request.form['lon']
    dist = float(request.form['dist'])
    query = "select * from Earthquake "
    cache_name = 'result'+str(lat)+str(lon)+str(dist)
    if r.get(cache_name):
        results = cPickle.loads(r.get(cache_name))
        t = 'with cache'
    else :
        con = sql.connect("database.db")
        cur = con.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        t = 'without cache'
        i = 0
        results = []
        while(i < len(rows)):
            dest_lat = rows[i][2]
            dest_lon = rows[i][3]
            distan = distance.distance((lat,lon), (dest_lat,dest_lon)).km
            if(distan<dist):
                results.append(rows[i])
            i = i+1
        r.set(cache_name,cPickle.dumps(results))
    end_t = time.time() - start_t
    print(end_t)
    return render_template("latlong.html",rows=results, time = end_t, cache=t)

@app.route('/testin')
def testin():
    return render_template('place.html')

@app.route('/test1',methods=['GET', 'POST'])
def test1():
#for i in range(100):
    countCache = 0
    countwithoutCache = 0
    start_t = time.time()
    if request.method =='POST':
        place = request.form['place']
        query ='SELECT * FROM Earthquake where "place" LIKE \'%' + place +'%\''
        cache_name = 'result'+ str(place)
        if r.get(cache_name):
            t = "with cache"
            print(t)
            countCache += 1
            print(countCache)
            rows = cPickle.loads(r.get(cache_name))
            #r.delete(cache)

        else:
            t = "without cache"
            print(t)
            countwithoutCache += 1
            print(countwithoutCache)
            con = sql.connect("database.db")
            cur = con.cursor()
            cur.execute(query)
            rows = cur.fetchall()
            con.close()
            r.set(cache_name, cPickle.dumps(rows))
        end_t = time.time() - start_t
        print(end_t)
    return render_template("list.html", rows=rows, time=end_t,cache= t)

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
        file.to_sql('titanic', con, schema=None, if_exists='replace', index=True, index_label=None, chunksize=None,
                    dtype=None)
        con.close()
        return render_template("result.html", msg="Record inserted successfully")



@app.route('/latlat',methods=['POST', 'GET'])
def latlat():
    j = 0
    countCache = 0
    countwithoutCache = 0
    withcachetime = 0
    withoutcachetime = 0
    latstart = float(request.form['lat1'])
    latend = float(request.form['lat2'])
    count = int(request.form['count'])
    lat1 = round(random.uniform(latstart, latend), 2)
    lat2 = round(random.uniform(latstart, latend), 2)
    cache_name = 'result'+str(lat1)+str(lat2)+str(count)
    for i in range(count):
        if r.exists(cache_name):
            isCache = 'with Cache'
            countCache += 1
            start_time = time.time()
            # rows = cPickle.loads(r.get(mag))
            r.get(cache_name)
            end_time_withcache = time.time() - start_time
            withcachetime += end_time_withcache
            # r.delete(mag)
        else:

            isCache = 'without Cache'
            countwithoutCache += 1
            con = sql.connect("database.db")
            cur = con.cursor()
            start_time = time.time()
            cur.execute("select * from quakes where latitude between " + str(lat1) + " and " + str(lat2))
            # rows = cur.fetchall();
            end_timewithoutcache = time.time() - start_time
            withoutcachetime += end_timewithoutcache
            con.close()
            # r.set(mag, cPickle.dumps(rows))
            r.set(cache_name,1)
        j = j+1
        print(j)
    return render_template('results.html', time1=withcachetime/countCache,cache=isCache, time=withoutcachetime/countwithoutCache, cc= countCache, cc1=countwithoutCache )






@app.route('/list', methods=['POST', 'GET'] )
def list():
    start_time = time.time()
    con = sql.connect("database.db")
    cur = con.cursor()
    lat1 = float(request.form['lat1'])
    lat2 = float(request.form['lat2'])
    cur.execute("select time,latitude, mag, place  from quakes where latitude between " + str(lat1) + " and " + str(lat2))
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
