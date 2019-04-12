from __future__ import print_function
from PIL import Image,ImageDraw
from math import *
import random
import sys
warned_of_error = False
import csv
from nltk.corpus import stopwords
import pygame
import simplejson
from pytagcloud import create_tag_image, make_tags


def readfile(file_name):
    f = open(file_name)
    lines=[line for line in f]

    # First line is the column titles
    colnames=lines[0].strip().split('\t')[:]
    print(colnames)
    rownames=[]
    data=[]
    for line in lines[1:]:
        p=line.strip().split('\t')
        # First column in each row is the rowname
        if len(p)>1:
            rownames.append(p[0])
            # The data for this row is the remainder of the row
            data.append([float(x) for x in p[1:]])
    return rownames,colnames,data

def readfile_country(file_name):
    f = open(file_name)
    lines=[line for line in f]
  
    # First line is the column titles
    colnames=lines[0].strip().split(',')[:]
    rownames1=[]
    rownames2=[]
    data=[]
    for line in lines[1:]:
        p=line.strip().split(',')
        # First column in each row is the rowname
        if len(p)>2:
            rownames1.append(p[0])
            rownames2.append(p[1])
            # The data for this row is the remainder of the row
            data_line = []
            for x in p[2:]:
                if x != '':
                    data_line.append(float(x))
                else:
                    data_line.append(-1)
            data.append(data_line[:-1]) #delete isregion

    return rownames1,rownames2,colnames,data

def data_preparation(file_name):
    read_result = readfile_country(file_name)
    rownames1 = read_result[0]
    rownames2 = read_result[1]
    colnames = read_result[2]
    data = read_result[3]

    line_pos = 0
    for line in data:
        data_index = []
        missing_index = []
        for i in range(len(line)):
            if line[i] != -1:
                data_index.append(i)
            else:
                missing_index.append(i)
        if len(data_index) != len(line):
            distance = [0]*len(data)
            pos = 0
            invalid = False
            for line1 in data:
                sum = 0
                for i in data_index:
                    if line1[i] == -1:
                        distance[pos] = sys.maxsize
                        invalid = True
                        pos += 1
                        break
                    sum += (line1[i]-line[i])*(line1[i]-line[i])
                if invalid:
                    invalid = False
                    continue
                distance[pos] = sum
                pos += 1
            nearest = [line_pos, line_pos, line_pos, line_pos]
            for time in range(3):
                min = sys.maxsize
                for i in range(len(distance)):
                    if i not in nearest:
                        if distance[i] < min:
                            min = distance[i]
                            nearest[time] = i
            for i in missing_index:
                sum = 0
                added = 0
                for j in range(3):
                    if data[nearest[j]][i] == -1:
                        continue
                    sum += data[nearest[j]][i]
                    added += 1
                data[line_pos][i] = round(sum/added, 0)
        line_pos += 1

    return rownames1, rownames2, colnames, data

def rotatematrix(data):
    newdata=[]
    for i in range(len(data[0])):
        newrow=[data[j][i] for j in range(len(data))]
        newdata.append(newrow)
    return newdata


def print_2d_array(matrix):
    for i in range(len(matrix)):
        for j in range (len(matrix[i])):
            print (matrix[i][j], end = "")
        print('\n')


# different similarity metrics for 2 vectors
def manhattan(v1,v2):
    res=0
    dimensions=min(len(v1),len(v2))

    for i in range(dimensions):
        res+=abs(v1[i]-v2[i])

    return res


def euclidean(v1,v2):
    res=0
    dimensions=min(len(v1),len(v2))
    for i in range(dimensions):
        res+=pow(abs(v1[i]-v2[i]),2)

    return sqrt(float(res))


def cosine(v1,v2):
    dotproduct=0
    dimensions=min(len(v1),len(v2))

    for i in range(dimensions):
        dotproduct+=v1[i]*v2[i]

    v1len=0
    v2len=0
    for i in range (dimensions):
        v1len+=v1[i]*v1[i]
        v2len+=v2[i]*v2[i]

    v1len=sqrt(v1len)
    v2len=sqrt(v2len)

    return 1.0-(float(dotproduct)/(v1len*v2len))
  

def pearson(v1,v2):
    # Simple sums
    sum1=sum(v1)
    sum2=sum(v2)
  
    # Sums of the squares
    sum1Sq=sum([pow(v,2) for v in v1])
    sum2Sq=sum([pow(v,2) for v in v2])
  
    # Sum of the products
    pSum=sum([v1[i]*v2[i] for i in range(min(len(v1),len(v2)))])
  
    # Calculate r (Pearson score)
    num=pSum-(sum1*sum2/len(v1))
    den=sqrt((sum1Sq-pow(sum1,2)/len(v1))*(sum2Sq-pow(sum2,2)/len(v1)))
    if den==0: return 1.0
  
    return 1.0-num/den


def tanimoto(v1,v2):
    c1,c2,shr=0,0,0

    for i in range(len(v1)):
        if v1[i]!=0: c1+=1 # in v1
        if v2[i]!=0: c2+=1 # in v2
        if v1[i]!=0 and v2[i]!=0: shr+=1 # in both

    return 1.0-(float(shr)/(c1+c2-shr))


# Hierarchical clustering
class bicluster:
    def __init__(self,vec,left=None,right=None,distance=0.0,id=None):
        self.left=left
        self.right=right
        self.vec=vec
        self.id=id
        self.distance=distance

id_list = []
def find_all(current):
    if current.id >= 0:
        id_list.append(current.id)
    else:
        if current.left != None:
            find_all(current.left)
        if current.right!= None:
            find_all(current.right)


def min_dis(rows, distance, clust, i, j):
    global id_list
    id_list = []
    find_all(clust[i])
    i_id = id_list.copy()

    id_list = []
    find_all(clust[j])
    j_id = id_list.copy()

    mini = sys.maxsize
    for h in i_id:
        for k in j_id:
            d = distance(rows[h], rows[k])
            if d<mini:
                mini = d

    return mini

def max_dis(rows, distance, clust, i, j):
    global id_list
    id_list = []
    find_all(clust[i])
    i_id = id_list.copy()

    id_list = []
    find_all(clust[j])
    j_id = id_list.copy()

    maxi = -1
    for h in i_id:
        for k in j_id:
            d = distance(rows[h], rows[k])
            if d>maxi:
                maxi = d

    return maxi

def ave_dis(rows, distance, clust, i, j):
    global id_list
    id_list = []
    find_all(clust[i])
    i_id = id_list.copy()

    id_list = []
    find_all(clust[j])
    j_id = id_list.copy()

    num = 0
    sum = 0
    for h in i_id:
        for k in j_id:
            sum += distance(rows[h], rows[k])
            num += 1

    return sum/num

def centroid_dis(rows, distance, clust, i, j):
    return distance(clust[i].vec,clust[j].vec)

def hcluster(rows, inter_dis, distance=euclidean):
    distances={}
    currentclustid=-1

    # Clusters are initially just the rows
    clust=[bicluster(rows[i],id=i) for i in range(len(rows))]

    while len(clust)>1:
        lowestpair=(0,1)
        closest=distance(clust[0].vec,clust[1].vec)

        # loop through every pair looking for the smallest distance
        for i in range(len(clust)):
            for j in range(i+1,len(clust)):
                # distances is the cache of distance calculations
                if (clust[i].id,clust[j].id) not in distances:
                    distances[(clust[i].id,clust[j].id)]=inter_dis(rows, distance, clust, i, j)

                d=distances[(clust[i].id,clust[j].id)]

                if d<closest:
                    closest=d
                    lowestpair=(i,j)

        # calculate the average of the two clusters
        mergevec=[
            (clust[lowestpair[0]].vec[i]+clust[lowestpair[1]].vec[i])/2.0
                    for i in range(len(clust[0].vec))]

        # create the new cluster
        newcluster=bicluster(mergevec,left=clust[lowestpair[0]],
                             right=clust[lowestpair[1]],
                             distance=closest,id=currentclustid)

        # cluster ids that weren't in the original set are negative
        currentclustid-=1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(newcluster)

    return clust[0]

def best_inter_distance_for_countries():
    data_result = data_preparation("dataset.csv")
    rownames1 = data_result[0]
    rownames2 = data_result[1]
    colnames = data_result[2]
    data = data_result[3]

    clust=hcluster(data,min_dis, euclidean)
    drawdendrogram(clust,rownames2,jpeg='countries_MIN.jpg')

    clust=hcluster(data,max_dis, euclidean)
    drawdendrogram(clust,rownames2,jpeg='countries_MAX.jpg')

    clust=hcluster(data,ave_dis, euclidean)
    drawdendrogram(clust,rownames2,jpeg='countries_group_average.jpg')

    clust=hcluster(data,centroid_dis, euclidean)
    drawdendrogram(clust,rownames2,jpeg='countries_centroid_distance.jpg')


def printhclust(clust,labels=None,n=0):
    # indent to make a hierarchy layout
    for i in range(n):
        print (' ', end="")
    if clust.id<0:
    # negative id means that this is branch
        print ('-')
    else:
    # positive id means that this is an endpoint
        if labels==None: print (clust.id)
        else: print (labels[clust.id])

    # now print the right and left branches
    if clust.left!=None: printhclust(clust.left,labels=labels,n=n+1)
    if clust.right!=None: printhclust(clust.right,labels=labels,n=n+1)


# draw hierarchical clusters
def getheight(clust):
    # Is this an endpoint? Then the height is just 1
    if clust.left==None and clust.right==None: return 1

    # Otherwise the height is the same of the heights of
    # each branch
    return getheight(clust.left)+getheight(clust.right)


def getdepth(clust):
    # The distance of an endpoint is 0.0
    if clust.left==None and clust.right==None: return 0

    # The distance of a branch is the greater of its two sides
    # plus its own distance
    return max(getdepth(clust.left),getdepth(clust.right))+clust.distance


def drawdendrogram(clust,labels,jpeg='clusters.jpg'):
    # height and width
    h=getheight(clust)*20
    w=1200
    depth=getdepth(clust)

    # width is fixed, so scale distances accordingly
    scaling=float(w-150)/depth

    # Create a new image with a white background
    img=Image.new('RGB',(w,h),(255,255,255))
    draw=ImageDraw.Draw(img)

    draw.line((0,h/2,10,h/2),fill=(255,0,0))

    # Draw the first node
    drawnode(draw,clust,10,(h/2),scaling,labels)
    img.save(jpeg,'JPEG')


def drawnode(draw,clust,x,y,scaling,labels):
    if clust.id<0:
        h1=getheight(clust.left)*20
        h2=getheight(clust.right)*20
        top=y-(h1+h2)/2
        bottom=y+(h1+h2)/2
        # Line length
        ll=clust.distance*scaling
        # Vertical line from this cluster to children
        draw.line((x,top+h1/2,x,bottom-h2/2),fill=(255,0,0))

        # Horizontal line to left item
        draw.line((x,top+h1/2,x+ll,top+h1/2),fill=(255,0,0))

        # Horizontal line to right item
        draw.line((x,bottom-h2/2,x+ll,bottom-h2/2),fill=(255,0,0))

        # Call the function to draw the left and right nodes
        drawnode(draw,clust.left,x+ll,top+h1/2,scaling,labels)
        drawnode(draw,clust.right,x+ll,bottom-h2/2,scaling,labels)
    else:
        # If this is an endpoint, draw the item label
        draw.text((x+5,y-7),labels[clust.id],(0,0,0))


# k-means clustering
def kcluster(rows,distance=euclidean,k=4):
    # Determine the minimum and maximum values for each point
    ranges=[(min([row[i] for row in rows]),max([row[i] for row in rows]))
    for i in range(len(rows[0]))]

    # Create k randomly placed centroids
    clusters=[[random.random()*(ranges[i][1]-ranges[i][0])+ranges[i][0]
    for i in range(len(rows[0]))] for j in range(k)]
  
    lastmatches=None
    bestmatches = None

    for t in range(100):
        bestmatches=[[] for i in range(k)]
    
        # Find which centroid is the closest for each row
        for j in range(len(rows)):
            row=rows[j]
            bestmatch=0
            for i in range(k):
                d=distance(clusters[i],row)
                if d<distance(clusters[bestmatch],row): bestmatch=i
            bestmatches[bestmatch].append(j)

        # If the results are the same as last time, this is complete
        if bestmatches==lastmatches: break
        lastmatches=bestmatches
    
        # Move the centroids to the average of their members
        for i in range(k):
            avgs=[0.0]*len(rows[0])
            if len(bestmatches[i])>0:
                for rowid in bestmatches[i]:
                    for m in range(len(rows[rowid])):
                        avgs[m]+=rows[rowid][m]
                for j in range(len(avgs)):
                    avgs[j]/=len(bestmatches[i])
                clusters[i]=avgs
      
    return bestmatches, clusters


def scaledown(data,distance=pearson,rate=0.01):
    n=len(data)

    # The real distances between every pair of items
    realdist=[[distance(data[i],data[j]) for j in range(n)]
             for i in range(0,n)]

    # Randomly initialize the starting points of the locations in 2D
    loc=[[random.random(),random.random()] for i in range(n)]
    fakedist=[[0.0 for j in range(n)] for i in range(n)]
  
    lasterror=None
    for m in range(0,1000):
        # Find projected distances
        for i in range(n):
            for j in range(n):
                fakedist[i][j]=sqrt(sum([pow(loc[i][x]-loc[j][x],2)
                                 for x in range(len(loc[i]))]))
  
        # Move points
        grad=[[0.0,0.0] for i in range(n)]
    
        totalerror=0
        for k in range(n):
            for j in range(n):
                if j==k: continue
                # The error is percent difference between the distances
                errorterm=(fakedist[j][k]-realdist[j][k])/realdist[j][k]
        
                # Each point needs to be moved away from or towards the other
                # point in proportion to how much error it has
                grad[k][0]+=((loc[k][0]-loc[j][0])/fakedist[j][k])*errorterm
                grad[k][1]+=((loc[k][1]-loc[j][1])/fakedist[j][k])*errorterm

                # Keep track of the total error
                totalerror+=abs(errorterm)
        print ("Total error:",totalerror)

        # If the answer got worse by moving the points, we are done
        if lasterror and lasterror<totalerror: break
        lasterror=totalerror
    
        # Move each of the points by the learning rate times the gradient
        for k in range(n):
            loc[k][0]-=rate*grad[k][0]
            loc[k][1]-=rate*grad[k][1]

    return loc


def draw2d(data,labels,jpeg='mds2d.jpg'):
    img=Image.new('RGB',(2000,2000),(255,255,255))
    draw=ImageDraw.Draw(img)
    for i in range(len(data)):
        x=(data[i][0]+0.5)*1000
        y=(data[i][1]+0.5)*1000
        draw.text((x,y),labels[i],(0,0,0))
    img.save(jpeg,'JPEG')

def sse(rows, distance, bestmatches, clusters):
    k = len(bestmatches)
    sum = 0
    for i in range(k):
        for rowid in bestmatches[i]:
            d = distance(clusters[i],rows[rowid])
            sum += d*d
    return sum

def bisectkcluster(rows,distance=euclidean,k=4):
    num_trial = 5
    result = []
    result_center = []

    for i in range(k-1):
        max_sse = -1
        separateid = -1
        for clusterid in range(len(result)):
            test_sse = sse(rows, distance, [result[clusterid]], [result_center[clusterid]])
            if test_sse>max_sse:
                max_sse = test_sse
                separateid = clusterid
        if separateid == -1:
            new_rows = rows
        else:
            new_rows = [rows[line] for line in result[separateid]]

        min_sse = sys.maxsize
        for time in range(num_trial):
            trial = kcluster(new_rows,distance,2)
            trial_sse = sse(new_rows, distance, trial[0], trial[1])
            if trial_sse < min_sse:
                best_bisection = trial
                min_sse = trial_sse
        if separateid != -1:
            result_buffer1 = [result[separateid][index] for index in best_bisection[0][0]]
            result_buffer2 = [result[separateid][index] for index in best_bisection[0][1]]
            result.pop(separateid)
            result_center.pop(separateid)
            result.append(result_buffer1)
            result.append(result_buffer2)
        else:
            result.append(best_bisection[0][0])
            result.append(best_bisection[0][1])
        result_center.append(best_bisection[1][0])
        result_center.append(best_bisection[1][1])
    return result, result_center

def best_distance_for_countries():
    data_result = data_preparation("dataset.csv")
    rownames1 = data_result[0]
    rownames2 = data_result[1]
    colnames = data_result[2]
    data = data_result[3]
    sse_list = []

    result = bisectkcluster(data, euclidean, 5)
    bestmatches = result[0]
    clusters = result[1]
    print("clusters by euclidean distance")
    sse_list.append(sse(data, euclidean, bestmatches, clusters))
    print("SSE =", sse_list[-1])
    for i in range(5):
        print("Cluster", i)
        print([rownames2[j] for j in bestmatches[i]])

    print()
    result = bisectkcluster(data, pearson, 5)
    bestmatches = result[0]
    clusters = result[1]
    print("clusters by pearson correlation")
    sse_list.append(sse(data, euclidean, bestmatches, clusters))
    print("SSE =", sse_list[-1])
    for i in range(5):
        print("Cluster", i)
        print([rownames2[j] for j in bestmatches[i]])

    print()
    result = bisectkcluster(data, cosine, 5)
    bestmatches = result[0]
    clusters = result[1]
    print("clusters by cosine distance")
    sse_list.append(sse(data, euclidean, bestmatches, clusters))
    print("SSE =", sse_list[-1])
    for i in range(5):
        print("Cluster", i)
        print([rownames2[j] for j in bestmatches[i]])

    print()
    result = bisectkcluster(data, manhattan, 5)
    bestmatches = result[0]
    clusters = result[1]
    print("clusters by manhattan distance")
    sse_list.append(sse(data, euclidean, bestmatches, clusters))
    print("SSE =", sse_list[-1])
    for i in range(5):
        print("Cluster", i)
        print([rownames2[j] for j in bestmatches[i]])
    print()

    print(sse_list)
    print()

def best_k_for_countries():
    data_result = data_preparation("dataset.csv")
    rownames1 = data_result[0]
    rownames2 = data_result[1]
    colnames = data_result[2]
    data = data_result[3]
    sse_list = []

    for k in range(2, 16):
        result = bisectkcluster(data, euclidean, k)
        bestmatches = result[0]
        clusters = result[1]
        print("separate into", k, "clusters")
        sse_list.append(sse(data, euclidean, bestmatches, clusters))
        print("SSE =", sse_list[-1])
        for i in range(k):
            print("Cluster", i)
            print([rownames2[j] for j in bestmatches[i]])
        print()
    print(sse_list)

def compare_bisecting():
    data_result = data_preparation("dataset.csv")
    rownames1 = data_result[0]
    rownames2 = data_result[1]
    colnames = data_result[2]
    data = data_result[3]
    sum_k = 0
    sum_bisect = 0

    for time in range(10):
        result = bisectkcluster(data, euclidean, 6)
        bestmatches = result[0]
        clusters = result[1]
        sum_bisect += sse(data, euclidean, bestmatches, clusters)

    for time in range(10):
        result = kcluster(data, euclidean, 6)
        bestmatches = result[0]
        clusters = result[1]
        sum_k += sse(data, euclidean, bestmatches, clusters)

    print("SSE by normal k-means algorithm:", sum_k/10)
    print("SSE by bisecting k-means algorithm:", sum_bisect/10)
    print()

def output_result_and_label():
    data_result = data_preparation("dataset.csv")
    rownames1 = data_result[0]
    rownames2 = data_result[1]
    colnames = data_result[2]
    data = data_result[3]
    result = bisectkcluster(data, euclidean, 6)

    file_object = open('result.csv', 'w', newline='')
    writeCSV = csv.writer(file_object, delimiter=',')
    for cluster in result[0]:
        writeCSV.writerow([rownames2[i] for i in cluster])

    file_object.close()

    descriptive_label(data, result[0])

def to_json():
    first_line = True
    file_object = open('./public_html/data.js', 'w', newline='')
    file_object.write("var data1 = [")
    with open("result.csv", encoding="utf-8") as csvfile:
        readCSV = csv.reader((line.replace('\0','') for line in csvfile), delimiter=',')
        i = 0
        for row in readCSV:
            cluster = list(row)
            print(cluster)
            for c in cluster:
                if first_line:
                    file_object.write(write_content(first_line, c, i))
                    first_line = False
                else:
                    file_object.write(write_content(first_line, c, i))
            i += 1
    file_object.write('];')
    file_object.close()
    csvfile.close()

def write_content(first_line, c, i):
    line = ""
    if not first_line:
        line += ', '
    if c == "Arab countries":
        countries = ["Bahrain", "Comoros", "Djibouti", "Kuwait", "Lebanon", "Libya", "Mauritania", "Oman", "Palestine", "Qatar", "Somalia", "Sudan", "Syria", "Tunisia", "United Arab Emirates", "Yemen"]
        for country in countries[:-1]:
            line += '{"Country": "' + country + '", "Cluster": ' + str(i) + '}, '
        line += '{"Country": "' + countries[-1] + '", "Cluster": ' + str(i) + '}'
    elif c == "Africa West":
        countries = ["Benin", "Cabo Verde", "Ivory Coast", "Gambia", "Guinea", "Guinea-Bissau", "Liberia", "Niger", "Senegal", "Sierra Leone", "Togo"]
        for country in countries[:-1]:
            line += '{"Country": "' + country + '", "Cluster": ' + str(i) + '}, '
        line += '{"Country": "' + countries[-1] + '", "Cluster": ' + str(i) + '}'
    elif c == "Africa East":
        countries = ["Eritrea", "Ethiopia", "South Sudan", "Madagascar", "Mauritius", "Seychelles", "Reunion", "Mayotte", "Burundi", "Kenya", "Malawi", "Mozambique"]
        for country in countries[:-1]:
            line += '{"Country": "' + country + '", "Cluster": ' + str(i) + '}, '
        line += '{"Country": "' + countries[-1] + '", "Cluster": ' + str(i) + '}'
    elif c == "Slovak Rep":
        line += '{"Country": "' + "Slovakia" + '", "Cluster": ' + str(i) + '}'
    elif c == "Czech Rep":
        line += '{"Country": "' + "Czech Republic" + '", "Cluster": ' + str(i) + '}'
    elif c == "Dominican Rep":
        line += '{"Country": "' + "Dominican Republic" + '", "Cluster": ' + str(i) + '}'
    elif c == "Kyrgyz Rep":
        line += '{"Country": "' + "Kyrgyzstan" + '", "Cluster": ' + str(i) + '}'
    elif c == "Macedonia Rep":
        line += '{"Country": "' + "Macedonia" + '", "Cluster": ' + str(i) + '}'
    elif c == "U.S.A.":
        line += '{"Country": "' + "United States" + '", "Cluster": ' + str(i) + '}'
    elif c == "Korea South":
        line += '{"Country": "' + "South Korea" + '", "Cluster": ' + str(i) + '}'
    else:
        line += '{"Country": "' + c + '", "Cluster": ' + str(i) + '}'
    return line

import numpy as np
import seaborn as sns
def heat_map(rows, clusters, output_name):
    sns.set()

    difference = np.empty([len(rows), len(rows)], float)

    row = 0
    col = 0
    for cluster_i in clusters:
        for i in cluster_i:
            for cluster_j in clusters:
                for j in cluster_j:
                    difference[row, col] = euclidean(rows[i], rows[j])
                    col += 1
            col = 0
            row += 1

    i = 0
    axis = []
    for cluster in clusters:
        axis += ['']*len(cluster)
        axis.append("C"+str(i))
        i += 1
    ax = sns.heatmap(difference, xticklabels=axis, yticklabels=axis)
    ax.get_figure().savefig(output_name)

hierarchical = []
def cut_hierarchical(top_cluster, depth):
    global id_list

    if depth == 0:
        id_list = []
        find_all(top_cluster)
        hierarchical.append(id_list.copy())
    else:
        if top_cluster.left != None:
            cut_hierarchical(top_cluster.left, depth-1)
        if top_cluster.right != None:
            cut_hierarchical(top_cluster.right, depth-1)


def create_cloud (oname, words, maxsize=60, fontname='Lobster'):
    '''Creates a word cloud (when pytagcloud is installed)
    Parameters
    ----------
    oname : output filename
    words : list of (value,str)
    maxsize : int, optional
        Size of maximum word. The best setting for this parameter will often
        require some manual tuning for each input.
    fontname : str, optional
        Font to use.
    '''

    # gensim returns a weight between 0 and 1 for each word, while pytagcloud
    # expects an integer word count. So, we multiply by a large number and
    # round. For a visualization this is an adequate approximation.

    #words = [(w,int(v*1000)) for w,v in words]
    tags = make_tags(words, maxsize=maxsize)
    create_tag_image(tags, oname, size=(2000, 2000), fontname=fontname)

def descriptive_label(rows, clusters):
    words = [None]*6
    count = -1
    with open('dimensions_keywords.csv') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in csvreader:
            if count == -1:
                count += 1
                continue
            list(row)
            words[count] = [row[1].split(), row[2].split()]
            count += 1

    count = 0
    for cluster in clusters:
        centroid = [0]*6
        num = 0
        word_dic = []
        for i in cluster:
            for j in range(6):
                centroid[j] += rows[i][j]
            num += 1
        for i in range(6):
            centroid[i] = int(centroid[i]/num)
            print(centroid[i],end = ' ')
            for w in words[i][0]:
                word_dic.append((w, centroid[i]))
            for w in words[i][1]:
                word_dic.append((w, 100-centroid[i]))
        print()
        create_cloud("word_cloud_cluster_"+str(count)+".png", word_dic)
        count += 1
'''
data_result = data_preparation("dataset.csv")
rownames1 = data_result[0]
rownames2 = data_result[1]
colnames = data_result[2]
data = data_result[3]
result = bisectkcluster(data, euclidean, 6)
descriptive_label(data, result[0])

clust= hcluster(data, max_dis, euclidean)
cut_hierarchical(clust, 3)
heat_map(data, hierarchical, "hierarchical_heat_map.png")
'''
#best_distance_for_countries()
#best_k_for_countries()
#compare_bisecting()

#output_result()
#to_json()

#best_inter_distance_for_countries()
