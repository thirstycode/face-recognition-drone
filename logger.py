import time
from datetime import datetime as dt

subjects=[]
full_count = None

def start(subjects_):
    global subjects
    global full_count
    subjects = subjects_
    list1 = []
    for name in subjects:
        list1.append((name,0))
    full_count = dict(list1)

def increment(subj_name,status):
    global full_count
    full_count[subj_name] += 1

def end1():
    global full_count

    t= dt.now()
    out_name = t.strftime('%I-%M-%p-%d-%m')
    file_name = out_name + ".txt"
    with open("Reports/" + file_name,'w+') as file:
        file.write("============  This Is Log Report Of Drone ============\n")
        totalnames = list(full_count.keys())
        for name in totalnames[1:]:
            file.write("We Found Person Named : " + name + " In " + str(full_count[name]) + " Frames.\n")
