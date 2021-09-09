import random
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter import filedialog
import csv
import pandas as pd


#You dont need to understand this, just know that it takes a list of numbers, and a desired value. It will return the closest value in the list to K
def closest(lst, K):
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]

#open a dialog box and get the selected csv file
root = Tk()
root.withdraw()
csvfile = filedialog.askopenfile(parent=root,mode='r',filetypes=[('Excel file','*.csv')],title='Choose CSV file')

# open another dialog box and get the selected tsv file
tsvfile = filedialog.askopenfile(parent=root,mode='r',filetypes=[('Excel file','*.tsv')],title='Choose TSV file')

# What kind of events do you want? These should be the EXACT text of what is in the tsv column
eventtype = 'Event/Description/Driver begins braking'
eventtimes = [] # create a blank list of event times
f=256 # ABM sample frequency

# open the selected tsv file
with open(tsvfile.name) as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"') # read the file, line by line, into seperate rows
    for row in rd: # for each row in the tsv file
        data = row[2].split(',') # this 'splits' our row string by commas and creates a list of elements
        if eventtype in data[1]: # this checks to see if our event type string (eventtype above) is inside of current tsv file event
            eventtimes.append(float(row[1])) # if its the kind of event we want, lets add it to our eventtimes list

df = pd.read_csv(csvfile.name)
times = df.Time.tolist()
print('The Total Number of Events = ' + str(len(eventtimes)))
#print (eventtimes)

for eventtime in eventtimes: # go through each event start time, one at a time
    EventStartTime = eventtime * 1000  # Convert to Milliseconds
    EventEndTime = (eventtime + .650) * 1000  # Add .650 seconds for p300 then convert to Milliseconds 

    # with our new times, lets find the closest times in our csv file
    StartIndex = round(EventStartTime/(1000/f)) 
    EndIndex =   round(EventEndTime/(1000/f))
    
    #print('The Start Index for the braking event is ' + str(StartIndex))
    #print('The End Index for the braking event is ' + str(EndIndex))

    # so now that we have our start and ending index within the data for our event, we can do whatever analysis we wanna do around that chunk of time

    # do stuff with number

    # do even more stuff with numbers

    # take a break

    # The new Cheez-it grooves are pretty good have you tried them?

    # Annnnnnd done with whatever we wanna do here.

    # for now, lets just make up some random numbers that we want to save to the csv file.. I will just create 2 random lists of numbers for ease
    randomlist1 = []
    randomlist2 = []
    for i in range(5): # creating 5 samples
        x = random.randint(1, 30) # generate a random number between 1 and 30
        randomlist1.append(x) # add that number to a list
        y = random.randint(1, 30)  # generate a second random number
        randomlist2.append(y) # add that number to our second list

    # so now we have some data we want to save to csv
    columnHeaders = ['RandomNumbers1', 'RandomNumbers2'] # create some generic column headers to write
    with open('cheez-its.csv', 'w', newline='') as file: # open a new csv file to save the information into, feel free to add a path to a file, just make sure the path exsists, but the file itself does NOT
        writer = csv.writer(file)
        writer.writerow(columnHeaders)
        for index in range(len(randomlist1)):
            writer.writerow([randomlist1[index], randomlist2[index]]) # add more columns by just add more " + ',' "





