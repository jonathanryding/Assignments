#visualise GB
#Sets data from a csv file into arrays and plots towns and cities onto an image of the UK using longitude and latitude for a rough visualisation

print("Welcome to my data displayer, this code will visualise the GBplaces.csv file onto a map of the UK for you!")

import matplotlib.pyplot as plt
import numpy as np


places = [] #place
types = [] #type
pop = [] #population
lat = [] #latitude
long = [] #longitude

opened = 0

#create a variable to set if file opens successfully, prevents code cutting out

try:
    
#try to open the file, change a variable if opens 
    
    readFile = open('GBplaces.csv','r')
    #open file GBplaces.csv, read
    
    opened = 1
    
   #file opened
   
    print("GBplaces.csv opened successfully.")
    #tell user file was opened successfully

except:
    
    print("Error, could not open file.")
    
    #file did not open, print a failure message

    
readLine = 0

 #initially set readLine variable to false so the first line with the headings in the file is skipped.
    
if opened is 1:
    #first line ignored
    
    for line in readFile:
        #for loop processes each row of the csv file
        
        if readLine:
          
            splitLine = line.split(',')
            
            #splits up line through commas, then add the different data types to different arrays.

            places.append(splitLine[0])
            
            types.append(splitLine[1])
            
            pop.append(int(splitLine[2]))
            
            lat.append(float(splitLine[3]))
            
            long.append(float(splitLine[4]))
            
            
         
    #data gets stored into its own array accordingly using array.append() and the correct part of the split line
            
        else:
            
            readLine = 1
            
            #change the readLine variable to true after the first line is skipped
            
            
    readFile.close()
    
    print("GBplaces.csv closed.")
    
    #close file, tell user
    


placesPlot = np.array(places)
typesPlot = np.array(types)

latPlot = np.array(lat)
longPlot = np.array(long)
popPlot= np.array(pop)

#create new arrays using numpy so latitude and longitude can be plotted
    
        
#search for London in the array so it can be marked as capital on the map        
#split up array into cities, towns
        
towns = [i for i, x in enumerate(typesPlot) if x == "Town"]

cities = [j for j, y in enumerate(typesPlot) if y == "City"]

capital = [k for k, z in enumerate(placesPlot) if z == "London"]

#searches for type and counts each one a number so they can be plotted differently


reply = input("Would you like a certain town or city to be highlighted?")
#asks user if they would like a specific town to be highlighted so it can be seen more clearly on the map

while reply.lower() != "yes" and reply.lower() != "no":
    reply = input("Please respond yes or no.")
    #demands a yes or no
    
    
# if they chose yes, asks for place, searches, if found will put a pink cross on the point so it can be quickly identified.
if reply.lower() == "yes":
    
    place = input("Which town or city would you like to be highlighted? (E.g. Aberdeen)")
    
    if place == "London":
        
        print("London is already highlighted.")
    
    print("If " + place + " is in the csv file, its marker will be highlighted by a pink cross.")
    
         
    foundPlace = [i for i, x in enumerate(placesPlot) if x == place]
    #slightly different search code because the place should already be in the file so no need to rewrite and search again
    
    plt.scatter(longPlot[foundPlace], latPlot[foundPlace], c = "pink", zorder = 4, label = place, alpha = 0.9, marker = "x")
    #plots



PointSize = 0.00008*(popPlot)
#adjusting area of points according to population size

plt.scatter(longPlot[towns], latPlot[towns], c = "black", s = PointSize[towns], marker = "x", zorder = 3, label = "Town", alpha = 0.7)
plt.scatter(longPlot[cities], latPlot[cities], c = "blue", s = PointSize[cities], edgecolors = "white", zorder = 1, label = "City", alpha = 0.6)
plt.scatter(longPlot[capital], latPlot[capital], c = "white", s = PointSize[capital], edgecolors = "black", zorder = 2, label = "Capital, London")

#Points are plotted separately for each type, 
#crosses for towns, circles for cities, alpha is used to vary transparency of points. Zorder selects which layer of the image the plotted points will be on.
 
        
print("Displaying map")
    

mapEmpty = plt.imread("UK.png")

plt.imshow(mapEmpty, aspect = 1.46, zorder = 0, extent = [-11.55, 2.77, 49.44, 61.4])

#aspect ratio of empty map maintained, used as a background through "zorder == 0" 
#lattitude and longititude of map aligned using  real coords of map edges from http://boundingbox.klokantech.com/
#towns marked with crosses, rest with circles


plt.xlabel("Longitude (°)")
plt.ylabel("Latitude (°)")
plt.title("UK map")
plt.legend(loc = 4)

#labels and attaches a legend to graph

plt.show()
#final step to show graph to user

print("Thank you for using my data displayer!")
    
    