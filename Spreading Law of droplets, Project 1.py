import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#Spreading Law of droplets, Project 1
#Determine the spreading law (relationship between the speed of the contact line and the contact angle) of picolitre droplets using the spherical cap approximation.


#function definitions for manipulating data sets given

def radiusSpeed(radius, time):
#calculates the discrete derivative of radii list input.

    v = []
    #empty array created in order to save values step by step
    R = radius
    t = time
    
    for i in range ( 0 , len(radius) - 1 ):
        dR = R[i +1] - R[i]
        dt = t[i+1] - t[i]
        v_step = dR / dt
        v.append(v_step)
        
#forward divided difference scheme, velocities are approximated to be the difference of the radii at the point and one time step ahead, divided by the time step.
#alt. schemes could be a backward difference divide scheme or a central divided scheme.
        
    return v


def meanOf2Array(array1, array2):
#calculates the mean between elements of the same index across 2 arrays (lists).
    mean = []
    
    for i in range(0, len(array1)):
        
        mean_step = np.mean( (array1[i], array2[i]) )
        #np.mean calculates average of elements written.
        mean.append(mean_step)
        #adds to initially empty array continually until all are computed
    
    return mean

def meanOf3Array(array1, array2, array3):
#calculates the mean between elements of the same index across 3 arrays (lists).
    mean = []
    
    for i in range(0, len(array1)):
        
        mean_step = np.mean( (array1[i], array2[i], array3[i]) )
        mean.append(mean_step)
    
    return mean


def stdevOf2Array(array1, array2):

    
    stdev = []
    
    for i in range(0, len(array1)):
        
        stdev_step = np.std( (array1[i], array2[i]))
        #np.std in a similar way to np.mean
        stdev.append(stdev_step)
    
    return stdev




def stdevOf3Array(array1, array2, array3):
#calculates the standard deviation of elements that have the same index in 3 arrays (lists) through the same method as stdevOf2Array(array1, array2) 
    stdev = []
    
    for i in range(0, len(array1)):
        
        stdev_step = np.std( (array1[i], array2[i], array3[i]) )
        stdev.append(stdev_step)
    
    return stdev

    
def propagatedErrorOnV(ErrorOnRadius, timestep):
    #Defining a function to use for error propagation
    
    DV = []
    DR = ErrorOnRadius
    

    for i in range ( 0 , len(DR) - 1 ):
        #length of radii - 1 as a point was used in calculation
        DV_step =(pow(( pow( (DR[i +1]),2) + pow((DR[i]),2) ), 0.5)/ timestep)
        #uncertainties on the radii in front and at the point are added in quadrature, due to how the velocity was defined as the forward discrete derivative.
        #alternatively, could create a polynomial fit to the time-evolution of the radius graph, then differentiate that but that would include an error from the fit.
        
      
        DV.append(DV_step)
    
    return DV
    

def heightArray(radius):
#calculate the droplet (height at the centre) given the radius using the spherical cap approximation and output as an array.
#
    height = []
    
    for i in range(0, (len(radius) - 1) ):
        
        Hpoly = [1, 0 , 3 * pow(radius[i] , 2), -(6 * vol / np.pi)] 
        #polynomial of h from a spherical top of a sphere.
        
        a= np.roots(Hpoly)
        #np.roots returns the roots of a polynomial of order n that is given in a rank-1 array of length n + 1, where the array elements are the coefficients of decreasing order.
        #e.g. inputting [a, b, c] corresponds to polynomial ax^2 + bx + c  = 0
        #https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.roots.html
        
        height.append(np.real(a[2]))
        #roots where h is non-negative and real are selected. Negative and imaginary heights have no physical significance.
        #Units of h are micrometres (radii is micrometres and volume micrometres cubed)
    
    return height

def thetaArray(radius, height):
#calculates the contact angle using heights calculated by heightArray(radius) of the droplet to the surface and organises in an array.
#expect contacts angles to be less than pi/2 (hydrophilic)
    
    theta = []
    
    for i in range(0, (len(radius) - 1) ):
        
        #in range 0 to number of data points - 1 as there is one les data point in the velocity lists than the radii lists.
        #This makes the velocity and theta data sets the same size so they can be plotted to observe the spreading law.
        #I have chosen to remove the last velocity as there are more results in that region of the contact angle and I would expect the drop to become stationary after a long period of time.
        
        theta_step = np.pi / 2 - np.arctan( (pow(radius[i],2) - pow(height[i], 2)) / (2 * radius[i] * height[i]) )
        #from geometry of a spherical cap, where theta is angle from the base of the cap to the tangent of the sphere.
        
        theta.append(theta_step)
        
    return theta

def reducedChiSquared(Ydata, Yfit, Yerr, Nparameters):
#calculates the reduced chi-squared of a general fit to data step-by-step by summing the square of each residual and dividing by the number of degrees of freedom of the fit.
#Yfit is the array of velocities from the fitted curve at the same angle as Ydata.
#chi-squared is an indicator of how appropriate the fit is to the data. 
    
    chiold = 0 #initial chi-squared value begins at 0.
    
    for i in range(0, len(Ydata)):
        #calculating the  weighted sum of the residuals squared for each data point
        
        chi_step = pow( ((Ydata[i] - Yfit[i])/Yerr[i]) , 2 )
        chiSquared = chi_step + chiold
        chiold = chiSquared
        
    reduChiSquare = (chiSquared / (len(Ydata) - Nparameters) )
    #To calculate the reduced chi-squared, divide the chi-squared by the number of degrees of freedom. This is equal to the number of data points used minus the number of parameteres of the fit.
    
    return reduChiSquare


def interpolate(X , Xspacing, Ydata):

    Finterpolated = interp1d(X, Ydata, kind = "linear", fill_value = "extrapolate")
    #returns a function that uses interpolation to find the value of new points
    #smooths data sets that shows small isolated errors
    #spline interpolation approximates the function by a mean of series of polynomials (series over adjacedent intervals, each has a continuous derivative)
     #'kind:' specifies the order of spline polynomials used, e.g. a linear spline is a continiuous function formed by connecting linear segments, a cubic spline connects cubic segments
    
    Y = Finterpolated(Xspacing)
    #evenly spaced, interpolated data points
    
    return(Y)

opened = 0
#reading in data for two droplets spreading on two different surfaces
try:
    
    data11 = np.loadtxt("Top_view_drop_1_data_run1.txt")
    data12 = np.loadtxt("Top_view_drop_1_data_run2.txt")
    data13 = np.loadtxt("Top_view_drop_1_data_run3.txt")
    data21 = np.loadtxt("Top_view_drop_2_data_run1.txt")
    data22 = np.loadtxt("Top_view_drop_2_data_run2.txt")
    
    t1 = data11[:,0]  #time since spreading started, measured in seconds for the first droplet
     
    R1_1 = data11[:,1] #instantaneous radii in micrometres. #run 1, drop 1
    R2_1 = data12[:,1] #run 2
    R3_1 = data13[:,1] #run3
    
    t2 = data21[:,0]  #second drop 2 data, time in seconds.
       
    R1_2 = data21[:,1] # radii #run 1, drop 2
    R2_2 = data22[:,1] #run 2
    
    
    opened = 1
    
    print("Files loaded in")
    
except:
    print("Files could not be opened")
    
    
#volume of drops
vol = 7600  #7.6 picolitres, 7600 micrometres cubed
#The volume of the droplet was the same for all data sets.
    
if opened == 1:
    #opened successfully, begin data manipulation and analysis
   
    #drop 1, using functions to maniupulate data given
    
    meanR_1 = meanOf3Array(R1_1, R2_1, R3_1) #Array (list) containing the means of all drop 1 data sets where the first index is at t = 0s *All radii measured in micrometres.
    stdErrR_1 = stdevOf3Array(R1_1, R2_1, R3_1) #Standard deviation of all data from drop 1
    
    V1_1 = radiusSpeed(R1_1, t1) #Contact speed of the first data set, for drop 1
    V2_1= radiusSpeed(R2_1, t1)  #Contact speed of the second data set, for drop 1
    V3_1 = radiusSpeed(R3_1, t1) #Contact speed of the third data set, for drop 1
    
    meanV_1 = meanOf3Array(V1_1, V2_1, V3_1)  #Array containing mean contact speeds of drop 1
    stdErrV_1 = stdevOf3Array(V1_1, V2_1, V3_1) #Data spreads (standard deviation) of the contact speed of drop 1
    
    H1_1 = heightArray(R1_1) #The spherical cap model heights of the data set 1, from drop 1 
    H2_1 = heightArray(R2_1) #The spherical cap model heights of the data set 2, from drop 1
    H3_1 = heightArray(R3_1) #The spherical cap model heights of the data set 3, from drop 1
    
    meanH_1 = meanOf3Array(H1_1, H2_1, H3_1) #Mean drop heights of drop 1
    stdErrH_1 = stdevOf3Array(H1_1, H2_1, H3_1) #Data spreads of drop 1
    
    theta1_1 = thetaArray(R1_1, H1_1) #Contact angles calculated from heights for data set 1 from drop 1
    theta2_1 = thetaArray(R2_1, H2_1) #Contact angles for data set 2 from drop 1
    theta3_1 = thetaArray(R3_1, H3_1) #Contact angles for data set 3 from drop 1
    
    meanTheta_1 = meanOf3Array(theta1_1, theta2_1, theta3_1) #Mean contact angles for drop 1
    stdErrTheta_1 = stdevOf3Array(theta1_1, theta2_1, theta3_1) #Standard deviations for drop 1
    
    #drop 2

    meanR_2 = meanOf2Array(R1_2, R2_2) #Mean radii for drop 2
    stdErrR_2 = stdevOf2Array(R1_2, R2_2) #Standard deviations from the mean radii, drop 2
    
    V1_2 = radiusSpeed(R1_2, t2) #Contact speed of the first data set of drop 2
    V2_2= radiusSpeed(R2_2, t2) #Contact speed of the second data set of drop 2
    
    meanV_2 = meanOf2Array(V1_2, V2_2) #Mean contact speeds of drop 2
    stdErrV_2 = stdevOf2Array(V1_2, V2_2) #Standard deviations of the contact speed of drop 2
    
    H1_2 = heightArray(R1_2) #The spherical cap model height of the data set 1, from drop 2
    H2_2 = heightArray(R2_2) #The spherical cap model height of the data set 2, from drop 2
    
    meanH_2 = meanOf2Array(H1_2, H2_2) #Mean drop heights of drop 2
    stdErrH_2 = stdevOf2Array(H1_2, H2_2) #Standard deviations of drop 2
    
    theta1_2 = thetaArray(R1_2, H1_2) #Contact angles for data set 1 from drop 2
    theta2_2 = thetaArray(R2_2, H2_2) #Contact angles for data set 2 from drop 2
    
    meanTheta_2 = meanOf2Array(theta1_2, theta2_2) #Mean contact angles from drop 2
    stdErrTheta_2 = stdevOf2Array(theta1_2, theta2_2) #Standard deviations from drop 2
    
     #Errors on contact speeds and angles can be propagated from the initial standard deviation of the data sets given.
     #This will be done in order to compare the errors and the more appropriate errors will be chosen.
     
    #Propagating errors to velocity:
    
    errV_1 = propagatedErrorOnV(stdErrR_1, (t1[1]-t1[0]) )
    errV_2 = propagatedErrorOnV(stdErrR_2, (t2[1]-t2[0]) )
    #Assuming error on time is negligible, use function as defined earlier
  
    #Propagating errors to drop height:
    
    errH_1 = [] #empty height error arrays to append errors to as they are calculated 
    errH_2 = []
    
    for i in range(0, len(stdErrR_1)-1):
        
        errH1_step = (3*stdErrR_1[i])/(meanH_1[i] + (2* vol)/(np.pi * pow(meanH_1[i], 2))) 
        errH1_step = stdErrR_1[i] * pow( ((2 * vol /np.pi *meanH_1[i]) - (pow(meanH_1[i], 2)/3)),0.5) /((vol / (np.pi*pow(meanH_1[i], 2))) + (meanH_1[i]/3) )
        
        #errors estimated by rewriting  Hpoly = [1, 0 , 3 * pow(radius[i] , 2), -(6 * vol / np.pi)] so R is a function of h, f(h).
        # R(h) = sqrt(( 2*volume / pi* h) - (h^2 / 3) ) 
        #Then error on h = error on r / |partial d/dh(f(h))| evaluated at each point
        #partial d/dh(f(h)) is -( vol / (pi * h^2) + h / 3) * ( 2 * vol / pi*h - h^2 /3)^-0.5
    
        errH_1.append(errH1_step)
        
        errH2_step = stdErrR_2[i] * pow( ((2 * vol /np.pi *meanH_2[i]) - (pow(meanH_2[i], 2)/3)),0.5) /((vol / (np.pi*pow(meanH_2[i], 2))) + (meanH_2[i]/3) )
        errH_2.append(errH2_step)
  
    errTheta_1 = [] #empty theta error arrays to append errors to as they are calculated 
    errTheta_2 = []
    
    for i in range(0, len(meanH_1)):
        errTheta_1step =( ( 9 * vol ) / ( pow((((2*vol )/ (np.pi * meanH_1[i])) - (pow(meanH_1[i], 2))/ 3 ), 0.5)*(3*vol + np.pi * meanH_1[i])) )* errH_1[i]
        errTheta_2step =( ( 9 * vol ) / ( pow((((2*vol )/ (np.pi * meanH_2[i])) - (pow(meanH_2[i], 2))/ 3 ), 0.5)*(3*vol + np.pi * meanH_2[i])) )* errH_2[i]
        errTheta_1.append(errTheta_1step)
        errTheta_2.append(errTheta_2step)
        
        
#Equation for contact angle is rewritten so it only a function of h. Then the error on theta = error on h * |partial d/dh(f(h))|
#The errors of R and h cannot be adde in quadrature as it is assumed that variables are independent.
        
#simpler partial d/dh(f(h)) form confirmed by using www.wolframalpha.com:
#https://www.wolframalpha.com/input/?i=%E2%88%82%2F%E2%88%82x(+-arctan(+++++++++(V%2F(%CF%80x)+-+2(x%5E2)%2F3)+%2F+(x%E2%88%9A(2V%2F%CF%80x+-+x%5E2+%2F3+)++++)+)
    
   
   
    #Plot the time-evolution of the average radius of the spreading droplets
    print("Plotting the time-evolution of the average radius of droplets 1 & 2")
   
    plt.plot(t1, meanR_1)
    plt.title("Time-evolution of the radius (Drop 1)")
    plt.xlabel("Time, s")
    plt.ylabel("Radius, μm/s")
    plt.errorbar(t1, meanR_1, yerr = stdErrR_1, ecolor = "r")
    plt.show()
   
    #Plot the time-evolution of the average radius of the spreading droplet
   
    plt.plot(t2, meanR_2)
    plt.title("Time-evolution of the radius (Drop 2)")
    plt.xlabel("Time, s")
    plt.ylabel("Radius, μm/s")
    plt.errorbar(t2, meanR_2, yerr = stdErrR_2, ecolor = "r")
    plt.show()
    
    #Plots showing errors and then choosing an error method based on the results overall.
    print("Displaying results from two error evaluation methods")
    
    plt.plot(meanTheta_1, meanV_1)
    plt.title("Errors from using spread of the contact speed and angle (Drop 1)")
    plt.xlabel("Contact angle, radians")
    plt.ylabel("Contact speed, μm/s")
    plt.errorbar(meanTheta_1, meanV_1, xerr = stdErrTheta_1, yerr = stdErrV_1, ecolor = "r")
    plt.show()
    
    plt.plot(meanTheta_1, meanV_1)
    plt.title("Errors from propagation (Drop 1)")
    plt.xlabel("Contact angle, radians")
    plt.ylabel("Contact speed, μm/s") 
    plt.errorbar(meanTheta_1, meanV_1, xerr = errTheta_1, yerr = errV_1, ecolor = "g")
    plt.show()
        
    #spreading laws with different errors methods for drop 2
    
    plt.plot(meanTheta_2, meanV_2)
    plt.title("Errors from using spread of the contact speed and angle (Drop 2)")
    plt.xlabel("Contact angle, radians")
    plt.ylabel("Contact speed, μm/s")
    plt.errorbar(meanTheta_2, meanV_2, xerr = stdErrTheta_2, yerr = stdErrV_2, ecolor = "r")
    plt.show()
    
    plt.plot(meanTheta_2, meanV_2)
    plt.title("Errors from propagation (Drop 2)")
    plt.xlabel("Contact angle, radians")
    plt.ylabel("Contact speed, μm/s")
    plt.errorbar(meanTheta_2, meanV_2, xerr = errTheta_2, yerr = errV_2, ecolor = "g")
    plt.show()
    
  #drop1, Propagated errors on contact speed appear reasonable, with the exception of the point near 0.034 radians.
  #drop1, Propagated errors on contact angle are generally the same order as the data spread errors, with some appearing too small to be true.
  #Data spread is good for large amounts of data
  #drop1, Propagated errors on the contact angle are much larger than the data spread.
  #Will be using the errors from the data spread for fitting quadratics, cubic, De Gennes law and Cox-Voinox law to the spreading law data.
    
     #quadratic fits, drop 1
    print("Displaying results from fits to Drop 1 data using errors from data spread.")
    
    (quadcoef1, quadcovr1) = np.polyfit(meanTheta_1, meanV_1, 2 , cov = True) 
    quadcovr1 = np.array(quadcovr1).ravel()
    
    quadfitline1 = []
    
    for i in range(0, len(meanTheta_1)):
        polyfitline =  quadcoef1[0]*np.power(meanTheta_1[i], 2) + quadcoef1[1]*meanTheta_1[i] + quadcoef1[2]
        quadfitline1.append(polyfitline)  
    
    errquadcoef1_1 = np.sqrt(quadcovr1[0])
    errquadcoef2_1 = np.sqrt(quadcovr1[4])
    errquadcoef3_1 = np.sqrt(quadcovr1[8])
    
    plt.plot(meanTheta_1, meanV_1)
    plt.plot(meanTheta_1, quadfitline1)
    plt.title("Spreading law for Drop 1. Quadratic fit.")
    plt.xlabel("Contact angle, radians")
    plt.ylabel("Contact speed, μm/s")
    plt.errorbar(meanTheta_1, meanV_1, xerr = stdErrTheta_1, yerr = stdErrV_1, ecolor = "r")
    plt.show()
    
    quadredChiSqu_1 = reducedChiSquared(meanV_1, quadfitline1, stdErrV_1, 3) #calculate reduced chisquared of fit
    print("The best fitted quadratic is: U(θ) = (%5.2f ± %5.2f) θ^2 + (%5.2f ± %5.2f) θ + (%5.2f ± %5.2f) " %(quadcoef1[0], errquadcoef1_1, quadcoef1[1], errquadcoef2_1, quadcoef1[2], errquadcoef3_1))
    print("This has reduced chi-squared: %5.2f" %quadredChiSqu_1)
    
   
   #cubic
   
    (cubicoef1, cubicovr1) = np.polyfit(meanTheta_1, meanV_1, 3 , cov = True) 
    cubicovr1 = np.array(cubicovr1).ravel()
   
    cubicfitline1 = []
    
    for i in range(0, len(meanTheta_1)):
        polyfitline =  cubicoef1[0]*np.power(meanTheta_1[i], 3) + cubicoef1[1]*np.power(meanTheta_1[i],2) + cubicoef1[2]*meanTheta_1[i] + cubicoef1[3]
        cubicfitline1.append(polyfitline)
        
    
    errcubicoef1_1 = np.sqrt(cubicovr1[0])
    errcubicoef2_1 = np.sqrt(cubicovr1[5])
    errcubicoef3_1 = np.sqrt(cubicovr1[10])  
    errcubicoef4_1 = np.sqrt(cubicovr1[15])
    
    plt.plot(meanTheta_1, meanV_1)
    plt.plot(meanTheta_1, cubicfitline1)
    plt.title("Spreading law for Drop 1. Cubic fit.")
    plt.xlabel("Contact angle, radians")
    plt.ylabel("Contact speed, μm/s")
    plt.errorbar(meanTheta_1, meanV_1, xerr = stdErrTheta_1, yerr = stdErrV_1, ecolor = "r")
    plt.show()
  
    CUBICredChiSqu_1 = reducedChiSquared(meanV_1, cubicfitline1, stdErrV_1, 4)
    
    print("The best fitted cubic is: U(θ) = (%5.2f ± %5.2f) θ^3 + (%5.2f ± %5.2f) θ^2 + (%5.2f ± %5.2f) θ + (%5.2f ± %5.2f) " %(cubicoef1[0], errcubicoef1_1, cubicoef1[1], errcubicoef2_1, cubicoef1[2], errcubicoef3_1, cubicoef1[3], errcubicoef4_1))
    print("This fit has reduced chi-squared: %5.2f" %CUBICredChiSqu_1)
    
     #De Gennes law
     
    (DeGcoef1, DeGcovr1) = np.polyfit( (np.power(meanTheta_1, 2)), meanV_1, 1 , cov = True) 
    DeGcovr1 = np.array(DeGcovr1).ravel()
    
    DeGfitline1 = []
    
    for i in range(0, len(meanTheta_1)):
        fitline =  DeGcoef1[0]*pow(meanTheta_1[i], 2) + DeGcoef1[1]
        DeGfitline1.append(fitline)
    
    errDeGcoef1_1 = np.sqrt(DeGcovr1[0])
    errDeGcoef2_1 = np.sqrt(DeGcovr1[3])
    
    plt.plot(meanTheta_1, meanV_1)
    plt.plot(meanTheta_1, DeGfitline1)
    plt.title("Spreading law for Drop 1. De Gennes fit.")
    plt.xlabel("Contact angle, radians")
    plt.ylabel("Contact speed, μm/s")
    plt.errorbar(meanTheta_1, meanV_1, xerr = stdErrTheta_1, yerr = stdErrV_1, ecolor = "r")
    plt.show()
    
    DeGredChiSqu_1 = reducedChiSquared(meanV_1, DeGfitline1, stdErrV_1, 2)  
    
    print("The best De Gennes fit is: U(θ) = (%5.2f ± %5.2f) θ^2 + (%5.2f ± %5.2f) " %(DeGcoef1[0], errDeGcoef1_1, DeGcoef1[1], errDeGcoef2_1 ))
    print("This fit has reduced chi-squared: %5.2f" %DeGredChiSqu_1)
    
     # Cox- Voinox law
    
    (coxcoef1, coxcovr1) = np.polyfit( (np.power(meanTheta_1, 3)), meanV_1, 1 , cov = True) 
    coxcovr1 = np.array(coxcovr1).ravel()
 
    COXfitline1 = []
    
    for i in range(0, len(meanTheta_1)):
        fitline =  coxcoef1[0]*pow(meanTheta_1[i], 3) + coxcoef1[1]
        COXfitline1.append(fitline)
        
    
    errcoxcoef1_1 = np.sqrt(coxcovr1[0])
    errcoxcoef2_1 = np.sqrt(coxcovr1[3])
    
    plt.plot(meanTheta_1, meanV_1)
    plt.plot(meanTheta_1, COXfitline1)
    plt.title("Spreading law for Drop 1 Cox-Voinox fit.")
    plt.xlabel("Contact angle, radians")
    plt.ylabel("Contact speed, μm/s")
    plt.errorbar(meanTheta_1, meanV_1, xerr = stdErrTheta_1, yerr = stdErrV_1, ecolor = "r")
    plt.show()
    
    COXredChiSqu_1 = reducedChiSquared(meanV_1, COXfitline1, stdErrV_1, 2)
    
    
    print("The best Cox-Voinox fit is: U(θ) = (%5.2f ± %5.2f) θ^3 + (%5.2f ± %5.2f) " %(coxcoef1[0], errcoxcoef1_1, coxcoef1[1], errcoxcoef2_1 ))
    print("This fit has reduced chi-squared: %5.2f" %COXredChiSqu_1)
   

    #Interpolate contact angle and contact speed time-evolutions for less jagged data and more uniform data. 
      #smoothing by spline interpolation on data sets for contact speed and angle for all 3 sets, and approximate errors on both by stdev
   
    X1 = np.linspace(meanTheta_1[0], meanTheta_1[-1], len(meanV_1)) # evenly spaced array of points where I want to interpolate my data to

    interpolatedV1_1 = interpolate(theta1_1, X1, V1_1)
    interpolatedV2_1 = interpolate(theta2_1, X1, V2_1)
    interpolatedV3_1 = interpolate(theta3_1, X1, V3_1)
    
    #interpolating evenly spread points from all sets of data then taking the mean and stdev for an estimate to the uncertainty on velocity
    
    meanIntV_1 = meanOf3Array(interpolatedV1_1, interpolatedV2_1, interpolatedV3_1) #taking the mean of all the interpolation results
    stdErrIntV_1 = stdevOf3Array(interpolatedV1_1, interpolatedV2_1, interpolatedV3_1) #standard deviation as an approximation of the error
    
    plt.plot(X1, meanIntV_1)
    plt.title("Spreading law for Drop 1 with interpolated data.")
    plt.xlabel("Contact angle, radians")
    plt.ylabel("Contact speed, μm/s")
    plt.errorbar(X1, meanIntV_1, yerr= stdErrIntV_1, ecolor = "g")
    plt.show()
    
    #applying cox-voinox and de gennes fits again to now interpolated data
    
     #De Gennes law interpolated data
     
    (intDeGcoef1, intDeGcovr1) = np.polyfit( (np.power(X1, 2)), meanIntV_1, 1 , cov = True) 
    intDeGcovr1 = np.array(intDeGcovr1).ravel()
    
    intDeGfitline1 = []
    
    for i in range(0, len(X1)):
        fitline =  intDeGcoef1[0]*pow(X1[i], 2) + intDeGcoef1[1]
        intDeGfitline1.append(fitline)
    
    intErrDeGcoef1_1 = np.sqrt(intDeGcovr1[0])
    intErrDeGcoef2_1 = np.sqrt(intDeGcovr1[3])
    
    plt.plot(X1, meanIntV_1)
    plt.plot(X1, intDeGfitline1)
    plt.title("Spreading law for Drop 1. De Gennes fit. (Interpolated data)")
    plt.xlabel("Contact angle, radians")
    plt.ylabel("Contact speed, μm/s")
    plt.errorbar(X1, meanIntV_1, yerr = stdErrV_1, ecolor = "r")
    plt.show()
    
    intDeGredChiSqu_1 = reducedChiSquared(meanIntV_1, intDeGfitline1, stdErrIntV_1, 2)  
    
    print("The best De Gennes fit is: U(θ) = (%5.2f ± %5.2f) θ^2 + (%5.2f ± %5.2f) " %(intDeGcoef1[0], intErrDeGcoef1_1, intDeGcoef1[1], intErrDeGcoef2_1 ))
    print("This fit has reduced chi-squared: %5.2f" %intDeGredChiSqu_1)
    
    #cox-voinox law interpolated data
    (intcoxcoef1, intcoxcovr1) = np.polyfit( (np.power(X1, 3)), meanIntV_1, 1 , cov = True) 
    intcoxcovr1 = np.array(intcoxcovr1).ravel()
    
    intcoxfitline1 = []
    
    for i in range(0, len(X1)):
        fitline =  intcoxcoef1[0]*pow(X1[i], 3) + intcoxcoef1[1]
        intcoxfitline1.append(fitline)
    
    intErrcoxcoef1_1 = np.sqrt(intcoxcovr1[0])
    intErrcoxcoef2_1 = np.sqrt(intcoxcovr1[3])
    
    plt.plot(X1, meanIntV_1)
    plt.plot(X1, intcoxfitline1)
    plt.title("Spreading law for Drop 1. Cox-Voinox fit. (Interpolated data)")
    plt.xlabel("Contact angle, radians")
    plt.ylabel("Contact speed, μm/s")
    plt.errorbar(X1, meanIntV_1, yerr = stdErrV_1, ecolor = "r")
    plt.show()
    
    intcoxredChiSqu_1 = reducedChiSquared(meanIntV_1, intcoxfitline1, stdErrIntV_1, 2)  
    
    print("The best Cox-Voinox fit is: U(θ) = (%5.2f ± %5.2f) θ^3 + (%5.2f ± %5.2f) " %(intcoxcoef1[0], intErrcoxcoef1_1, intcoxcoef1[1], intErrcoxcoef2_1 ))
    print("This fit has reduced chi-squared: %5.2f" %intcoxredChiSqu_1)
    
    #Using interpolated data has made it clearer to see that the de Gennes law is a better fit to larger contact angles.
    
    #drop 2
    
    #quadratic fit, similar to method before
    
    (quadcoef2, quadcovr2) = np.polyfit(meanTheta_2, meanV_2, 2 , cov = True) 
    quadcovr2 = np.array(quadcovr2).ravel()
   
    quadfitline2 = []
    
    for i in range(0, len(meanTheta_2)):
        polyfitline =  quadcoef2[0]*np.power(meanTheta_2[i], 2) + quadcoef2[1]*meanTheta_2[i] + quadcoef2[2]
        quadfitline2.append(polyfitline)
   
    
    errquadcoef1_2 = np.sqrt(quadcovr2[0])
    errquadcoef2_2 = np.sqrt(quadcovr2[4])
    errquadcoef3_2 = np.sqrt(quadcovr2[8])
    
    plt.plot(meanTheta_2, meanV_2)
    plt.plot(meanTheta_2, quadfitline2)
    plt.title("Spreading law for Drop 2. Quadratic fit.")
    plt.xlabel("Contact angle, radians")
    plt.ylabel("Contact speed, μm/s")
    plt.errorbar(meanTheta_2, meanV_2, xerr = stdErrTheta_2, yerr= stdErrV_2, ecolor = "g")
    plt.show()
    
    quadredChiSqu_2 = reducedChiSquared(meanV_2, quadfitline2, stdErrV_2, 3)
    
    print("The best fitted quadratic is: U(θ) = (%5.2f ± %5.2f) θ^2 + (%5.2f ± %5.2f) θ + (%5.2f ± %5.2f) " %(quadcoef2[0], errquadcoef1_2, quadcoef2[1], errquadcoef2_2, quadcoef2[2], errquadcoef3_2))
    print("This has reduced chi-squared: %5.2f" %quadredChiSqu_2)
    
    #A different fit may be more appropriate for Drop 2 (smaller contact angles) as the fit suggests contact speed begins to increase at lower contact angles (essentially when the droplet is almost fully spread out over its surface. This does not make sense physically as we expect that the contact speed slows as the drop reaches its miniumum contact angle. )
    
    #cubic fit, similar to method before
    
    (cubicoef2, cubicovr2) = np.polyfit(meanTheta_2, meanV_2, 3 , cov = True) 
    cubicovr2 = np.array(cubicovr2).ravel()
    
    cubicfitline2 = []
    
    for i in range(0, len(meanTheta_2)):
        polyfitline =  cubicoef2[0]*np.power(meanTheta_2[i], 3) + cubicoef2[1]*np.power(meanTheta_2[i],2) + cubicoef2[2]*meanTheta_2[i] + cubicoef2[3]
        cubicfitline2.append(polyfitline)
        
    
    errcubicoef1_2 = np.sqrt(cubicovr2[0])
    errcubicoef2_2 = np.sqrt(cubicovr2[5])
    errcubicoef3_2 = np.sqrt(cubicovr2[10])  
    errcubicoef4_2 = np.sqrt(cubicovr2[15])
    
    plt.plot(meanTheta_2, meanV_2)
    plt.plot(meanTheta_2, cubicfitline2)
    plt.title("Spreading law for Drop 2. Cubic fit. ")
    plt.xlabel("Contact angle, radians")
    plt.ylabel("Contact speed, μm/s")
    plt.errorbar(meanTheta_2, meanV_2, xerr = stdErrTheta_2, yerr= stdErrV_2, ecolor = "g")
    plt.show()
          
    CUBICredChiSqu_2 = reducedChiSquared(meanV_2, cubicfitline2, stdErrV_2, 4)
    
    print("The best fitted cubic is: U(θ) = (%5.2f ± %5.2f) θ^3 + (%5.2f ± %5.2f) θ^2 + (%5.2f ± %5.2f) θ + (%5.2f ± %5.2f) " %(cubicoef2[0], errcubicoef1_2, cubicoef2[1], errcubicoef2_2, cubicoef2[2], errcubicoef3_2, cubicoef2[3], errcubicoef4_2))
    print("This fit has reduced chi-squared: %5.2f" %CUBICredChiSqu_2)
  
    #De Gennes law, similar to method before
     
    (DeGcoef2, DeGcovr2) = np.polyfit( (np.power(meanTheta_2, 2)), meanV_2, 1 , cov = True) 
    DeGcovr2 = np.array(DeGcovr2).ravel()
   
    DeGfitline2 = []
    
    for i in range(0, len(meanTheta_2)):
        fitline =  DeGcoef2[0]*pow(meanTheta_2[i], 2) + DeGcoef2[1]
        DeGfitline2.append(fitline)
    
    errDeGcoef1_2 = np.sqrt(DeGcovr2[0])
    errDeGcoef2_2 = np.sqrt(DeGcovr2[3])
    
    plt.plot(meanTheta_2, meanV_2)
    plt.plot(meanTheta_2, DeGfitline2)
    plt.title("Spreading law for Drop 2. De Gennes fit.")
    plt.xlabel("Contact angle, radians")
    plt.ylabel("Contact speed, μm/s")
    plt.errorbar(meanTheta_2, meanV_2, xerr = stdErrTheta_2, yerr= stdErrV_2, ecolor = "g")
    plt.show()
    
    DeGredChiSqu_2 = reducedChiSquared(meanV_2, DeGfitline2, stdErrV_2, 2)
    
    print("The best De Gennes fit is: U(θ) = (%5.2f ± %5.2f) θ^2 + (%5.2f ± %5.2f) " %(DeGcoef2[0], errDeGcoef1_2, DeGcoef2[1], errDeGcoef2_2 ))
    print("This fit has reduced chi-squared: %5.2f" %DeGredChiSqu_2)
    
    # Cox- Voinox law, similar to method before
    
    (coxcoef2, coxcovr2) = np.polyfit( (np.power(meanTheta_2, 3)), meanV_2, 1 , cov = True) 
    coxcovr2 = np.array(coxcovr2).ravel()
   
    COXfitline2 = []
    
    for i in range(0, len(meanTheta_2)):
        fitline =  coxcoef2[0]*pow(meanTheta_2[i], 3) + coxcoef2[1]
        COXfitline2.append(fitline)
    
    
    errcoxcoef1_2 = np.sqrt(coxcovr2[0])
    errcoxcoef2_2 = np.sqrt(coxcovr2[3])
    
    plt.plot(meanTheta_2, meanV_2)
    plt.plot(meanTheta_2, COXfitline2)
    plt.title("Spreading law for Drop 2. Cox-Voinox fit.")
    plt.xlabel("Contact angle, radians")
    plt.ylabel("Contact speed, μm/s")
    plt.errorbar(meanTheta_2, meanV_2, xerr = stdErrTheta_2, yerr= stdErrV_2, ecolor = "g")
    plt.show()
    
    COXredChiSqu_2 = reducedChiSquared(meanV_2, COXfitline2, stdErrV_2, 2)
    
    print("The best Cox-Voinox fit is: U(θ) = (%5.2f ± %5.2f) θ^3 + (%5.2f ± %5.2f) " %(coxcoef2[0], errcoxcoef1_2, coxcoef2[1], errcoxcoef2_2 ))
    print("This fit has reduced chi-squared: %5.2f" %COXredChiSqu_2)
 
    #interpolated fits, similar to method before
    
    X2 = np.linspace(meanTheta_2[0], meanTheta_2[-1], len(V1_2)) # evenly spaced array of points where I want to interpolate my data to

    interpolatedV1_2 = interpolate(theta1_2, X2, V1_2)
    interpolatedV2_2 = interpolate(theta2_2, X2, V2_2)
    
    meanIntV_2 = meanOf2Array(interpolatedV1_2, interpolatedV2_2)
    stdErrIntV_2 = stdevOf2Array(interpolatedV1_2, interpolatedV2_2)
    
    plt.plot(X2, meanIntV_2)
    plt.title("Spreading law for Drop 2 with interpolated data.")
    plt.xlabel("Contact angle, radians")
    plt.ylabel("Contact speed, μm/s")
    plt.errorbar(X2, meanIntV_2, yerr= stdErrIntV_2, ecolor = "g")
    plt.show()
    
      #De Gennes law interpolated data
     
    (intDeGcoef2, intDeGcovr2) = np.polyfit( (np.power(X2, 2)), meanIntV_2, 1 , cov = True) 
    intDeGcovr2 = np.array(intDeGcovr2).ravel()
    
    intDeGfitline2 = []
    
    for i in range(0, len(X2)):
        fitline =  intDeGcoef2[0]*pow(X2[i], 2) + intDeGcoef2[1]
        intDeGfitline2.append(fitline)
    
    intErrDeGcoef1_2 = np.sqrt(intDeGcovr2[0])
    intErrDeGcoef2_2 = np.sqrt(intDeGcovr2[3])
    
    plt.plot(X2, meanIntV_2)
    plt.plot(X2, intDeGfitline2)
    plt.title("Spreading law for Drop 2. De Gennes fit. (Interpolated data)")
    plt.xlabel("Contact angle, radians")
    plt.ylabel("Contact speed, μm/s")
    plt.errorbar(X2, meanIntV_2, yerr = stdErrIntV_2, ecolor = "r")
    plt.show()
    
    intDeGredChiSqu_2 = reducedChiSquared(meanIntV_2, intDeGfitline2, stdErrIntV_2, 2)  
    
    print("The best De Gennes fit is: U(θ) = (%5.2f ± %5.2f) θ^2 + (%5.2f ± %5.2f) " %(intDeGcoef2[0], intErrDeGcoef1_2, intDeGcoef2[1], intErrDeGcoef2_2 )) 
    print("This fit has reduced chi-squared: %5.2f" %intDeGredChiSqu_2)
    
    #Cox-voinox law interpolated data
    
    (intcoxcoef2, intcoxcovr2) = np.polyfit( (np.power(X2, 3)), meanIntV_2, 1 , cov = True) 
    intcoxcovr2 = np.array(intcoxcovr2).ravel()
    
    intcoxfitline2 = []
    
    for i in range(0, len(X2)):
        fitline =  intcoxcoef2[0]*pow(X2[i], 3) + intcoxcoef2[1]
        intcoxfitline2.append(fitline)
    
    intErrcoxcoef1_2 = np.sqrt(intcoxcovr2[0])
    intErrcoxcoef2_2 = np.sqrt(intcoxcovr2[3])
    
    plt.plot(X2, meanIntV_2)
    plt.plot(X2, intcoxfitline2)
    plt.title("Spreading law for Drop 2. Cox-Voinox fit. (Interpolated data)")
    plt.xlabel("Contact angle, radians")
    plt.ylabel("Contact speed, μm/s")
    plt.errorbar(X2, meanIntV_2, yerr = stdErrIntV_2, ecolor = "r")
    plt.show()
    
    intcoxredChiSqu_2 = reducedChiSquared(meanIntV_2, intcoxfitline2, stdErrIntV_2, 2)  
    
    print("The best Cox-Voinox fit is: U(θ) = (%5.2f ± %5.2f) θ^3 + (%5.2f ± %5.2f) " %(intcoxcoef2[0], intErrcoxcoef1_2, intcoxcoef2[1], intErrcoxcoef2_2 )) # U(θ) = (5430.96 ± 414.80) θ^3  - 34.13 ± 5.05
    print("This fit has reduced chi-squared: %5.1f" %intcoxredChiSqu_2)
    
    #These results agree with that De Gennes law is a better fit for small contact angles compared to Cox_Voinox law, and vice versa for larger angles.
    #However, more measurements are necessary because there are not enough data sets for the error calculated by the standard deviation to be reliable
    #On surface 1, the velocity scale (U0) can be estimated to be 6397. ± 145.34 micrometres per second, this was the calculated coefficient of theta squared from the Cox-Voinox fit
    #On surface 2, the velocity scale (U0) can be estimated to be 5430. ± 414.80 micrometres per second.
    
    