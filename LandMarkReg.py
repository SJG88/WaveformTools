import numpy as np
from scipy import interpolate

class LandMarkReg(object):
    """This is a class to warp the velocity of signals so that the user
    identified landmarks align. This class is based on the MATLAB class
    written by Dr Chris Richter and described in the paper by Moudy et al 2018.

    Modifications:
    - Uses divide and conquer, binary search algorithm to speed up convergence.
    - Uses Akima spline rather than cubic spline to reduce overshooting between
      landmarks.
    - Smaller error tolerance resutls in better alignment.
    
       
    Attributes:
        nrows: An int representing the number of rows in the signal.
        current_pos: A int representing the position of the ladmark in the
        current signal.
        landmark_pos: An int representing the position of the registration 
        landmark.
    
    
    Moudy, S., Richter, C., & Strike, S. (2018). Landmark registering waveform
    data improves the ability to predict performance measures.
    ournal of biomechanics, 78, 109-117.

    Writen by Dr Shane Gore 2020. Contact Shane.Gore2@Gmail.com"""


    def __init__(self, nrows =101, current_pos ='', landmark_pos=''):
        
        self.nrows = nrows
        self.current_pos  = current_pos
        self.landmark_pos = landmark_pos

    def DynamicTimeWarp(self):

        #Preallocate array for speed and set up array for warping.
        speedMAT = np.empty((self.nrows,1))
        speedMAT.fill(np.nan)
        speedMAT[np.array([0, self.landmark_pos, self.nrows -1])] = 1
    
        #landmarkss
        mpos = [0,self.landmark_pos, self.nrows]
        cpos = [0,self.current_pos, self.nrows]
    
        #Loop through the number of landmarks and warp velocity as required.
        warp = np.empty([0,1],np.float64)     
        for n in range(len(mpos)-1):
            c_len = cpos[n+1] - cpos[n]
            
            if n == 0:
                cutting = False
            else:
                cutting = True
               
            warp = np.append(warp,self.warpfnc(speedMAT[mpos[n]:mpos[n+1]+1],c_len,cutting))
        
        #convert to frames, zero index
        warp = np.cumsum(warp) -1
        
        return(warp)
        

    def warpfnc(self,raw_sig,c_len,cutting):
                
       #create a copy of raw signal.    
        c_sig = np.empty_like(raw_sig)
        c_sig[:] = raw_sig

        
        #Initiate function search with excessive threshold.
        c_thresh = 5
        thresh = 5
        midpoint = round(len(c_sig)*0.5)
        c_sig[midpoint] = c_thresh
    
        #interpolate between start mid and end of signal and calculate
        #the sum of the signal.
        x1 = np.where(~np.isnan(c_sig))
        x2 = np.where(np.isnan(c_sig))
        
        f = interpolate.Akima1DInterpolator(x1[0], c_sig[~np.isnan(c_sig)])
        c_sig[np.isnan(c_sig)] =  f(x2[0])
        
        #Binary search algorithm to find the midpoint value which will
        # provide an appropriate time warping function
        while abs(sum(c_sig) - c_len) >  0.001:
            if sum(c_sig)  > c_len:# clandmark occures early, must be slowed down.
                
                thresh = thresh/2
                c_thresh  = (c_thresh - thresh)
                
                c_sig = np.empty_like(raw_sig)
                c_sig[:] = raw_sig
                c_sig[midpoint] = c_thresh

                #interpolate between start mid and end of signal and calculate
                #the sum of the signal.
                f = interpolate.Akima1DInterpolator(x1[0], c_sig[~np.isnan(c_sig)])
                c_sig[np.isnan(c_sig)] =  f(x2[0])

                # in case there are minus values make them 0
                c_sig[c_sig<0] = 0;
                
                # remove first data point if requested
                if cutting == True:
                    c_sig = c_sig[1:]

            elif sum(c_sig)  < c_len: # clandmark occures late, must be sped up.
                thresh = thresh/2
                c_thresh  = (c_thresh + thresh)
                c_sig = np.empty_like(raw_sig)
                c_sig[:] = raw_sig
                c_sig[midpoint] = c_thresh

                #interpolate between start mid and end of signal and calculate
                #the sum of the signal.
                f = interpolate.Akima1DInterpolator(x1[0], c_sig[~np.isnan(c_sig)])
                c_sig[np.isnan(c_sig)] =  f(x2[0])

                # remove first data point if requested
                if cutting == True:
                    c_sig = c_sig[1:]
                    
        return(c_sig)
