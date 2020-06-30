import numpy as np
import matplotlib.pyplot as plt

class ACP(object):
    """This is a class used to extract key phases from time series data
    based on the method 'Analysis of Characterising Phases' as described
    in the paper by Richter et al. (2014).

    Attributes:
        data: An n*m array where each row represents a case and each column
        represents a time point.
        threshold: An int (10:90) representing the percentage of the peak 
        vector loading that should be extracted. Default = 90.


    Richter, C., O’Connor, N. E., Marshall, B., & Moran, K. (2014).
    Analysis of characterizing phases on waveforms: an application to vertical
    jumps. Journal of applied biomechanics, 30(2), 316-321.

    Writen by Dr Shane Gore. Contact: Shane.Gore2@Gmail.com
    """

    def __init__(self,data = '',threshold = 90):

        self.data = data
        self.threshold  = threshold

    def identify_phases(self):

        #zero mean data
        data_mean = np.mean(self.data,axis=0)  #Compute mean of each time point
        data_centered = self.data - np.tile(data_mean,(self.data.shape[0],1))

        #Calculate the covariance matrix
        cov_data = np.cov(data_centered.T)

        #Compute Eigen decomosition and order by eigen values
        eigen_values, eigen_vectors = np.linalg.eig(cov_data)
        desc_order = np.flip(np.argsort(eigen_values))
        eigen_values = eigen_values[desc_order]
        eigen_vectors = np.real(eigen_vectors[:, desc_order])

        #Calculate variance explained
        var_explained = (eigen_values / np.sum(eigen_values)) *100

        #Retain eigen vectors that explain at least 1% of the variance
        princ_comp = eigen_vectors[:,var_explained > 1]
        #plt.plot(princ_comp)
        #plt.show()
        
        #Apply Varimax Rotation to minimize the variance of the squared
        #components thereby maximising individual component loading.
        rot_comps = self.varimax(princ_comp)
        #plt.plot(np.abs(rot_comps / np.max(np.abs(rot_comps), axis = 0)))
        #plt.show()
        
        #Extract key phases
        phase_start, phase_end  = self.PC2_keyphase(rot_comps)
        
        #Arrange key phases in descending order
        desc_order   =  np.argsort(phase_start, axis=0)
        phase_start  = phase_start[desc_order]
        phase_end   = phase_end[desc_order]    

        #Merge overlapping phases if present.
        c_phase_end = np.empty(0,int)
        c_phase_start = np.empty(0,int)
        while len(phase_end) > 0:
                mergeidx = (abs(phase_end[0] - phase_end)< 2)
                if  (np.sum(mergeidx) > 1):                
                    c_phase_end = np.append(c_phase_end,max(phase_end[mergeidx]))
                    c_phase_start = np.append(c_phase_start,min(phase_start[mergeidx]))
                    phase_end = np.delete(phase_end,np.where(mergeidx))   
                    phase_start = np.delete(phase_start,np.where(mergeidx))  
                else:
                    c_phase_end   = np.append(c_phase_end,phase_end[0])
                    c_phase_start = np.append(c_phase_start,phase_start[0])
                    phase_end = np.delete(phase_end,0)
                    phase_start  = np.delete(phase_start,0)
        
        
        return(c_phase_start.astype('int'), c_phase_end.astype('int'))


    def PC2_keyphase(self,rot_comps):
        #Function to identify key phases of the rotated
        #principle compnonents.

        threshold = round((self.threshold /100),2)

        #convert rotated PC vectors to absolute values
        rot_comps = abs(rot_comps)

        #identiy peak of PC vectors
        peak_pos = np.argmax(rot_comps,axis=0)
        peak  = np.max(rot_comps,axis=0)

        phase_start = np.zeros((rot_comps.shape[1],1))
        phase_end = np.zeros((rot_comps.shape[1],1))
#        for i,pc in enumerate (rot_comps):
#            pc = abs(pc)
#            max_f = np.argmax(pc)
#            ff = np.where(pc[:max_f]< threshold)[0][-1]
#            lf = np.where(pc[max_f:]< threshold)[0][0]
#            phase_start[i],phase_end[i] = ff,lf
#        return (phase_start,phase_end)


        for n in range(rot_comps.shape[1]):
            try:
                phase_start[n] = np.max(np.where(rot_comps[0:peak_pos[n]+1,n]
                < peak[n] * threshold))
            except:
                phase_start[n] = 0 #if threshold not found
            try:
                phase_end[n] =  peak_pos[n] + np.min(np.where
                (rot_comps[peak_pos[n]:rot_comps.shape[0]+1,n]
                < peak[n] * threshold)) + 1
            except:
                phase_end[n] = rot_comps.shape[0] #if threshold not found

        return (phase_start,phase_end)


    def varimax(self,Phi, gamma = 1, q = 20, tol = 1e-6):
    # Function to rotate prinicple components to minimize the variance
    #of the squared components.

    #Adapted from https://en.wikipedia.org/wiki/Talk:Varimax_rotation

        p,k = Phi.shape
        R = np.eye(k)
        d=0
        for i in range(q):
            d_old = d
            Lambda = np.dot(Phi, R)
            u,s,vh = np.linalg.svd(np.dot(Phi.T,np.asarray(Lambda)**3 - 
                            (gamma/p) * np.dot(Lambda, 
                            np.diag(np.diag(np.dot(Lambda.T,Lambda))))))
            R = np.dot(u,vh)
            d = np.sum(s)
            try:
                if d/d_old  < tol: break
            except: 
                continue #account for division by zero.
            return np.dot(Phi, R)
