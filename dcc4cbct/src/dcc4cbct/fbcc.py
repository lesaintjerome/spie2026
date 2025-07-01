import itk
from itk import RTK as rtk
import matplotlib.pyplot as plt
import numpy as np

from .misc import RecupParam, ExtractSlice, GetMatrixFromParams, DecomposeProjectionMatrix, Normalize2DPoint, GetDetectorCoordinatesToFixedSystemMatrix



class FBProblem() :
    """
        Define a problem based on pair-wise consistency measure, involving epipolar geometry.
        The attribute self.dcc carries the type of DCC that will be used in the computation of the cost function.
    """
    def __init__(self,geometry,proj) :
        self.geo = geometry
        self.proj = proj
        
        if (np.array(proj.GetLargestPossibleRegion().GetSize())[-1] == len(geometry.GetGantryAngles())) :
            self.nb_proj = len(geometry.GetGantryAngles())
        else :
            raise ValueError('Size of proj does not match size of geometry')

        self.pairs = []
        self.pairids = []

    def create_pair(self,pairid) :
        i0 = pairid[0]
        i1 = pairid[1]
        g0 = rtk.ThreeDCircularProjectionGeometry.New()
        sid,sdd,ga,dx,dy,oa,ia,sx,sy = RecupParam(self.geo,i0)
        g0.AddProjectionInRadians(sid,sdd,ga,dx,dy,oa,ia,sx,sy)
        #g0.AddProjection(self.geo.GetMatrix(i0))
        g1 = rtk.ThreeDCircularProjectionGeometry.New()
        sid,sdd,ga,dx,dy,oa,ia,sx,sy = RecupParam(self.geo,i1)
        g1.AddProjectionInRadians(sid,sdd,ga,dx,dy,oa,ia,sx,sy)
        #g1.AddProjection(self.geo.GetMatrix(i1))
        p0 = ExtractSlice(self.proj,i0)
        p1 = ExtractSlice(self.proj,i1)
        pair = EpipolarPair(g0,g1,p0,p1)
        pair.compute_line_integrals()
        return pair

    def add_pair_to_problem(self,pairid) :
        self.pairids.append(pairid)
        self.pairs.append(self.create_pair(pairid))

    def AnalyzePair(self,pairid):
        if pairid not in self.pairids:
            self.create_pair(pairid)
        idx = self.pairids.index(pairid)
        pair = self.pairs[idx] 
        plt.figure()
        plt.subplot(121)
        plt.plot(pair.cf0,label=f'proj {pairid[0]}')
        plt.plot(pair.cf1,label=f'proj {pairid[1]}')
        plt.legend()
        plt.title('Fanbeam consistency function')
        plt.subplot(122)
        plt.plot(pair.cg0,label=f'proj {pairid[0]}')
        plt.plot(pair.cg1,label=f'proj {pairid[1]}')
        plt.legend()
        plt.title('Grangeat consistency function')
        return 1

    
class EpipolarPair() :
    """
        This class implements the pair-wise epipolar consistency condition
    """
    def __init__(self,g0,g1,p0,p1) :
        """ 
            g0,g1 : geometries of each projection (in the sense of RTK)
            p0,p1 : projection ITK images
        """
        self.sizep0 = np.array(p0.GetLargestPossibleRegion().GetSize())
        self.sizep1 = np.array(p1.GetLargestPossibleRegion().GetSize())
        
        # Set (u,v) coords in image 0
        self.u0 = np.arange(p0.GetOrigin()[0],p0.GetOrigin()[0]+(self.sizep0[0]-1)*p0.GetSpacing()[0]+1e-9,p0.GetSpacing()[0])
        self.v0 = np.arange(p0.GetOrigin()[1]+(self.sizep0[1]-1)*p0.GetSpacing()[1],p0.GetOrigin()[1]-1e-9,(-1.0)*p0.GetSpacing()[1]) #Change orientation of v-axis
        self.U0,self.V0 = np.meshgrid(self.u0,self.v0)
        
        # Set (u,v) coords in image 1
        self.u1 = np.arange(p1.GetOrigin()[0],p1.GetOrigin()[0]+(self.sizep1[0]-1)*p1.GetSpacing()[0]+1e-9,p1.GetSpacing()[0])
        self.v1 = np.arange(p1.GetOrigin()[1]+(self.sizep1[1]-1)*p1.GetSpacing()[1],p1.GetOrigin()[1]-1e-9,(-1.0)*p1.GetSpacing()[1]) # Change orientation of v-axis
        self.U1,self.V1 = np.meshgrid(self.u1,self.v1)


        self.g0 = g0
        self.g1 = g1
        ga0 = self.g0.GetGantryAngles()[0]
        ga1 = self.g1.GetGantryAngles()[0]

        # Change of coords to account for the fact that RTK 3D coord sys is left-handed. Flip y-axis (i.e. right-multiply the projection matrix by Md.
        # Also flip v-axis to account for (u,v) = (x,y) (left-multiply the resulting matrix by M)
        flip_v_matrix = np.identity(3)
        flip_v_matrix[1,1] = -1.0
        flip_y_matrix = np.identity(4)
        flip_y_matrix[1,1] = -1.0

        self.p0 = p0
        sid,sdd,ga,dx,dy,oa,ia,sx,sy = RecupParam(self.g0,0)
        tmp = GetMatrixFromParams(sid,sdd,ga,dx,dy,oa,ia,sx,sy)
        #tmp = np.array(self.g0.GetMatrix(0)).reshape([3,4])
        self.P0 = np.dot(flip_v_matrix,np.dot(tmp,flip_y_matrix))
        s0 , K0 , R0 = DecomposeProjectionMatrix(self.P0)
        self.s0 = s0 # DecomposeProjectionMatrix() returns s0 as a 4-vector, with 4th coord set to 1.
        self.f0 = K0[0,0]*(-1.0) # z-axis is pointing in the direction of the source.
        self.offsetu0 = K0[0,2]
        self.offsetv0 = K0[1,2]
        
        self.p1 = p1
        sid,sdd,ga,dx,dy,oa,ia,sx,sy = RecupParam(self.g1,0)
        tmp = GetMatrixFromParams(sid,sdd,ga,dx,dy,oa,ia,sx,sy)
        #tmp = np.array(self.g1.GetMatrix(0)).reshape([3,4])
        self.P1 = np.dot(flip_v_matrix,np.dot(tmp,flip_y_matrix))
        s1 , K1 , R1 = DecomposeProjectionMatrix(self.P1)
        self.s1 = s1
        self.f1 = K1[0,0]*(-1.0) 
        self.offsetu1 = K1[0,2]
        self.offsetv1 = K1[1,2]
        
        # Compute the epipoles
        self.e0 = Normalize2DPoint(np.dot(self.P0,self.s1))
        self.e1 = Normalize2DPoint(np.dot(self.P1,self.s0))
        np.abs(self.e0[0]) < self.u0
        # Check if epipoles are in the detector
        

        ## Compute fundamental matrix
        #tmp1 = SkewSymMatrixFromVector(self.e1)
        #pinv0 = np.linalg.pinv(self.P0)
        #self.F = np.dot(tmp1,np.dot(self.P1,pinv0)) # The fundamental matrix
        
        self.ar0 = itk.GetArrayFromImage(self.p0)
        self.ar1 = itk.GetArrayFromImage(self.p1)
        
        # Compute cosine weight
        self.cosine0 = self.f0/np.sqrt((self.U0-self.offsetu0)**2+(self.V0-self.offsetv0)**2+self.f0**2)
        self.cosine1 = self.f1/np.sqrt((self.U1-self.offsetu1)**2+(self.V1-self.offsetv1)**2+self.f1**2)

        # Compute inverse of distance to the epipole
        self.invdist0 = 1.0/np.sqrt((self.U0-self.e0[0])**2+(self.V0-self.e0[1])**2)
        self.invdist1 = 1.0/np.sqrt((self.U1-self.e1[0])**2+(self.V1-self.e1[1])**2)

        self.weightedProj0 = self.ar0*self.cosine0*self.invdist0
        self.weightedProj1 = self.ar1*self.cosine1*self.invdist1

        # Compute the 1/cos term before the integral
        self.exe = (self.s1-self.s0)[:3]
        self.exe /=np.linalg.norm(self.exe)
        ezo0 = R0[2,:]
        ezo1 = R1[2,:]
        self.preweight0 = 1.0/np.abs(np.dot(ezo0,self.exe))
        self.preweight1 = 1.0/np.abs(np.dot(ezo1,self.exe))

        # Compute the epipolar geometry
        self.eye = np.cross(self.s0[:3],self.s1[:3])
        self.eye /= np.linalg.norm(self.eye)
        self.eze = np.cross(self.exe,self.eye)
        self.R12 = np.array([self.exe,self.eye,self.eze]).T # From epipolar to world
        self.d = np.abs(np.dot(self.eze,self.s0[:3]))
        
        self.LineIntegralsAreComputed = False
        
        ## Compute the homography H0 from virtual det to detector 0
        #alpha0 = np.dot(self.s0[:3],self.exe)
        #Kv0 = np.identity(3)
        #Kv0[0,0] = -self.d 
        #Kv0[1,1] = -self.d
        #Kv0[0,2] = alpha0
        #A0 = np.dot(self.P0[:,:3],self.R12)
        #self.H0 = np.dot(K0,np.linalg.inv(A0))
        #
        ## Compute the homography H1 from virtual det to detector 1
        #alpha1 = np.dot(self.s1[:3],self.exe)
        #Kv1 = np.identity(3)
        #Kv1[0,0] = -self.d 
        #Kv1[1,1] = -self.d
        #Kv1[0,2] = alpha1
        #A1 = np.dot(self.P1[:,:3],self.R12)
        #self.H1 = np.dot(K1,np.linalg.inv(A1))

    def compute_fanbeam_consistency(self) :
        if self.LineIntegralsAreComputed == False :
            self.compute_line_integrals()
        return ((self.cf0 - self.cf1)**2).sum()
    
    def compute_grangeat_consistency(self) :
        if self.LineIntegralsAreComputed == False :
            self.compute_line_integrals()
        return ((self.cg0 - self.cg1)**2).sum()

    def compute_line_integrals(self) :
        ImageType = itk.Image[itk.F,3]
        
        ## Compute the fan-beam function in projection 1
        tmp = np.zeros([self.weightedProj1.shape[1],3,self.weightedProj1.shape[0]],dtype=np.float32)
        tmp[:,1,:] = self.weightedProj1.T[::-1]
        self.im1 = itk.GetImageFromArray(tmp)
        self.sizeim1 = np.array(self.im1.GetLargestPossibleRegion().GetSize())
        sp1 = self.p1.GetSpacing()[0]
        self.im1.SetSpacing(np.array([sp1,sp1,sp1]))
        self.im1.SetOrigin(-0.5*sp1*(self.sizeim1-1))
        # Set geometry g1
        g1 = rtk.ThreeDCircularProjectionGeometry.New()
        sid = (-1.0)*self.e1[0] # U-axis and Z-axis in opposite dir
        sx = (-1.0)*self.e1[1] # V-axis of detect and Y-axis are in opposite dir
        halfdetectoru =  (self.sizeim1[2]*sp1/2.0 + sp1/2.0)
        f1 = np.abs(sid)+halfdetectoru  # Place 1D detector next to physical 2D detector.
        halfdetectorv = (self.sizeim1[2]*sp1/2.0)
        g1.AddProjection(sid,f1,0,0.0,0.0,0.0,0.0,sx,0.0)
        # Set Projection 1
        const = rtk.ConstantImageSource[ImageType].New()
        const.SetSize((int(self.sizeim1[0]),1,1))
        const.SetSpacing((sp1,sp1,sp1))
        const.SetOrigin(-0.5*sp1*(np.array(const.GetSize())-1))
        fwd = rtk.JosephForwardProjectionImageFilter[ImageType,ImageType].New()
        fwd.SetInput(const.GetOutput())
        fwd.SetInput(1,self.im1)
        fwd.SetGeometry(g1)
        fwd.Update()
        self.fanbeam1 = itk.GetArrayFromImage(fwd.GetOutput())[0,0,:]
        self.gf1 = self.preweight1*self.fanbeam1
        
        ## Compute the fan-beam function in projection 0
        tmp = np.zeros([self.weightedProj0.shape[1],3,self.weightedProj0.shape[0]],dtype=np.float32)
        tmp[:,1,:] = self.weightedProj0.T#[::-1]
        self.im0 = itk.GetImageFromArray(tmp)
        self.sizeim0 = np.array(self.im0.GetLargestPossibleRegion().GetSize())
        sp0 = self.p0.GetSpacing()[0]
        self.im0.SetSpacing(np.array([sp0,sp0,sp0]))
        self.im0.SetOrigin(-0.5*sp0*(self.sizeim0-1))
        # Set geometry g0
        g0 = rtk.ThreeDCircularProjectionGeometry.New()
        sid = self.e0[0] # U-axis and Z-axis in opposite dir
        sx = self.e0[1] # V-axis of detect and Y-axis are in opposite dir
        halfdetectoru =  (self.sizeim0[2]*sp0/2.0 + sp0/2.0)
        f0 = np.abs(sid)+halfdetectoru  # Place 1D detector next to physical 2D detector.
        halfdetectorv = (self.sizeim0[2]*sp0/2.0)
        g0.AddProjection(sid,f0,0,0.0,0.0,0.0,0.0,sx,0.0)
        # Set Projection 0
        const = rtk.ConstantImageSource[ImageType].New()
        const.SetSize((int(self.sizeim0[0]),1,1))
        const.SetSpacing((sp0,sp0,sp0))
        const.SetOrigin(-0.5*sp0*(np.array(const.GetSize())-1))
        fwd = rtk.JosephForwardProjectionImageFilter[ImageType,ImageType].New()
        fwd.SetInput(const.GetOutput())
        fwd.SetInput(1,self.im0)
        fwd.SetGeometry(g0)
        fwd.Update()
        self.fanbeam0 = itk.GetArrayFromImage(fwd.GetOutput())[0,0,:]
        self.gf0 = self.preweight0*self.fanbeam0


        # Compute 3D position of each pixel of the 1D detector
        self.fb_detector_0 = np.zeros([4,self.sizep0[1]])
        self.fb_detector_0[0,:] = (-1.0)*np.ones(self.sizep0[1])*0.5*sp0*(self.sizep0[0]+1)
        self.fb_detector_0[1,:] = self.v0
        self.fb_detector_0[3,:] = np.ones(self.sizep0[1])
        
        sid,sdd,ga,dx,dy,oa,ia,sx,sy = RecupParam(self.g0,0)
        mat0 = GetDetectorCoordinatesToFixedSystemMatrix(sid,sdd,ga,dx,dy,oa,ia,sx,sy)
        #mat0 = np.array(self.g0.GetProjectionCoordinatesToFixedSystemMatrix(0)).reshape([4,4])
        world_coord = np.dot(mat0,self.fb_detector_0)-self.s0.reshape([4,1])
        fb_det_0_epi = np.dot(self.R12.T,world_coord[:3,:])
        # Convert the 3D position into an epipolar angle theta
        self.theta_0 = np.arcsin(fb_det_0_epi[1,:]/np.sqrt(fb_det_0_epi[1,:]**2+fb_det_0_epi[2,:]**2))[::-1] # s'arranger pour que ce soit croissant


        # Compute 3D position of each pixel of the 1D detector
        self.fb_detector_1 = np.zeros([4,self.sizep1[1]])
        self.fb_detector_1[0,:] = np.ones(self.sizep1[1])*0.5*sp1*(self.sizep1[0]+1)
        self.fb_detector_1[1,:] = self.v1
        self.fb_detector_1[3,:] = np.ones(self.sizep1[1])
        
        sid,sdd,ga,dx,dy,oa,ia,sx,sy = RecupParam(self.g1,0)
        mat1 = GetDetectorCoordinatesToFixedSystemMatrix(sid,sdd,ga,dx,dy,oa,ia,sx,sy)
        #mat1 = np.array(self.g1.GetProjectionCoordinatesToFixedSystemMatrix(0)).reshape([4,4])
        world_coord = np.dot(mat1,self.fb_detector_1)-self.s1.reshape([4,1])
        fb_det_1_epi = np.dot(self.R12.T,world_coord[:3,:])
        # Convert the 3D position into an epipolar angle theta
        self.theta_1 = np.arcsin(fb_det_1_epi[1,:]/np.sqrt(fb_det_1_epi[1,:]**2+fb_det_1_epi[2,:]**2))[::-1] # s'arranger pour que ce soit croissant

        # Relevant only if any plane valid in proj 0 is also valid ini proj 1
        # i.e. ne tient pas compte des situations géometriques "tordues".
        if self.theta_0.shape[0] == self.theta_1.shape[0] :
            self.nbsamples = self.theta_0.shape[0]
        else :
            raise ValueError("Number of planes in proj 0 does not equal number of planes in proj 1")
             
        
        self.theta_min = max(self.theta_0[0],self.theta_1[0])
        self.theta_max = min(self.theta_0[-1],self.theta_1[-1])
        
        self.step_theta = (sp0/f0+sp1/f1)/2.0 # Inutile de prendre un pas d'echantillonnage en theta plus fin que le pas d'ech de la proj fb
        self.sampled_theta = np.linspace(self.theta_min,self.theta_max,self.nbsamples)
        self.cf0 = np.zeros(self.nbsamples)
        self.cf1 = np.zeros(self.nbsamples)
        self.cg0 = np.zeros(self.nbsamples)
        self.cg1 = np.zeros(self.nbsamples)
        
        # Compute fanbeam function by interpolation.
        for i in range(self.sampled_theta.shape[0]) :
            self.cf0[i] = np.interp(self.sampled_theta[i],self.theta_0,self.gf0)
            self.cf1[i] = np.interp(self.sampled_theta[i],self.theta_1,self.gf1)
        
        # Compute grangeat function by centred finite difference.
        self.cg0 = (self.cf0[1:]-self.cf0[:-1])/(2*self.step_theta)
        self.cg1 = (self.cf1[1:]-self.cf1[:-1])/(2*self.step_theta)

        self.LineIntegralsAreComputed = True

        return 1
    
    def analyze_pair(self, figsize=(10,10), title=None) :
        fig,ax = plt.subplots(2,2,figsize=figsize)
        fig.suptitle(title)
        ax[0,0].plot(self.cf0,label=f'proj 0')
        ax[0,0].plot(self.cf1,label=f'proj 1')
        ax[0,0].legend()
        ax[0,0].set_title(f'Fanbeam consistency function : {self.compute_fanbeam_consistency():.3f}')
        ax[0,1].plot(self.cg0,label=f'proj 0')
        ax[0,1].plot(self.cg1,label=f'proj 1')
        ax[0,1].legend()
        ax[0,1].set_title(f'Grangeat consistency function - {self.compute_grangeat_consistency():.3f}')
        self.display_pair(fig,ax[1])
        return 1

    def check_pair(self):
        """This function indicates if the epipoles are in the detector or not.
        More precisely, it computes
        - the distance d(e,Od) : distance from epipole to detector center
        - the distance d(Od,)"""
        # Find
        self.e0_within_u = np.abs(self.e0[0]) < self.u0.max()
        self.e0_within_v = np.abs(self.e0[1]) < self.v0.max()
        self.e1_within_u = np.abs(self.e1[0]) < self.u1.max()
        self.e1_within_v = np.abs(self.e1[1]) < self.v1.max()
        self.e0_within_det = self.e0_within_u and self.e0_within_v
        self.e1_within_det = self.e1_within_u and self.e1_within_v
        self.invalid_pair = self.e0_within_det or self.e1_within_det

    def check_pair_2(self):
        # Segment projection with a margin
        # Convert epipole coords in pixel coords
        # Check if epipole in the support of proj.
        return 1

    def compute_epipolar_lines(self):
        # Det 0
        umin0 = self.u0.min()
        umax0 = self.u0.max()
        vumin0 = (self.fb_detector_0[1]-self.e0[1])/(self.fb_detector_0[0]-self.e0[0])*(umin0 - self.e0[0] + self.e0[1])
        vumax0 = (self.fb_detector_0[1]-self.e0[1])/(self.fb_detector_0[0]-self.e0[0])*(umax0 - self.e0[0] + self.e0[1])

        # Det 1
        umin1 = self.u1.min()
        umax1 = self.u1.max()
        vumin1 = (self.fb_detector_1[1]-self.e1[1])/(self.fb_detector_1[0]-self.e1[0])*(umin1 - self.e1[0] + self.e1[1])
        vumax1 = (self.fb_detector_1[1]-self.e1[1])/(self.fb_detector_1[0]-self.e1[0])*(umax1 - self.e1[0] + self.e1[1])
        return umin0, umax0, vumin0, vumax0, umin1, umax1, vumin1, vumax1




    def display_pair(self,fig,ax):
        vmin = min(self.ar0.min(),self.ar1.min())
        vmax = max(self.ar0.max(),self.ar1.max())

        # Get epipolar lines
        umin0, umax0, vumin0, vumax0, umin1, umax1, vumin1, vumax1 = self.compute_epipolar_lines()

        #fig, ax = plt.subplots(1,2,figsize=(10,5))
        ax[1].imshow(self.p0[:],extent=(self.u0.min(),self.u0.max(),self.v0.min(),self.v0.max()), vmin=vmin,vmax=vmax)
        ax[1].set_axis_off()
        ax[1].set_title("Projection 0")
        for i, (ymin,ymax) in enumerate(zip(vumin0[::20], vumax0[::20])):
            ax[1].plot((umin0, umax0), (ymin,ymax),'r')

        ax[0].imshow(self.p1[:],extent=(self.u1.min(),self.u1.max(),self.v1.min(),self.v1.max()), vmin=vmin,vmax=vmax)
        ax[0].set_axis_off()
        ax[0].set_title("Projection 1")
        for i, (ymin,ymax) in enumerate(zip(vumin1[::20], vumax1[::20])):
            ax[0].plot((umin1, umax1), (ymin,ymax),'r')


