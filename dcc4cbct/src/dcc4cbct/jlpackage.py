#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 13:58:04 2019

@author: jl258167
"""

import itk
from itk import RTK as rtk
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage


#######################################################################
# RTK functions
#######################################################################

def Simulate2DProjectionsSheppLoganPhantom(geom,image_size,image_spacing,phantomscale) :
    """ Returns a stack of simulated projections of a Shepp Logan phantom.
    image_size : size of the detector ([n,m])
    image_spacing : size of pixel (float). Assumes isotropic pixel.
    pahntomscale : size of the phantom (float). See RTK doc.

    """
    ImageType = itk.Image[itk.F, 3]
    const = rtk.ConstantImageSource[ImageType].New()
    const.SetConstant(0.)
    const.SetSpacing([image_spacing,image_spacing,image_spacing])
    const.SetSize([image_size[0],image_size[1],len(geom.GetGantryAngles())])
    const.SetOrigin(-0.5*image_spacing*(np.array(const.GetSize())-1))
    sl = rtk.SheppLoganPhantomFilter[ImageType,ImageType].New()
    sl.SetInput(const.GetOutput())
    sl.SetGeometry(geom)
    sl.SetPhantomScale(phantomscale)
    sl.Update()
    return sl.GetOutput()


def Simulate2DProjectionsFromVoxelizedPhantom(geom,vox_phantom,image_size,image_spacing) :
    """ Returns a stack of simulated projections of a 2D pixelized phantom.
    image_size : size of the detector ([n,m])
    image_spacing : size of pixel (float). Assumes isotropic pixel.
    pahntomscale : size of the phantom (float). See RTK doc.

    """
    ImageType = itk.Image[itk.F, 3]
    const = rtk.ConstantImageSource[ImageType].New()
    const.SetConstant(0.)
    const.SetSpacing([image_spacing,image_spacing,image_spacing])
    const.SetSize([image_size[0],image_size[1],len(geom.GetGantryAngles())])
    const.SetOrigin(-0.5*image_spacing*(np.array(const.GetSize())-1))
    fwd = rtk.JosephForwardProjectionImageFilter[ImageType,ImageType].New()
    fwd.SetInput(const.GetOutput())
    fwd.SetInput(1,vox_phantom)
    fwd.SetGeometry(geom)
    fwd.Update()
    return fwd.GetOutput()


def NormalizeVector(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def FDKRecons(geom,
              proj,
              reconstructed_image_size,
              reconstructed_image_spacing,
              shortscan_flag=False,
              recons_origin=[]) :
    ImageType = itk.Image[itk.F,3]
    const = rtk.ConstantImageSource[ImageType].New()
    const.SetConstant(0.)
    const.SetSpacing(reconstructed_image_spacing)
    const.SetSize(reconstructed_image_size)
    if recons_origin == [] :
        const.SetOrigin(-0.5*reconstructed_image_spacing*(np.array(const.GetSize())-1))
    else :
        const.SetOrigin(recons_origin)
    
    fdk = rtk.FDKConeBeamReconstructionFilter[ImageType].New()
    fdk.SetGeometry(geom)
    fdk.SetInput(const.GetOutput())
    
    if shortscan_flag :
        # Parker Short Scan weight
        pss = rtk.ParkerShortScanImageFilter[ImageType].New()
        pss.SetInput(proj)
        pss.SetGeometry(geom)
        pss.InPlaceOff()
        fdk.SetInput(1,pss.GetOutput())
    else :
        fdk.SetInput(1,proj)
    
    fdk.Update()
    return fdk.GetOutput()

def FBFDKRecons(geom,sino,reconstructed_image_size,pixel_size,reconstructed_image_spacing,shortscan_flag=False,recons_origin=[]) :
    # ATTENTION : geom doit être plane... i.e. ia = sy = dy = 0
    # Aucun controle ici.
    # Sino dim : nbprojs x nb_pixels
    # Suppose proj centrees
    proj = itk.GetImageFromArray(np.array(np.repeat(sino[:,np.newaxis,:],3,1),dtype=np.float32))
    proj.SetSpacing(pixel_size)
    x,y,z = proj.GetLargestPossibleRegion().GetSize()
    proj.SetOrigin([-0.5*pixel_size*(x-1),-0.5*pixel_size*(y-1),-0.5*pixel_size*(z-1)])
    
    return FDKRecons(geom,proj,reconstructed_image_size,reconstructed_image_spacing,shortscan_flag,recons_origin=recons_origin)


def ReadGeometryFile(file) :
    """ Read a RTK geometry file and returns an RTK geometry object"""
    reader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
    reader.SetFilename(file)
    reader.GenerateOutputInformation()
    return reader.GetOutputObject()


def AddNoiseToProjections(proj_no_noise,I0,dh2o) :
    ar = itk.GetArrayFromImage(proj_no_noise)
    ar = I0*np.exp(-dh2o*ar)
    ar = np.maximum(np.random.poisson(ar),1)
    ar = np.log(I0/ar)/dh2o
    proj = itk.GetImageFromArray(ar.astype(np.float32))
    proj.CopyInformation(proj_no_noise)
    
    return proj

#######################################################################
# divergent pairwise DCC
#######################################################################
class fbdccProblem() :
    def __init__(self,proj,geo) :
        self.proj = proj
        self.geo = geo
        self.pairs = []
        
        self.spu = self.proj.GetSpacing()[0]
        self.origu = self.proj.GetOrigin()[0]
        self.dimu = self.proj.GetLargestPossibleRegion().GetSize()[0]
        self.u = self.origu + self.spu*np.arange(self.dimu)
        
        self.D = self.geo.GetSourceToDetectorDistances()[0]
        self.cos_weights = self.D/np.sqrt(self.u**2+self.D**2)
        
        
    def compute_fbdcc(self,pair) :
        i = int(pair[0])
        j = int(pair[1])
        
        # Get epipoles
        source_0 = np.array(self.geo.GetSourcePosition(i))
        source_1 = np.array(self.geo.GetSourcePosition(j))
        matrix_0 = np.array(self.geo.GetMatrix(i))
        matrix_1 = np.array(self.geo.GetMatrix(j))
        epipole_0 = NormalizePointInHomogeneousCoords(np.dot(matrix_0,source_1))
        epipole_1 = NormalizePointInHomogeneousCoords(np.dot(matrix_1,source_0))
        
        # Compute distance to epipoles
        dist_u0_e0 = np.abs(self.u-epipole_0[0])
        dist_u1_e1 = np.abs(self.u-epipole_1[0])
        
        # Weight projections 
        proj0 = self.proj[i,:,:]
        proj1 = self.proj[j,:,:]
        weighted_proj_0 = proj0 * self.cos_weights / dist_u0_e0
        weighted_proj_1 = proj1 * self.cos_weights / dist_u1_e1
        
        ## Compute outer weight
        # Compute normal to detectors
        normal_0 = np.array(self.geo.GetRotationMatrix(i))[2,:3]
        normal_1 = np.array(self.geo.GetRotationMatrix(j))[2,:3]
        # Compute baseline unit vector
        baseline = (source_1[:3]-source_0[:3])/np.linalg.norm(source_1[:3]-source_0[:3])
        # Compute outer weight
        outer_weight_0 = 1./np.abs(np.dot(baseline,normal_0))
        outer_weight_1 = 1./np.abs(np.dot(baseline,normal_1))
        
        l0 = outer_weight_0 * weighted_proj_0.sum()*self.spu
        l1 = outer_weight_1 * weighted_proj_1.sum()*self.spu
        
        return l0,l1
    
            
    def evaluate_dcc(self,pairs) :
        dcc_vals = np.zeros([len(pairs),2])
        for i,p in enumerate(pairs) : 
            dcc_vals[i,:] = self.compute_fbdcc(p)
        return dcc_vals

class fbdccPair() :
    def __init__(self,pb,pairid) :
        self.pb = pb
        self.pairid = pairid
    
    def compute_fbdcc(self) :
        i = int(self.pairid[0])
        j = int(self.pairid[1])
        
        # Get epipoles
        self.source_0 = np.array(self.pb.geo.GetSourcePosition(i))
        self.source_1 = np.array(self.pb.geo.GetSourcePosition(j))
        self.matrix_0 = np.array(self.pb.geo.GetMatrix(i))
        self.matrix_1 = np.array(self.pb.geo.GetMatrix(j))
        self.epipole_0 = NormalizePointInHomogeneousCoords(np.dot(self.matrix_0,self.source_1))
        self.epipole_1 = NormalizePointInHomogeneousCoords(np.dot(self.matrix_1,self.source_0))
        
        # Compute distance to epipoles
        self.dist_u0_e0 = np.abs(self.pb.u-self.epipole_0[0])
        self.dist_u1_e1 = np.abs(self.pb.u-self.epipole_1[0])
        
        # Weight projections 
        proj0 = self.pb.proj[i,:,:]
        proj1 = self.pb.proj[j,:,:]
        self.weighted_proj_0 = proj0 * self.pb.cos_weights / self.dist_u0_e0
        self.weighted_proj_1 = proj1 * self.pb.cos_weights / self.dist_u1_e1
        
        ## Compute outer weight
        # Compute normal to detectors
        self.normal_0 = np.array(self.pb.geo.GetRotationMatrix(i))[2,:3]
        self.normal_1 = np.array(self.pb.geo.GetRotationMatrix(j))[2,:3]
        # Compute baseline unit vector
        self.baseline = NormalizeVector((self.source_1[:3]-self.source_0[:3]))#/np.linalg.norm(self.source_1[:3]-self.source_0[:3])
        # Compute outer weight
        self.outer_weight_0 = 1./np.abs(np.dot(self.baseline,self.normal_0))
        self.outer_weight_1 = 1./np.abs(np.dot(self.baseline,self.normal_1))
        
        self.l0 = self.outer_weight_0 * self.weighted_proj_0.sum()*self.pb.spu
        self.l1 = self.outer_weight_1 * self.weighted_proj_1.sum()*self.pb.spu
        
        return self.l0,self.l1


# Etude fonction de cout
def loss(x,y,losstype='squarediff') :
    if x.shape != y.shape :
        raise ValueError('Shape mismatch !')
    if losstype == 'squarediff' :
        l = 0.5*(x-y)**2
        grad = np.stack([x-y,y-x],axis=0)
    elif losstype == 'absdiff' :
        l = np.abs(x-y)
        grad = np.stack([np.zeros_like(x),np.zeros_like(y)],axis=0)
    elif losstype == "diff" :
        l = x-y
        grad = np.stack([np.ones_like(x),-np.ones_like(y)],axis=0)
    return l,grad

def loss_from_array(array,geo,pairs,template,method='diff') :
    """
    Attend un array (N,1,P), N nb projs, P nb pixels per proj.
    """
    sino = itk.GetImageFromArray(array)
    sino.CopyInformation(template)
    pb_fb = fbdccProblem(sino, geo)
    dcc_fb = pb_fb.evaluate_dcc(pairs) # This is the time consuing one !
    l,g = loss(dcc_fb[:,0],dcc_fb[:,1],losstype=method)
    return np.dot(l.T,l)




#######################################################################
# ITK
#######################################################################

def ReadImageFile(file) :
    """ Read an image file and returns an ITK image object"""
    return itk.ImageFileReader(FileName=file)


def ExtractSlice(stack,num,axis='z') :
    """
        Returns one slice from a 3D volumes (or from a stack of images )
        Default dir is z. Assume a stack of projections. z dir corresponds to rotation angle.
    """
    ar = itk.GetArrayFromImage(stack)
    spacing = np.array(stack.GetSpacing())
    if axis == 'z' :
        newar = ar[num,:,:]
        newspacing = np.array([spacing[0],spacing[1]])
    elif axis == 'y' :
        newar = ar[:,num,:]
        newspacing = np.array([spacing[0],spacing[2]])
    elif axis == 'x' :
        newar = ar[:,:,num]
        newspacing = np.array([spacing[1],spacing[2]])
    newim = itk.GetImageFromArray(newar)
    newsize = np.array(newim.GetLargestPossibleRegion().GetSize())
    newim.SetSpacing(newspacing)
    newim.SetOrigin(-0.5*newspacing*(newsize-1.0))
    return newim


def WriteImage(image,file) :
    im_dim = image.GetImageDimension()
    writer = itk.ImageFileWriter[itk.Image[itk.F,im_dim]].New()
    writer.SetFileName(file)
    writer.SetInput(image)
    writer.Write()
    return 1

def WriteImageFromArray(imarray,filename,spacing) :
    im = itk.GetImageFromArray(np.array(imarray,dtype=np.float32))
    im.SetSpacing(spacing)
    im.SetOrigin(-0.5*np.array(spacing)*(np.array(im.GetLargestPossibleRegion().GetSize())-1))
    #im.SetOrigin([-0.5*spacing*(x-1) for x in  im.GetLargestPossibleRegion().GetSize()])
    WriteImage(im,filename)
    return 1

def ConvertArrayToItkImage(array,spacing) :
    im = itk.GetImageFromArray(array)
    im.SetSpacing(spacing)
    im.SetOrigin(-0.5*spacing*(np.array(im.GetLargestPossibleRegion().GetSize())-1))
    return im


#######################################################################
# Plotting functions
#######################################################################

def PlotProfile(x0,y0,x1,y1,im):# Make a line with "num" points...
    #x0, y0 = 5, 4.5 # These are in _pixel_ coordinates!!
    #x1, y1 = 60, 75
    num = int(np.sqrt((x1-x0)**2+(y1-y0)**2))
    x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)

    # Extract the values along the line, using cubic interpolation
    zi = scipy.ndimage.map_coordinates(im.T, np.vstack((x,y)))

    #-- Plot...
    fig, axes = plt.subplots(nrows=2)
    axes[0].imshow(im)
    axes[0].plot([x0, x1], [y0, y1], 'ro-')
    axes[0].set_axis_off()

    axes[1].plot(zi)
    plt.tight_layout()
    plt.show()


def PlotProfileManual(im):
    """ Plot profile from (x0,y0) to (x1,y1) which are selected manually
        by the user on the image.
        Profile is plotted in a subplot below the image and the line is drawn over the image.
    """
    fig , axes  = plt.subplots(nrows=2)
    axes[0].imshow(im)
    axes[0].set_axis_off()
    plt.tight_layout()
    plt.show()
    while plt.fignum_exists(fig.number) :
        points = fig.ginput(2)
        x0 , y0 = points[0]
        x1 , y1 = points[1]
        num = int(np.sqrt((x1-x0)**2+(y1-y0)**2))
        axes[0].plot([x0, x1], [y0, y1], 'ro-', markersize=3)
        x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
    
        # Extract the values along the line, using cubic interpolation
        zi = scipy.ndimage.map_coordinates(im.T, np.vstack((x,y)))
    
        #-- Plot...
        axes[1].plot(zi)
        
    return 1
    
    
    
def ShowImage(imseq,p=1,q=1,cmap='gray') :
    if (p==1 and q==1) :
        fig , ax = plt.subplots()
        ax.imshow(imseq[0],cmap=cmap)
        ax.set_axis_off()
    elif (p==1 or q==1) :
        fig , ax = plt.subplots(p,q)
        for i in range(len(imseq)) :
            ax[i].imshow(imseq[i],cmap=cmap)
            ax[i].set_axis_off()
    else :
        fig,ax = plt.subplots(p,q)
        idx = 0
        for i in range(len(imseq)) :
            idx_i = idx//q
            idx_j = np.mod(idx,q)
            ax[idx_i,idx_j].imshow(imseq[i],cmap=cmap)
            ax[idx_i,idx_j].set_axis_off()
            idx+=1
    plt.tight_layout()
    return ax


def PlotRadonLines(theta,r,ax,colors,linestyle,imshape) :
    """ Plot the line in Radon coordinates (r,theta) on ax"""
    print(theta.shape,r.shape)
    #fig,ax = plt.subplots()
    #ax.imshow(im)
    #ax.set_axis_off()
    for i in range(theta.shape[0])  :
        y0 = (r[i]-0*np.cos(theta[i]))/np.sin(theta[i])
        y1 = (r[i]-imshape[1]*np.cos(theta[i]))/np.sin(theta[i])
        ax.plot((0,imshape[1]),(y0,y1),color=colors[i],linestyle=linestyle,linewidth=1)
    ax.set_xlim(0,imshape[1])
    ax.set_ylim(imshape[0],0)
    plt.tight_layout()

def Plot3DSurface(X,Y,Z) :
    """Plot of a 3D surface"""
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter


    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    # Customize the z axis.
    #ax.set_zlim(-1.01, 1.01)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()
    
#######################################################################
# Divers
#######################################################################

# Recuperation de la matrice de projection
def GetMatrixFromGeo(geo,idx) :
    mat = geo.GetMatrix(idx)
    tmp = mat.GetVnlMatrix().as_matrix()
    return itk.GetArrayFromVnlMatrix(tmp)

def GetProjectionToFixedCoordsFromGeo(geo,idx) :
    mat = geo.GetProjectionCoordinatesToFixedSystemMatrix(idx)
    tmp = mat.GetVnlMatrix().as_matrix()
    return itk.GetArrayFromVnlMatrix(tmp)



def NormalizePointInHomogeneousCoords(x) :
    if float(x[-1]) == 0.0 :
        return x
    else :
        return x/(float(x[-1]))


def DecomposeProjectionMatrix(P) :
    """ Decompose the 3x4 matrix P in KR[I|-S], according to Section 1.2.3 of Jerome Lesaint PhD thesis."""
    A = P[:,:3]
    s = np.zeros(4)
    s[:3] = (-1.0)*np.dot(np.linalg.inv(A),P[:,3])
    s[3] = 1.0
    u0 = np.dot(A[0,:],A[2,:])
    v0 = np.dot(A[1,:],A[2,:])
    f = np.sqrt(np.dot(A[0,:],A[0,:])-u0**2)
    K = np.array([[-f,0,u0],[0,-f,v0],[0,0,1]])
    R = np.dot(np.linalg.inv(K),A)
    return s,K,R


def Rebin(arr,newshape) :
    shape = (newshape[0],arr.shape[0]//newshape[0],newshape[1],arr.shape[1]//newshape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def get_tps_nbmes(filename) :
    f = open(filename+'.spr','r')
    lines = f.readlines()
    f.close()
    return np.float32(lines[4][:-1]),np.float32(lines[6][:-1])

def get_tps_nbmes_acq(filename) :
    f = open(filename+'.spr','r')
    lines = f.readlines()
    f.close()
    return np.float32(lines[4][:-1]),np.float32(lines[6][:-2])

def htop() :
    conn = wmi.WMI()
    for process in conn.Win32_Process():
        print("ID: {0} \t HandleCount: {1}\t ProcessName: {2}".format(\
              process.ProcessId, process.HandleCount, process.Name)\
              )


def generate_drm(n,e_min,e_max,filename) :
    """ Generates a input file for sindbad. The DRM is the identity matrix.
    n : the size of the spectrum in the .spe file
    e_min,e_max : min and max energy of the source spectrum.
    filename : the full path and filename to create the file
    
    NB : The DRM does not use the exact same energy values of the spectrum,
    so that interpolation happens (especially at K-edges).
    """
    I = np.identity(n)
    e = np.linspace(e_min*0.9,e_max*1.1,n)
    
    f = open(filename,'w')
    
    f.write(f'{n+1}\n')
    f.write(f'{n+1}\n')
    f.write("{:0<6f}".format(0))
    for i in range(n) :
        f.write(" {:0<6f}".format(e[i]))
    f.write(' \n')
    
    for i in range(n) :
        f.write("{:0<6f}".format(e[i]))
        for j in range(n) :
            if i == j :
                f.write(" {:0<6f}".format(1))
            else :
                f.write(" {:0<6f}".format(0))
        f.write(' \n')
    
    
def CompleteMissingPixels(projs) :
    """ Hypothese : last dimension of projs is pixel number"""
    nb_pixels = projs.shape[-1]
    new_shape = list(projs.shape)
    nb_blocs = int(nb_pixels/128)
    new_shape[-1] = int(nb_pixels+2*(nb_blocs-1))
    sino_i = np.zeros(new_shape,dtype=np.float32)
    
    for i in range(nb_blocs) :
        sino_i[...,130*i:130*i+128] = projs[...,128*i:128*i+128]
    
    for idx in range(nb_blocs-1) :
        i = idx+1
        gap = (sino_i[...,130*i]-sino_i[...,130*i-3])
        sino_i[...,130*i-2] = sino_i[...,130*i-3]+gap/3
        sino_i[...,130*i-1] = sino_i[...,130*i-3]+gap*2/3
    return sino_i


def RecuperationCoeffAttenuationLineaireFromSindbad(filename,energies) :
    """ Fonction qui lit le fichier filename (chemin+nom de fichier)
    et recupere un echantillonnage du coefficient d'attenuation 
    aux valeurs d'energies passees
    ATTENTION au format du chemin passe selon l'OS.
    """
    with open(filename,'r') as f :
        lines = f.readlines()
        
    rho = np.float32(lines[6].split(' ')[0])
    idx = 0
    for i in range(len(lines)) :
        if lines[i][:9] == "# energie" :
            idx = i+1
    if idx == 0 :
        raise ValueError('toto')
    nb_lines = len(lines)-2-idx+1 #2 lignes "fake" en fin de fichier
    A = np.zeros([nb_lines,2])
    for i in range(nb_lines) :
        x = lines[idx+i].split('\t')
        A[i,0] = np.float32(x[0])
        A[i,1] = np.float32(x[1]) # Le coefficient d'att massique tau (en cm^2/g)
    
    # Gestion des K-edge
    ind = np.where(A[1:,0] == A[:-1,0])
    for idx in range(len(ind)) :
        A[idx+1,0] = A[idx,0]+0.0001
    
    # Interpolation aux energies souhaitees, en log/log
    f = scipy.interpolate.interp1d(np.log(A[:,0]),np.log(A[:,1]))
    mu = rho*np.exp(f(np.log(energies))) # en multipliant par rho, on obtient le coeff d'att lin (en cm^-1)
    return mu


        

########################
# Phase Contraast functions
########################


def derivativesByOpticalflow(intensityImage,derivative,pixsize=1,sig_scale=0):

    epsilon=np.finfo(float).eps
    Nim, Nx, Ny = derivative.shape #Image size
    dImX=np.zeros(((Nim,Nx,Ny)))
    dImY=np.zeros(((Nim,Nx,Ny)))
    
    for i in range(Nim):
        # fourier transfomm of the derivative and shift low frequencies to the centre
        ftdI = np.fft.fftshift(np.fft.fft2(derivative[i])) #Fourier transform of the derivative
        # calculate frequencies
        dqx = 2 * np.pi / (Nx)
        dqy = 2 * np.pi / (Ny)
    
        Qx, Qy = np.meshgrid((np.arange(0, Ny) - np.floor(Ny / 2) - 1) * dqy, (np.arange(0, Nx) - np.floor(Nx / 2) - 1) * dqx) #frequency ranges of the images in fqcy space
    
        #building filters
        sigmaX = dqx / 1. * np.power(sig_scale,2)
        sigmaY = dqy / 1. * np.power(sig_scale,2)
        #sigmaX=sig_scale
        #sigmaY = sig_scale
    
        g = np.exp(-(((Qx)**2) / 2. / sigmaX + ((Qy)**2) / 2. / sigmaY))
        #g = np.exp(-(((np.power(Qx, 2)) / 2) / sigmaX + ((np.power(Qy, 2)) / 2) / sigmaY))
        beta = 1 - g;
    
        # fourier filters
        ftfiltX = (1j * Qx / ((Qx**2 + Qy**2))*beta)
        ftfiltX[np.isnan(ftfiltX)] = 0
        ftfiltX[ftfiltX==0]=epsilon
    
        ftfiltY = (1j* Qy/ ((Qx**2 + Qy**2))*beta)
        ftfiltY[np.isnan(ftfiltY)] = 0
        ftfiltX[ftfiltY==0] = epsilon
    
        # output calculation
        dImX[i] = 1. / intensityImage[i] * np.fft.ifft2(np.fft.ifftshift(ftfiltX * ftdI)) #Displacement field
        dImY[i] = 1. / intensityImage[i] * np.fft.ifft2(np.fft.ifftshift(ftfiltY * ftdI))
    
    dX=np.median(dImX.real, axis=0)
    dY=np.median(dImY.real, axis=0)

    return dX, dY
    