import itk
from itk import RTK as rtk
import numpy as np
import matplotlib.pyplot as plt


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

def RecupParam(geo,idx) :
    """
        Extract the nine RTK geometric parameters of projection number idx in the geom geometry.
    """
    sid = geo.GetSourceToIsocenterDistances()[idx]
    sdd = geo.GetSourceToDetectorDistances()[idx]
    ga = geo.GetGantryAngles()[idx]
    dx = geo.GetProjectionOffsetsX()[idx]
    dy = geo.GetProjectionOffsetsY()[idx]
    oa = geo.GetOutOfPlaneAngles()[idx]
    ia = geo.GetInPlaneAngles()[idx]
    sx = geo.GetSourceOffsetsX()[idx]
    sy = geo.GetSourceOffsetsY()[idx]
    return sid,sdd,ga,dx,dy,oa,ia,sx,sy
