import itk
from itk import RTK as srtk
import numpy as np


def ReadGeometryFile(file) :
    """ Read a RTK geometry file and returns an RTK geometry object"""
    reader = srtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
    reader.SetFilename(file)
    reader.GenerateOutputInformation()
    return reader.GetOutputObject()

def ReadImageFile(file) :
    """ Read an image file and returns an ITK image object"""
    return itk.ImageFileReader(FileName=file)

def SimulateSheppLoganPhantom(geom,image_size,image_spacing,phantomscale) :
    """ Returns a stack of simulated projections of a Shepp Logan phantom.
    image_size : size of the detector ([n,m])
    image_spacing : size of pixel (float). Assumes isotropic pixel.
    pahntomscale : size of the phantom (float). See RTK doc.
    
    """
    ImageType = itk.Image[itk.F, 3]
    const = srtk.ConstantImageSource[ImageType].New()
    const.SetConstant(0.)
    const.SetSpacing([image_spacing,image_spacing,image_spacing])
    const.SetSize([image_size[0],image_size[1],len(geom.GetGantryAngles())])
    const.SetOrigin(-0.5*image_spacing*(np.array(const.GetSize())-1))
    sl = srtk.SheppLoganPhantomFilter[ImageType,ImageType].New()
    sl.SetInput(const.GetOutput())
    sl.SetGeometry(geom)
    sl.SetPhantomScale(phantomscale)
    sl.Update()
    return sl.GetOutput()


def ExtractSlice(stack,num,axis='z') :
    """
        Returns one slice from a 3D volumes (or from a stack of images )
        Default dir is z. Assume a stack of projections. z dir corresponds to rotation angle.
    """
    size = np.array(stack.GetLargestPossibleRegion().GetSize())
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


def get_2D_line_from_two_2D_points(A,B):
    norm = np.sqrt(((B-A)**2).sum())
    vec = (B-A)/norm
    n = np.array([-vec[1], vec[0]]) # Normal to the line
    return np.array([*n, -np.dot(n,A)])


def DecomposeProjectionMatrix(P) :
    """ Decompose the 3x4 matrix P in 9 geometric parameters, according to Section 1.2.3 of Jérome Lesaint PhD thesis."""
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

def Normalize3DPoint(x) :
    return x/(float(x[-1]))
    
def Normalize2DPoint(x) :
    return x/(float(x[-1]))

def SkewSymMatrixFromVector(x):
    return np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])


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

def GetMatrixFromParams(sid,sdd,ga,dx,dy,oa,ia,sx,sy) :
    """ 
        Compute the 3x4 projection matrix from the 9 RTK geometric parameters. See RTK geometry doc.
    """
    matia = np.array([[np.cos(-ia),-np.sin(-ia),0,0],[np.sin(-ia),np.cos(-ia),0,0],[0,0,1,0],[0,0,0,1]])
    matoa = np.array([[1,0,0,0],[0,np.cos(-oa),-np.sin(-oa),0],[0,np.sin(-oa),np.cos(-oa),0],[0,0,0,1]])
    matga = np.array([[np.cos(-ga),0,np.sin(-ga),0],[0,1,0,0],[-np.sin(-ga),0,np.cos(-ga),0],[0,0,0,1]])
    rotmat = np.dot(matia,np.dot(matoa,matga))
    tmp1 = np.identity(3)
    tmp1[0,2] = sx-dx
    tmp1[1,2] = sy-dy
    tmp2 = np.array([[-sdd,0,0,0],[0,-sdd,0,0],[0,0,1,-sid]])
    tmp3 = np.identity(4)
    tmp3[0,3] = -sx
    tmp3[1,3] = -sy
    return np.dot(tmp1,np.dot(tmp2,np.dot(tmp3,rotmat)))

def GetDetectorCoordinatesToFixedSystemMatrix(sid,sdd,ga,dx,dy,oa,ia,sx,sy) :
    mat = np.identity(4)
    mat[0,3] = dx
    mat[1,3] = dy
    mat[2,3] = sid-sdd
    mat[2,2]= 0.0
    matia = np.array([[np.cos(-ia),-np.sin(-ia),0,0],[np.sin(-ia),np.cos(-ia),0,0],[0,0,1,0],[0,0,0,1]])
    matoa = np.array([[1,0,0,0],[0,np.cos(-oa),-np.sin(-oa),0],[0,np.sin(-oa),np.cos(-oa),0],[0,0,0,1]])
    matga = np.array([[np.cos(-ga),0,np.sin(-ga),0],[0,1,0,0],[-np.sin(-ga),0,np.cos(-ga),0],[0,0,0,1]])
    rotmat = np.dot(matia,np.dot(matoa,matga))
    return np.dot(np.linalg.inv(rotmat),mat)

def Interpolate(x,y,xvalue) :
    """ Interpolate the signal y in xvalue. Return yvalue.
        Assume increasing x values
    """
    tmp = x-xvalue
    tmp[tmp<0] = 1e5
    idx = np.argmin(tmp)-1
    if idx == x.shape[0]-1 : 
        return y[-1]
    else:
        alpha = (xvalue-x[idx])/(x[idx+1]-x[idx])
        return (1-alpha)*y[idx]+alpha*y[idx+1]
