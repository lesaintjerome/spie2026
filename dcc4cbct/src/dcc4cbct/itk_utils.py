import itk
import numpy as np


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
