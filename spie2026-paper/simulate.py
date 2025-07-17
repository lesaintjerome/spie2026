import itk
from itk import RTK as rtk
import json
import os
import numpy as np
import matplotlib.pyplot as plt

try: 
    from dcc4cbct.rtk_utils import AddNoiseToProjections, ReadGeometryFile, RecupParam
except ImportError:
    os.system("pip install ../dcc4cbct")
    from dcc4cbct.rtk_utils import AddNoiseToProjections, ReadGeometryFile, RecupParam
#%matplotlib osx


def main():
    with open("sf_config.json","r") as f:
        config = json.load(f)

    displacement_vector_x = config["phantom"]["translation_x_1"] - config["phantom"]["translation_x_2"]
    displacement_vector_z = config["phantom"]["translation_z_1"] - config["phantom"]["translation_z_2"]


    # Geowriter
    geowriter = rtk.ThreeDCircularProjectionGeometryXMLFileWriter.New()

    # Full geom
    geo = rtk.ThreeDCircularProjectionGeometry.New()
    corrected_geo = rtk.ThreeDCircularProjectionGeometry.New()

    # First half scan
    geo1 = rtk.ThreeDCircularProjectionGeometry.New()
    for i in range(config["geometry"]["nprojs_1"]):
        angle = config["geometry"]["first_angle_1"] + i * config["geometry"]["arc_1"] / (config["geometry"]["nprojs_1"])
        geo1.AddProjection(config["geometry"]["sid"], config["geometry"]["sdd"], angle)
        geo.AddProjection(config["geometry"]["sid"], config["geometry"]["sdd"], angle)
        corrected_geo.AddProjection(config["geometry"]["sid"], config["geometry"]["sdd"], angle)
    geowriter.SetFilename("geo1.xml")
    geowriter.SetObject(geo1)
    geowriter.WriteFile()

    # Second half scan
    geo2 = rtk.ThreeDCircularProjectionGeometry.New()
    for i in range(config["geometry"]["nprojs_2"]):
        angle = config["geometry"]["first_angle_2"] + i * config["geometry"]["arc_2"] / (config["geometry"]["nprojs_2"])
        geo2.AddProjection(config["geometry"]["sid"], config["geometry"]["sdd"], angle) 
        geo.AddProjection(config["geometry"]["sid"], config["geometry"]["sdd"], angle)

        sid,sdd,ga,dx,dy,oa,ia,sx,sy = RecupParam(geo2,i)
        rotmat = np.array(geo2.GetRotationMatrix(i))
        delta_dx, delta_dy, delta_sid, _ = np.dot(
            rotmat,
            np.array([displacement_vector_x, 0, displacement_vector_z, 0])
        )
        new_sid = sid + delta_sid
        new_dx = dx + delta_dx
        new_dy = dy + delta_dy
        new_sx = sx + delta_dx
        new_sy = sy + delta_dy
        corrected_geo.AddProjectionInRadians(new_sid, sdd, ga, new_dx, new_dy, oa, ia, new_sx, new_sy)

    geowriter.SetFilename("geo2.xml")
    geowriter.SetObject(geo2)
    geowriter.WriteFile()
    geowriter.SetFilename("geo.xml")
    geowriter.SetObject(geo)
    geowriter.WriteFile()
    geowriter.SetFilename("corrected_geo.xml")
    geowriter.SetObject(corrected_geo)
    geowriter.WriteFile()
    


    imtype = itk.Image[itk.F, 3]

    # Constantt vol 1
    cst1 = rtk.ConstantImageSource[imtype].New()
    cst1.SetConstant(0.0)
    cst1.SetSize([config["detector"]["Nu"], config["detector"]["Nv"], config["geometry"]["nprojs_1"]])
    cst1.SetSpacing([config["detector"]["spu"],] * 3)
    cst1.SetOrigin(-0.5*config["detector"]["spu"] * np.array((config["detector"]["Nu"]-1, config["detector"]["Nv"]-1, config["geometry"]["nprojs_1"]-1)))

    fwd1 = rtk.SheppLoganPhantomFilter[imtype,imtype].New()
    fwd1.SetInput(cst1)
    fwd1.SetGeometry(geo1)
    fwd1.SetOriginOffset(np.array([config["phantom"]["translation_x_1"], 0.0, config["phantom"]["translation_z_1"]])/config["phantom"]["scale"])
    fwd1.SetPhantomScale(config["phantom"]["scale"])
    fwd1.Update()
    stack1 = fwd1.GetOutput()
    itk.imwrite(stack1, "phantom1.mha")

    # Constantt vol 2
    cst2 = rtk.ConstantImageSource[imtype].New()
    cst2.SetConstant(0.0)
    cst2.SetSize([config["detector"]["Nu"], config["detector"]["Nv"], config["geometry"]["nprojs_2"]])
    cst2.SetSpacing([config["detector"]["spu"],] * 3)
    cst2.SetOrigin(-0.5*config["detector"]["spu"] * np.array((config["detector"]["Nu"]-1, config["detector"]["Nv"]-1, config["geometry"]["nprojs_2"]-1)))

    fwd2 = rtk.SheppLoganPhantomFilter[imtype,imtype].New()
    fwd2.SetGeometry(geo2)
    fwd2.SetInput(cst2)
    fwd2.SetOriginOffset(np.array([config["phantom"]["translation_x_2"], 0.0, config["phantom"]["translation_z_2"]])/config["phantom"]["scale"])
    fwd2.SetPhantomScale(config["phantom"]["scale"])
    fwd2.Update()
    stack2 = fwd2.GetOutput()
    itk.imwrite(stack2, "phantom2.mha")


    # Merge stacks
    stack = itk.GetImageFromArray(np.concatenate((stack1, stack2), axis=0))
    stack.CopyInformation(stack2)
    if config["projections"]["add_noise"]:
        noise_level = config["projections"]["noise_level"]
        stack = AddNoiseToProjections(stack, noise_level, 0.01879) # 0.01879: mu(water)@75keV
    itk.imwrite(stack, f"stack_{noise_level:.1E}.mha")


if __name__ == "__main__":
    main()