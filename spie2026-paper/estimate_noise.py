import itk
from itk import RTK as rtk
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from dcc4cbct.fbcc import FBProblem, EpipolarPair
from dcc4cbct.misc import RecupParam
from dcc4cbct.rtk_utils import AddNoiseToProjections, ReadGeometryFile


noisy_proj_files = ["stack_1.0E+02.mha","stack_1.0E+03.mha","stack_1.0E+04.mha"]


stacks = [itk.imread(filename) for filename in noisy_proj_files]
geo = ReadGeometryFile("geo.xml")
geo1 = ReadGeometryFile("geo1.xml")
geo2 = ReadGeometryFile("geo2.xml")

def create_new_geometry(displacement_vector_x,displacement_vector_z,geo1, geo2):
    n_projs1 = len(geo1.GetGantryAngles())
    n_projs2 = len(geo2.GetGantryAngles())
    new_geo = rtk.ThreeDCircularProjectionGeometry.New()
    for i in range(n_projs1):
        new_geo.AddProjection(geo1.GetMatrix(i))
    for i in range(n_projs2):
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
        new_geo.AddProjectionInRadians(new_sid, sdd, ga, new_dx, new_dy, oa, ia, new_sx, new_sy)
        # Geowriter
    geowriter = rtk.ThreeDCircularProjectionGeometryXMLFileWriter.New()
    geowriter.SetFilename("optimized_geo.xml")
    geowriter.SetObject(new_geo)
    geowriter.WriteFile()

    return new_geo



# Cost functions
# Minimizing the cost function
from scipy.optimize import minimize

# Define both cost functions (fanbeam and Grangeat)

def create_new_pair(pair,x,z):
    g0 = pair.g0
    g1 = pair.g1
    p0 = pair.p0
    p1 = pair.p1
    sid,sdd,ga,dx,dy,oa,ia,sx,sy = RecupParam(g1,0)
    rotmat = np.array(g1.GetRotationMatrix(0))
    delta_dx, delta_dy, delta_sid, _ = np.dot(rotmat, np.array([x, 0, z, 0]))
    g1bis = rtk.ThreeDCircularProjectionGeometry.New()
    new_sid = sid + delta_sid
    new_dx = dx + delta_dx
    new_dy = dy + delta_dy
    new_sx = sx + delta_dx
    new_sy = sy + delta_dy
    g1bis = rtk.ThreeDCircularProjectionGeometry.New()
    g1bis.AddProjectionInRadians(new_sid, sdd, ga, new_dx, new_dy, oa, ia, new_sx, new_sy)
    return EpipolarPair(g0, g1bis, p0, p1)


def cost_fun_fb(params, pb, verbose):
    x = params[0]
    z = params[1]
    fb = 0.
    for pair in pb.pairs:
        pair = create_new_pair(pair,x,z)
        #g0 = pair.g0
        #g1 = pair.g1
        #p0 = pair.p0
        #p1 = pair.p1
        #sid,sdd,ga,dx,dy,oa,ia,sx,sy = RecupParam(g1,0)
        #rotmat = np.array(g1.GetRotationMatrix(0))
        #delta_dx, delta_dy, delta_sid, _ = np.dot(rotmat, np.array([x, 0, z, 0]))
        #g1bis = rtk.ThreeDCircularProjectionGeometry.New()
        #new_sid = sid + delta_sid
        #new_dx = dx + delta_dx
        #new_dy = dy + delta_dy
        #new_sx = sx + delta_dx
        #new_sy = sy + delta_dy
        #g1bis = rtk.ThreeDCircularProjectionGeometry.New()
        #g1bis.AddProjectionInRadians(new_sid, sdd, ga, new_dx, new_dy, oa, ia, new_sx, new_sy)
        #pair = EpipolarPair(g0, g1bis, p0, p1)
        fb += pair.compute_fanbeam_consistency()
    if verbose:
        print(f"Eval at ({x:.4f},{z:.4f}) \t:\t {fb:.4f}")
    return fb

def cost_fun_gr(params, pb, verbose):
    x = params[0]
    z = params[1]
    gr = 0.
    for pair in pb.pairs:
        g0 = pair.g0
        g1 = pair.g1
        p0 = pair.p0
        p1 = pair.p1
        sid,sdd,ga,dx,dy,oa,ia,sx,sy = RecupParam(g1,0)
        rotmat = np.array(g1.GetRotationMatrix(0))
        delta_dx, delta_dy, delta_sid, _ = np.dot(rotmat, np.array([x, 0, z, 0]))
        g1bis = rtk.ThreeDCircularProjectionGeometry.New()
        new_sid = sid + delta_sid
        new_dx = dx + delta_dx
        new_dy = dy + delta_dy
        new_sx = sx + delta_dx
        new_sy = sy + delta_dy
        g1bis = rtk.ThreeDCircularProjectionGeometry.New()
        g1bis.AddProjectionInRadians(new_sid, sdd, ga, new_dx, new_dy, oa, ia, new_sx, new_sy)
        pair = EpipolarPair(g0, g1bis, p0, p1)
        gr += pair.compute_grangeat_consistency()
    if verbose:
        print(f"Eval at ({x:.4f},{z:.4f}) \t:\t {gr:.4f}")
    return gr


def build_pairs(strategy, plot=False):
    i_idxs = []
    j_idxs = []
    if strategy == "two_pairs":
        i_idxs = (30,80)
        j_idxs = (100,150)
    elif strategy == "ten_pairs":
        i_idxs = (20,40,20,40, 80,80, 60,60)
        j_idxs = (100,100,120,120,140,160, 140,160)
    elif strategy == "max":
        for i in range(0,90,10):
            for j in range(90,180,10):
                if (j + i < 175 or j + i > 185) and j - i < 120 and j - i > 50:
                    i_idxs.append(i)
                    j_idxs.append(j)
    pairs = list(zip(i_idxs,j_idxs))
    if plot:
        fig,ax = plt.subplots()
        im = ax.scatter(j_idxs,i_idxs)
        ax.set_ylim((90,0))
        ax.set_xlim((90,180))
        ax.set_aspect("equal","box")
    return pairs


strategy = "two_pairs"
sols = []
for stack in stacks:
    pb = FBProblem(geo, stack)
    pairs = build_pairs(strategy)
    for p in pairs:
        pb.add_pair_to_problem(p)

    verbose = True
    sol = minimize(cost_fun_fb,(10,270),(pb,verbose),method="Powell")
    sols.append(sol)
    print(sol)


strategy = "ten_pairs"
cost_funs = {}
strategies = ["two_pairs","ten_pairs"]#,"max"]
for strategy in strategies:
    i = 0
    for stack in stacks:
        cost = []
        pb = FBProblem(geo, stack)
        pairs = build_pairs(strategy)
        for p in pairs:
            pb.add_pair_to_problem(p)

        for tz in np.linspace(260,300):
            cost.append(cost_fun_fb([0,tz],pb,verbose))
            
        cost_funs.update({f"{strategy}_{i}":np.array(cost)})
        i+=1




strategy = "ten_pairs"
newpairs = []
pb = FBProblem(geo, stack)
pairs = build_pairs(strategy)
for p in pairs:
    pb.add_pair_to_problem(p)

for p in pb.pairs:
    newpair = create_new_pair(p,0,290)
    newpair.compute_line_integrals()
    newpairs.append(newpair)

for p in newpairs:
    p.analyze_pair()
for i,p in enumerate(newpairs):
    print(f"{i}: {p.compute_fanbeam_consistency()}")
    
build_pairs('ten_pairs')
