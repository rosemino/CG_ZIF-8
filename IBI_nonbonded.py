import numpy as np
import math as math

# ------------------------------------------------------------------------------------
# In this example, we are working with the update for the non-bonded 1-1 potential in mapping 1.
# The RDF is printed by LAMMPS and contains 4 columns (see the compute rdf command to understand
# more).
lines_rdf = 1500

columns_rdf_lammps = 4

rdf_lammps=[]
ofi = open("rdf_lammps.dat", 'r')
dump = ofi.readline()
dump = ofi.readline()
dump = ofi.readline()
dump = ofi.readline()
for it_1 in range(0,lines_rdf):
        dump = ofi.readline()
        #dump = dump[0:32]
        for e,it_2 in zip(dump.split(' '), range(columns_rdf_lammps)):
            rdf_lammps.append(float(e))
rdf_lammps = np.array(rdf_lammps,float)
rdf_lammps = rdf_lammps.reshape(lines_rdf,columns_rdf_lammps)

rdf_lammps = np.delete(rdf_lammps,3,1)
rdf_lammps = np.delete(rdf_lammps,0,1)

# ------------------------------------------------------------------------------------
# Read current potential for the 1-1 interaction.
lines_U = 1301

U_model_i=[]
ofi = open("CG1_CG1.pot", 'r')
dump = ofi.readline()
dump = ofi.readline()
dump = ofi.readline()
for it_1 in range(0,lines_U):
        dump = ofi.readline()
        #dump = dump[0:32]
        for e,it_2 in zip(dump.split('\t'), range(4)):
            U_model_i.append(float(e))
U_model_i = np.array(U_model_i,float)
U_model_i = U_model_i.reshape(lines_U,4)

U_model_i = np.delete(U_model_i,3,1)
U_model_i = np.delete(U_model_i,0,1)

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Read the target 1-1 RDF.
rdf_target=[]
ofi = open("CG1-CG1.dist.tgt", 'r')
dump = ofi.readline()
dump = ofi.readline()
dump = ofi.readline()
dump = ofi.readline()
dump = ofi.readline()
for it_1 in range(0,lines_rdf):
        dump = ofi.readline()
        #dump = dump[0:32]
        for e,it_2 in zip(dump.split(' '), range(2)):
            rdf_target.append(float(e))
rdf_target = np.array(rdf_target,float)
rdf_target = rdf_target.reshape(lines_rdf,2)

rdf_target[:,0] = rdf_target[:,0]*10

rdf_bin = 0.01000000000000000
# This is just an adjustment to have the x column be the center of the bin (was not necessary
# for the upcoming parts of the code).
for it_1 in range (0, len(rdf_target)):
    rdf_target[it_1,0] = it_1*rdf_bin + rdf_bin/2
        
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# This part is zero'ing any value below 1E-7 in the current distribution. Note that this is not
# the threshold used (nor mentioned in the thesis/paper). The zero'ing of values below the thresh-
# hold mentioned in the SM (ie 5e-3) comes afterwards.
for it_1 in range (0, lines_rdf):
    if rdf_target[it_1,1] <= 1E-7:
        rdf_target[it_1,1] = 0
    if rdf_lammps[it_1,1] <= 1E-7:
        rdf_lammps[it_1,1] = 0

# -----------------------------------------------------------------------------------------
# Part built to identify island non-zero values (i.e., ones that do not really correspond to
# a peak) and are surrounded by bins containing zero.
# To do this, I set a threshold where a peak is only considered to exist if the value of the
# distribution at some point gets to be higher than 0.1. Judging from the existing peaks in
# all the reference distributions I have, this is a rightful setup. And since the distributions
# given by BI potentials are usually much higher in amplitude than the target ones, setting up
# a minimum threshold is appropriate.
peak_found = 0
last_peak_line = 0
for it_1 in range (0, lines_rdf):
    # --------------------------------------------------------------------
    if peak_found == 1:
        if it_1 == last_peak_line + 1:
            peak_found = 0
        if it_1 < last_peak_line + 1:
            continue
    # -------------------------------------------------------------------
    if rdf_lammps[it_1,1] > 1E-7:
        flag = 0
        for it_2 in range (it_1, lines_rdf):
            if 1E-7 < rdf_lammps[it_2,1] < 0.1:
                continue
            if rdf_lammps[it_2,1] > 0.1:
                flag = 1
            if rdf_lammps[it_2,1] < 1E-7:
                last_peak_line = it_2
                peak_found = 1
                break
        if flag == 0:
            for it_2 in range (it_1, last_peak_line + 1):
                rdf_lammps[it_2,1] = 0.0

# NOTE: after doing IBI using my code, IF I realize there is something wrong with the result (i.e.,
# some problem for converging or something) and I suspect the problem might come from the part
# above that I used to delete island values, I can come back and modify it.
# -----------------------------------------------------------------------------------------
# Finally, taking care of possible changes in magnitude that happen in distribution values at
# x belonging to [5e-3,0.1].
# The 0'ing of all values smaller than 5e-3 also happens in this part of the code.

from scipy.interpolate import CubicSpline
import scipy.optimize 

# This is for the left-hand side of the onset region.
flag = 0
for it_1 in range (1, lines_rdf):
    if (rdf_lammps[it_1,1] > 1E-7) & (rdf_lammps[it_1-1,1] < 1E-7):
            x_array_interpolate = []
            y_array_interpolate = []
            flag = 1
            flag_2 = 0
            recalculate = []
            values_to_zero = []
            
    if flag == 1:
        if rdf_lammps[it_1,1] <= 5E-3:
            if flag_2 == 1:
                recalculate.append(it_1)
            if flag_2 == 0:
                values_to_zero.append(it_1)
        if 5E-3 < rdf_lammps[it_1,1] < 0.1:
            x_array_interpolate.append(rdf_lammps[it_1,0])
            y_array_interpolate.append(rdf_lammps[it_1,1])
            flag_2 = 1
        
        # I am assuming that once I reach values of this magnitude I am no longer at the
        # onset region. Again: if when doing IBI with this code I realize something is going
        # wrong, I can suspect such an criterium is not good.
        # NOTE: that it's not so arbitrary though: at points where the value of y are around 0.1,
        # there is no longer steep changes in magnitude or anything.
        if rdf_lammps[it_1,1] >= 0.1:
            if len(x_array_interpolate) < 6:
                term = 1
                while len(x_array_interpolate) != 6:
                    x_array_interpolate.append(rdf_lammps[it_1+term,0])
                    y_array_interpolate.append(rdf_lammps[it_1+term,1])
                    term = term + 1
                    print('WARNING: very few data for interpolation')
   
            my_interpolation = CubicSpline(x_array_interpolate, y_array_interpolate)
            for it_2 in reversed(range (0, len(recalculate))):
                given_line = recalculate[it_2]
                rdf_lammps[given_line,1] = my_interpolation(rdf_lammps[given_line,0])
                # This shouldnt be necessary: it is just in case the cubic spline interpolation
                # scheme gives me a value smaller than 5e-3. It is a "backup" methodology to try
                # to ensure monotonic behaviour: it takes 80% of the value of the right-hand side
                # neighboring bin. It is not certain to work, but it is one more attempt to en-
                # sure it. If I see that the setup IBI doesnt converge to potentials capable of
                # reproducing the structure, I can change something in an attempt to do so.
                if rdf_lammps[given_line,1] < 5e-3:
                    rdf_lammps[given_line,1] = 0.8*rdf_lammps[given_line+1,1]
                    
            for it_2 in range (0, len(values_to_zero)):
                given_line = values_to_zero[it_2]
                rdf_lammps[given_line,1] = 0
                    
            flag = 0
            flag_2 = 0

# Now the same thing but for the onset region of the right handside peaks.
# Note taht the possibility of the distribution starting from the middle of a peak when scanning 
# it from right to left is not considered. More specifically, in cases where it starts in the 
# middle of a peak it is only if it starts in the onset region that a pre-treatment would be
# necessary.
# Changing the code to consider this possibility and address it accordingly is possible if 
# desired. In the scope of the IBI performed in my work, this missing this part didnt reveal 
# to be a problem at all for converging to suitable potentials for reproducing structure. For
# what is worth it, none of the target distributions in hand indeed start (right to left) in 
# the middle of the onset region of a peak (for none of the three mappings).
flag = 0
for it_1 in reversed(range(0, lines_rdf - 1)):
    if ((rdf_lammps[it_1,1] > 1E-7) & (rdf_lammps[it_1+1,1] < 1E-7)):
            x_array_interpolate = []
            y_array_interpolate = []
            flag = 1
            flag_2 = 0
            recalculate = []
            values_to_zero = []
                           
    if flag == 1:
        if rdf_lammps[it_1,1] <= 5E-3:
            if flag_2 == 1:
                recalculate.append(it_1)
            if flag_2 == 0:
                values_to_zero.append(it_1)
        if 5E-3 < rdf_lammps[it_1,1] < 0.1:
            x_array_interpolate.append(rdf_lammps[it_1,0])
            y_array_interpolate.append(rdf_lammps[it_1,1])
            flag_2 = 1
        if rdf_lammps[it_1,1] >= 0.1:
            if len(x_array_interpolate) < 6:
                term = 1
                while len(x_array_interpolate) != 6:
                    x_array_interpolate.append(rdf_lammps[it_1-term,0])
                    y_array_interpolate.append(rdf_lammps[it_1-term,1])
                    term = term + 1
                    print('WARNING: very few data for interpolation')

            x_array_interpolate.reverse()
            y_array_interpolate.reverse()
            my_interpolation = CubicSpline(x_array_interpolate, y_array_interpolate)
            for it_2 in reversed(range (0, len(recalculate))):
                given_line = recalculate[it_2]
                rdf_lammps[given_line,1] = my_interpolation(rdf_lammps[given_line,0])
                # This shouldnt be necessary: it is just in case the cubic spline interpolation
                # scheme gives me a value smaller than 5e-3. It is a "backup" methodology to try
                # to ensure monotonic behaviour: it takes 80% of the value of the left-hand side
                # neighboring bin. It is not certain to work, but it is one more attempt to en-
                # sure it. If I see that the setup IBI doesnt converge to potentials capable of
                # reproducing the structure, I can change something in an attempt to do so.
                if rdf_lammps[given_line,1] < 5e-3:
                    rdf_lammps[given_line,1] = 0.8*rdf_lammps[given_line-1,1]
                    
            for it_2 in range (0, len(values_to_zero)):
                given_line = values_to_zero[it_2]
                rdf_lammps[given_line,1] = 0
            
            flag = 0
            flag_2 = 0

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Part where I do the boltzmann inverted of the distributions.
# In this context, values that should not exist (due to the logarithm of zero operation not
# existing), are instead represented by 0's in the arrays U_target and variable_1.
U_target = np.zeros((lines_rdf,2))
variable_1 = np.zeros((lines_rdf,2))
for it_1 in range (0, lines_rdf):
    if rdf_target[it_1,1] > 1.0e-7:
        U_target[it_1,1] = -2.49435*(0.2390057361)*math.log(rdf_target[it_1,1])
    if rdf_lammps[it_1,1] > 1.0e-7:
        variable_1[it_1,1] = -2.49435*(0.2390057361)*math.log(rdf_lammps[it_1,1])

# ----------------------------------------------------------------------------------------
# Part where I do the harmonic interpolation in order to derive data for the boltzmann inverted
# distributions in intervals where the values of the distributions were originally zero. Note that
# only intervals that are surrounded by two other for which original boltzmann inverted data exists
# are considered for this.
# It includes also a smoothening of the region where the data coming from the function fitted
# meets data coming from the boltzmann invertion (original data). This is done in a cubic spline
# based fashion.
def myHarmonic(x,a,b,c):
    return a*(x**2) + b*x + c

# Doing it for the Boltzmann inverted target distribution.
flag = 0
counter = 0
first_peak = 0
for it_1 in range (2, len(U_target)):
    if (U_target[it_1,1] != 0.0) & (U_target[it_1-1,1] == 0.0) & (U_target[it_1-2,1] == 0.0) & (first_peak == 0):
        first_peak = 1
        continue

    if (U_target[it_1,1] == 0.0) & (U_target[it_1-1,1] != 0.0) & (U_target[it_1-2,1] != 0.0):
        x_array_interpolate_lhs = []
        y_array_interpolate_lhs = []
        line_lhs = it_1
        for it_2 in range(it_1-8,it_1):
            x_array_interpolate_lhs.append(U_target[it_2,0])
            y_array_interpolate_lhs.append(U_target[it_2,1])
            
    if (U_target[it_1,1] != 0.0) & (U_target[it_1-1,1] == 0.0) & (U_target[it_1-2,1] == 0.0) & (first_peak != 0):
        x_array_interpolate_rhs = []
        y_array_interpolate_rhs = []
        counter = 0
        line_rhs = it_1
        flag = 1
    if flag == 1:
        if counter < 8:
            x_array_interpolate_rhs.append(U_target[it_1,0])
            y_array_interpolate_rhs.append(U_target[it_1,1])
            counter = counter + 1
        if counter == 8:
            x_array_interpolate = []
            y_array_interpolate = []
            for it_2 in range (0, len(x_array_interpolate_lhs)):
                x_array_interpolate.append(x_array_interpolate_lhs[it_2])
            for it_2 in range (0, len(x_array_interpolate_rhs)):
                x_array_interpolate.append(x_array_interpolate_rhs[it_2])
            for it_2 in range (0, len(y_array_interpolate_lhs)):
                y_array_interpolate.append(y_array_interpolate_lhs[it_2])
            for it_2 in range (0, len(y_array_interpolate_rhs)):
                y_array_interpolate.append(y_array_interpolate_rhs[it_2])
            params, cv = scipy.optimize.curve_fit(myHarmonic, x_array_interpolate, y_array_interpolate)
            for it_2 in range(line_lhs, line_rhs):
                U_target[it_2,1] = myHarmonic(U_target[it_2,0], params[0], params[1], params[2])
            flag = 0
            counter = 0
            
            # Now a smoothening of the joint region where there is original data and
            # the region of data coming from the harmonic interpolation.
            x_array_interpolate = []
            y_array_interpolate = []
            for it_2 in range (line_lhs-6, line_lhs-2):
                x_array_interpolate.append(U_target[it_2,0])
                y_array_interpolate.append(U_target[it_2,1])
            for it_2 in range (line_lhs+3, line_rhs):
                x_array_interpolate.append(U_target[it_2,0])
                y_array_interpolate.append(U_target[it_2,1])
            my_interpolation = CubicSpline(x_array_interpolate, y_array_interpolate)
            for it_2 in range (line_lhs-2, line_lhs+3):
                U_target[it_2,1] = my_interpolation(U_target[it_2,0])
            x_array_interpolate = []
            y_array_interpolate = []
            for it_2 in range (line_lhs, line_rhs-2):
                x_array_interpolate.append(U_target[it_2,0])
                y_array_interpolate.append(U_target[it_2,1])
            for it_2 in range (line_rhs+3, line_rhs+6):
                x_array_interpolate.append(U_target[it_2,0])
                y_array_interpolate.append(U_target[it_2,1])
            my_interpolation = CubicSpline(x_array_interpolate, y_array_interpolate)
            for it_2 in range (line_rhs-2, line_rhs+3):
                U_target[it_2,1] = my_interpolation(U_target[it_2,0])

# Doing the same for the Boltzmann inverted current distribution ("model distribution").
flag = 0
counter = 0
first_peak = 0
for it_1 in range (1, len(variable_1)):
    if (variable_1[it_1,1] != 0.0) & (variable_1[it_1-1,1] == 0.0) & (variable_1[it_1-2,1] == 0.0) & (first_peak == 0):
        first_peak = 1
        continue
    
    if (variable_1[it_1,1] == 0.0) & (variable_1[it_1-1,1] != 0.0) & (variable_1[it_1-2,1] != 0.0):
        x_array_interpolate_lhs = []
        y_array_interpolate_lhs = []
        line_lhs = it_1
        for it_2 in range(it_1-8,it_1):
            x_array_interpolate_lhs.append(variable_1[it_2,0])
            y_array_interpolate_lhs.append(variable_1[it_2,1])
            
    if (variable_1[it_1,1] != 0.0) & (variable_1[it_1-1,1] == 0.0) & (variable_1[it_1-2,1] == 0.0) & (first_peak != 0):
        x_array_interpolate_rhs = []
        y_array_interpolate_rhs = []
        counter = 0
        line_rhs = it_1
        flag = 1
    if flag == 1:
        if counter < 8:
            x_array_interpolate_rhs.append(variable_1[it_1,0])
            y_array_interpolate_rhs.append(variable_1[it_1,1])
            counter = counter + 1
        if counter == 8:
            x_array_interpolate = []
            y_array_interpolate = []
            for it_2 in range (0, len(x_array_interpolate_lhs)):
                x_array_interpolate.append(x_array_interpolate_lhs[it_2])
            for it_2 in range (0, len(x_array_interpolate_rhs)):
                x_array_interpolate.append(x_array_interpolate_rhs[it_2])
            for it_2 in range (0, len(y_array_interpolate_lhs)):
                y_array_interpolate.append(y_array_interpolate_lhs[it_2])
            for it_2 in range (0, len(y_array_interpolate_rhs)):
                y_array_interpolate.append(y_array_interpolate_rhs[it_2])
            params, cv = scipy.optimize.curve_fit(myHarmonic, x_array_interpolate, y_array_interpolate)
            for it_2 in range(line_lhs, line_rhs):
                variable_1[it_2,1] = myHarmonic(variable_1[it_2,0], params[0], params[1], params[2])
            flag = 0
            counter = 0
            
            # Now the smoothening of the joint region where there is original data and
            # the region of data coming from the harmonic interpolation.
            x_array_interpolate = []
            y_array_interpolate = []
            for it_2 in range (line_lhs-6, line_lhs-2):
                x_array_interpolate.append(variable_1[it_2,0])
                y_array_interpolate.append(variable_1[it_2,1])
            for it_2 in range (line_lhs+3, line_rhs):
                x_array_interpolate.append(variable_1[it_2,0])
                y_array_interpolate.append(variable_1[it_2,1])
            my_interpolation = CubicSpline(x_array_interpolate, y_array_interpolate)
            for it_2 in range (line_lhs-2, line_lhs+3):
                variable_1[it_2,1] = my_interpolation(variable_1[it_2,0])
            x_array_interpolate = []
            y_array_interpolate = []
            for it_2 in range (line_lhs, line_rhs-2):
                x_array_interpolate.append(variable_1[it_2,0])
                y_array_interpolate.append(variable_1[it_2,1])
            for it_2 in range (line_rhs+3, line_rhs+6):
                x_array_interpolate.append(variable_1[it_2,0])
                y_array_interpolate.append(variable_1[it_2,1])
            my_interpolation = CubicSpline(x_array_interpolate, y_array_interpolate)
            for it_2 in range (line_rhs-2, line_rhs+3):
                variable_1[it_2,1] = my_interpolation(variable_1[it_2,0])
            
# -----------------------------------------------------------------------------------------
# Defining the array delta_U
bin_U = 0.01 # angs
delta_U = np.zeros((lines_rdf,2))
for it_1 in range (0, len(delta_U)):
    delta_U[it_1,0] = it_1*bin_U + bin_U

delta_U[:,1] = (variable_1[:,1] - U_target[:,1])

table_begin = 2
while 2 > 1:
    if delta_U[0,0] == table_begin:
        break
    delta_U = np.delete(delta_U, 0, 0)
        
# -----------------------------------------------------------------------------------------
# Selecting range for which the update is going to be apply (i.e., range in which data exists
# in both variable_1 and U_target) effectively. 
for it_1 in range (0, len(variable_1)):
        if (U_target[it_1,1] != 0) & (variable_1[it_1,1] != 0):
            line_min = it_1
            break
for it_1 in range(0, line_min):
    U_target[it_1,1] = 0
    variable_1[it_1,1] = 0

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# The factor values used for the update is 0.25 or smaller.
# Note: the criteria set up really isnt the more reasonable criteria to choose. I didnt reason
# super well when I was setting up the criteria, but my aim was for a set up where a small value
# of pre-factor is used if distributions are ever very different from one another in terms of or-
# der of magnitude to prevent steep changes in the potential. In any case, this braindead option 
# that I noticed later on is not a problem for converging to IBI potentials successfully, since I 
# indeed did so using this code.
checkup = 0
for it_1 in range (0, len(delta_U)):
    if U_model_i[it_1,1] != 0.0:
        value = (delta_U[it_1,1])/U_model_i[it_1,1]
        if (value > 3) or (value < 1/3):
            checkup = 1

if checkup == 0:
    factor = 0.25
if checkup == 1:
    factor = 0.05
    
U_model_i_next = np.zeros((len(U_model_i),2))
U_model_i_next[:,0] = U_model_i[:,0]
U_model_i_next[:,1] = U_model_i[:,1] - factor*(delta_U[:,1])

# ----------------------------------------------------------------------------
# Here I will now re-build the data for U_model_i_next (i.e., potential of the next iteration)
# for the region that comes before the line_min variable. In fact, in a way, it is ultimately 
# as if I had never really considered the range xE[0, line_min] of delta_U's values into new 
# value of the potential. In fact, within this same range, the values of U_model_i per say dont
# really matter either: it is values of y of points with x following the x = line_min that will
# dictate the "looks" of the potential U_model_i_next in this x range since they serve as basis
# to create the linear extrapolation that will be useful to derive values for U_model_i_next
# at xE[0,line_min].
def myLinear(x,a,b):
    return a*(x) + b

flag = 0
counter = 0
x_array_interpolate = []
y_array_interpolate = []
for it_1 in range (0, len(delta_U)):
    if delta_U[it_1,1] != 0.0:
        max_line = it_1
        flag = 1
        break
    
for it_1 in range (max_line, len(delta_U)):
    x_array_interpolate.append(U_model_i_next[it_1,0])
    y_array_interpolate.append(U_model_i_next[it_1,1])
    counter = counter + 1
    if counter == 3:
        params, cv = scipy.optimize.curve_fit(myLinear, x_array_interpolate, y_array_interpolate, bounds = ([-math.inf, -math.inf], [-0.05, math.inf]))
        for it_2 in range(0, max_line):
            U_model_i_next[it_2,1] = myLinear(U_model_i_next[it_2,0], params[0], params[1])
        x_array_interpolate = []
        y_array_interpolate = []
        for it_2 in range (max_line-10, max_line-2):
            x_array_interpolate.append(U_model_i_next[it_2,0])
            y_array_interpolate.append(U_model_i_next[it_2,1])
        for it_2 in range (max_line+3, max_line+8):
            x_array_interpolate.append(U_model_i_next[it_2,0])
            y_array_interpolate.append(U_model_i_next[it_2,1])
        my_interpolation = CubicSpline(x_array_interpolate, y_array_interpolate)
        for it_2 in range (max_line-2, max_line+3):
            U_model_i_next[it_2,1] = my_interpolation(U_model_i_next[it_2,0])
        break
            
# ----------------------------------------------------------------------------------
# Creating binID column and force-field information (derivative is numerically approximated in
# quite a basic/simple way). This information should exist in the table potential used by LAMMPS.
column = np.zeros((len(U_model_i_next),1))

for it_1 in range (0, len(U_model_i_next)):
    if it_1 == len(U_model_i_next) - 1:
        column[it_1,0]= (U_model_i_next[it_1,1]-U_model_i_next[it_1-1,1])/(U_model_i_next[it_1,0]-U_model_i_next[it_1-1,0])
        continue
    column[it_1,0]= (U_model_i_next[it_1+1,1]-U_model_i_next[it_1,1])/(U_model_i_next[it_1+1,0]-U_model_i_next[it_1,0])

column = -1*(column)
output = np.concatenate((U_model_i_next, column), axis = 1)

column = np.zeros((len(U_model_i_next),1))
for it_1 in range (0, len(U_model_i_next)):
    column[it_1,0] = it_1 + 1
output = np.concatenate((column, output), axis = 1)

# -------------------------------------------------------------------------
# Shifting the potential at zero.
y_value = output[int(len(output)-1),2]
for it_1 in range (0, len(output)):
    output[it_1,2] = output[it_1,2] - y_value
    
# -------------------------------------------------------------------------   
ofi = open("U_model_i_next.dat", 'w')
ofi.write('POTENTIAL')
ofi.write('\n')
ofi.write('N 1301 R 2.000000 15.000000')
ofi.write('\n')
ofi.write('\n')
for it_1 in range(len(U_model_i_next)):
    ofi.write(str(int(output[it_1,0])))
    ofi.write('\t')
    ofi.write(str(output[it_1,1]))
    ofi.write('\t')
    ofi.write(str(output[it_1,2]))
    ofi.write('\t')
    ofi.write(str(output[it_1,3]))
    ofi.write('\n')
ofi.close()
