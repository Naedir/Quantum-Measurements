"""
Created on Thu Aug 15 08:53:12 2019

@author: eddy
"""

'''This is a mother code. it allows for creating a two qubit system (ancilla and qubit)
    it allows for coupling qubit to ancilla to a varying degree
    measure it, and apply the 'correction factor'
    As well as plot the results in the form in histograms, visualise the system on the Bloch sphere
'''
import numpy as np
from qutip import *
import scipy as sp
import cmath as cm
import matplotlib.pyplot as plt
import random
import time
import statistics as st

start_time = time.time()
plt.close('all')

## First define some variables that will stay constant throughout the whole code

'''
                                    PI'S
'''
v1=(basis(2,0)+basis(2,1)).unit()   # +ve x-axis
v2=(basis(2,0)-basis(2,1)).unit()   # -ve x-axis

piplus=v1*v1.dag()  # +ve x-axis
piminus=v2*v2.dag() # -ve x-axis



''' the 'system' class will introduce an initial state of the qubit and will couple it with the ancilla
    INITIAL STATE IS RESTRICTED TO LIE ON THE XY PLANE '''
    
class system:   
    ### first initiate an phase, that will be between 0 and 1 (0 by default)
    ### it will then be multiplied by 2*pi, to make a full circle
    ### also G is the coupling strength, between 0 and 1
    ### initial_state is a unit vector on xy plane
    

    def __init__(self,angle,G):
        self.angle=angle
        '''             CHOOSING AN ANGLE:
        *all the angles are with respect to the positive x-axis
        ...angle=0 means the qubit lies on the +ve x-axis
        *angle=1 is a full 2pi rotation, therefore it also lies on the +ve x-axis
        *angle=0.5 is a pi rotation, therefore it lies on a -ve x-axis
        *angle=0.25 is a pi/2 rotation, so it lies on a +ve y-axis
        *etc...
        '''
        self.G=G
        #ancilla
        sqrt=(np.sqrt(2)/2)
        self.ancilla=(basis(2,0)+(sqrt+sqrt*1j)*basis(2,1)).unit()   # just some random ancillla qubit on the equatorial plane
        self.flipped_ancilla=(basis(2,0)+cm.exp(1j*np.pi)*(sqrt+sqrt*1j)*basis(2,1)).unit()
        #initial qubit
        self.phase=angle*2*np.pi
        self.initial_state=(basis(2,0)+cm.exp(1j*self.phase)*basis(2,1)).unit()
        self.initial_state_coupled=tensor(self.initial_state,self.ancilla)
        #density matrices
        self.ancilla_dens=self.ancilla*self.ancilla.dag()
        self.initial_state_dens=self.initial_state*self.initial_state.dag()
        self.initial_state_coupled_dens=self.initial_state_coupled*self.initial_state_coupled.dag()
        self.flipped_ancilla_dens=self.flipped_ancilla*self.flipped_ancilla.dag()

    ## make a function to cahnge G and phase (nothing exciting here, just take a new G, and a new angle. it updates initial state, and coupled system, and phase)
    def change_G_and_angle(self,G,angle):
        self.G=G
        self.angle=angle
        self.phase=angle*2*np.pi
        self.initial_state=(basis(2,0)+cm.exp(1j*self.phase)*basis(2,1)).unit()
        self.initial_state_coupled=tensor(self.initial_state,self.ancilla)
        self.initial_state_dens=self.initial_state*self.initial_state.dag()
        self.initial_state_coupled_dens=self.initial_state_coupled*self.initial_state_coupled.dag()
        
    def couple(self,plot=False):
        '''
                                            TRANSFRMATION (Unitary)
        '''
        t1, t2 = tensor(piplus,qeye(2)), tensor(piminus,Qobj.expm(1j*self.G*sigmaz()*(np.pi/2)))    
        M=t1+t2     ### This is a unitary that is described in https://arxiv.org/abs/1211.0261 (eq. (5))
                    ### after it acts on the qubit and ancilla, it will couple it to a varying degree.
        self.system_coupled=M*self.initial_state_coupled_dens*M.dag() # this is the resulting coupled system, given the qubit and ancilla from the 'system' class
       
        '''
                                            POST TRANSFORMATION STATES
        '''
        self.post_unitary_ancilla=self.system_coupled.ptrace(1)
        self.post_unitary_qubit=self.system_coupled.ptrace(0)
        ########################################################################################################################
        
        if plot==True:          
            ## make a plot of the pre- and post-unitary states. it will be plotted on the Bloch Sphere if the plot option in the
            ##weak_modified function will be set  to True
            
            Bloch_vectors, Bloch_annotations = [piplus,piminus,self.ancilla, self.initial_state], [r"$\Pi_+$",r"$\Pi_-$",'Ancilla',r'    $\Psi$']
            Bloch_vectors_post_unitary, Bloch_annotations_post_unitary = [piplus,piminus,post_unitary_ancilla,post_unitary_qubit], [r"$\Pi_+$",r"$\Pi_-$",'Ancilla','Ancilla',r'    $\Psi$']
            name1, name2 = plt.figure('Before Transformation, G= %0.1f' % (self.G)), plt.figure('After Transformation, G= %0.1f' % (self.G))
            b, b2 = Bloch(fig=name1), Bloch(fig=name2)

            def bloch(self,vector,annotation):
                for i,j in zip(vector,annotation):
                    self.add_states(i)
                    self.add_annotation(i,j)
                self.make_sphere()
            bloch(b,Bloch_vectors,Bloch_annotations)
            bloch(b2,Bloch_vectors_post_unitary,Bloch_annotations_post_unitary)
    
    #   now a function that takes a coupled system and measures it.
    def measure(self, ensamble_size,plot=False):
        'ensamble_size is the number of measurements, equivalent to the ensemble size'
        'plot (False by default) produces a histogram of measurement results'
        #first build a  projection operator. 
        # the system will be projected on the initial ancilla. this way if the ancilla hasnt moved, the projection will be 1
        #ancilla rotated by pi winn give the projection = 0. ancilla rotated by pi/2 will be projected on  the init.ancilla giving 0.5, etc.
        measurement_operator = tensor(qeye(2), self.ancilla_dens)   #operator for not flipped ancilla
        measurement_operator2 = tensor(qeye(2), self.flipped_ancilla_dens)  #operator for flipped ancilla
        
        #Since the projection is on the initial ancilla, the probability that the ancilla collapsed on the initial state after 
        #measurement is Pi=Tr(Mi*rho), where Mi is the projection operator and rho is a density matrix of the system
        #if the ancilla didn't collapse on the initial state, it will must then be on the flipped_ancilla state, as M={M(ancilla), M(flipped_ancilla)} is a complete set
        
        self.couple()   #system has to be coupled first, to get the results
        result=(measurement_operator*self.system_coupled*measurement_operator.dag()).tr()  #Result is a number, that is probability of the ancilla didnt flip
        
        '''### also make a wavefunction of the system after measurement. it is reproduced using Rho_prime=Mn*rho*Mn.dag()/p(n), if the result was n'''
        
        '''now make 2 initial qubit wavefunctions that are a result of a measurement outcome (measurement and measurement2)'''

        state1=(measurement_operator*self.system_coupled*measurement_operator).unit()
        state1=state1.ptrace(0)

        state2=(measurement_operator2*self.system_coupled*measurement_operator2).unit()
        state2=state2.ptrace(0)
    
        '       MEASUREMENT STATISTICS    '
        binn=[]
        for i in range(ensamble_size):  ###result 1 means along ancilla, aka the ancilla didnt flip.binn is an array of results (0 or 1)
            if random.random()<result:
                binn.append(1)
                post_measurement_qubit=state1
            else:
                binn.append(0) 
                post_measurement_qubit=state2
        '''now the correction of data. an equation described in _____________ will be used to modify data so that 
        for low G values, the statistics should resemble those for high G values'''

        self.correction=(1/2)+(1/2)*(np.power(np.sin(self.G*np.pi/2),2)-np.power(np.cos(self.G*np.pi/2),2))
        number_of_flips=len(binn)-sum(binn)
        number_of_no_flips=sum(binn)
        corrected_number_of_flips=number_of_flips/self.correction
        corrected_number_of_no_flips=abs(len(binn)-corrected_number_of_flips)
        
        plain_results, corrected_results = [], []
        plain_results.append(number_of_flips)
        plain_results.append(number_of_no_flips)
        corrected_results.append(corrected_number_of_flips)
        corrected_results.append(corrected_number_of_no_flips)
        
        ### the output of this function will be two arrays, both with two columns, and 'ensemble _size' number of rows, and an integer (ensamble size)
        ### first array is an array of plain measurement results, and the second is an array of measurement results after applying the correction factor
        return plain_results, corrected_results, ensamble_size, post_measurement_qubit
    
    
    ##now make a function that will get a backaction measures
    def backaction(self,reps):  #reps is the number of runs to take an average out of
        
        data=[]
        for i in range(reps):
            post_measurement_qubit=self.measure(1)[3] #this just gets a post-measurement qubit after one measurement. the state is called from the measure function
            # to get the backaction, find a trace distance between post measurement and initial qubit wavefunction##
            distance=tracedist(self.initial_state_dens,post_measurement_qubit)
            data.append(distance)
        mean_trace_distance=np.mean(data)

        return mean_trace_distance
    
    ##Now make a function for the information gain:::
    def information(self,number_of_qubits):
        ''' Information gain is defined as a trace distance between initial_qubit and a reproduced qubit.
        Reproduced qubit is a qubit based on measurement statistics outcomes. 
        Initial guess for the reproduiced qubit is that it lies on Pi+ (+ve x-axis). then based on (normalised) number of flips, a phase rotation is added.
        such that if flips=1, there is a pi rotation. if there's 50/50 flips, there is a pi/2 rotation (or 3pi/2).
        since there are always two qubits possible to be reproduced, the closer one is chosed for the information gain.'''
        data=self.measure(number_of_qubits)
        number_of_flips=data[1][0]/number_of_qubits      ### Normalised number of flips for 'number_of_qubits' measurements
        
        '''now reproduce the wavefunction such that starting point is Pi+, and +/- number_of_flips_*pi is the rotation'''
        psi1=(basis(2,0)+cm.exp(1j*(number_of_flips*np.pi))*basis(2,1)).unit()
        psi2=(basis(2,0)+cm.exp(-1j*(number_of_flips*np.pi))*basis(2,1)).unit()
        ##now the trace distance. also trace distance is 0 when qubits are parallel and 1 if they are orthogonal, therefore take abs(1-tracedist) for the information gain
        distance1 = tracedist(self.initial_state,psi1)
        distance2 = tracedist(self.initial_state,psi2)
        results=[]
        results.append(distance1)
        results.append(distance2)
        information_gain=abs(1-min(results))
        return information_gain

    ### estimate an error in the onformation gain
    def error(self, ensamble_size, number_of_runs): #number of runs tells how many times to do the experiment (measurement_at_a_given_G)
    
        corrected_flips=[]
        for i in range(number_of_runs):
            corrected_flips.append(self.measure(ensamble_size)[1])
        corrected_flips=np.array(corrected_flips)/ensamble_size
        ## sum each row of the corrected_flips array
        #sum_of_flips=sum(np.transpose(corrected_flips)[0])
        ##now find a true mean, which is based on the initial position of the qubit::
        ''' it is found such thah, angle divided by 2pi gives the ratio of ancilla flips'''
        if self.angle > 0.5:
            init_angle=self.angle-0.5
            init_angle=init_angle*2
        else:
            init_angle=self.angle*2
        mean=float(init_angle)
        standard_deviation=st.stdev(corrected_flips.transpose()[0], xbar = mean)
        return standard_deviation
    
    ''' Now make a Master Function that gives the error, information gain and backaction for a given ensamble size, G and angle
    This is a statistical data, therefore number of runs has to be specified as well. number of runs is the number of data points to take statistics out of.
    The higher the number of runs, the better (closer to the real value) the numbers are, but also time taken to calculate it increases.'''
    
    def stat_master(self, ensamble_size, number_of_runs):
        stat_information=self.information(ensamble_size)
        stat_backaction=self.backaction(number_of_runs)
        stat_error=self.error(ensamble_size,number_of_runs)
        
        print("For an ensamble of {} qubits prepared at {}* pi w.r.t. +ve x-axis".format(ensamble_size, self.angle))
        print("\nthe information gain is {} %".format(int(stat_information*100)))
        print("\nthe backaction is {} %".format(int(stat_backaction*100)))
        print("\nand the associated error is {} %".format(int(stat_error*100)))

    @staticmethod
    def dir():
        print('\n        GUIDE:')
        print('----------------------------------------')
        print('Atributes:\n  *angle\n  *phase\n  *ancilla\n  *initial_state\n  *initial_state_coupled\n  *ancilla_dens\n  *initial_state_dens\n  *initial_state_coupled_dens\n  *correction\n  *system_coupled\n  *post_unitary_ancilla\n  *post_unitary_qubit\n  *flipped_ancilla')
        print('----------------------------------------')
        print('Methods:\n  *couple\n  *measure\n  *change_G_and_angle\n  *backaction\n  *information\n  *error\n  *stat_master')
        
        
print("--- %s seconds ---" % (time.time() - start_time))
