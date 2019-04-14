from pyquil import Program, get_qc
from pyquil.quil import DefGate
from pyquil.gates import *
import numpy as np 

def superdense_coding(bits, p):
	if bits == '00' : return p + I(qubit=0)

	elif bits == '01' : return p + Z(qubit=0)

	elif bits == '10' : return p + X(qubit=0)

	elif bits == '11' : return p + Y(qubit=0)

	else: raise ValueError 

def superdense_coding_arbitrary(bit_string, p):
	#for every 2 bits you want to send, prepare the first Bell state and run superdense coding
	#return a list of states that Bob can measure and concatenate the resulting bits together sequentially
	state_list = []
	index = 0
	for i in range(int(len(bit_string)/2)):
		bits = bit_string[index:index + 2]
		state_list.append(superdense_coding(bits, p))
		index +=2
	return state_list

def measure_bob(program_list):
	#Inputs: The program list (could just be one entry) resulting from Alice's applications of gates from the above functions
	#Output: The desired bit string
	bit_string = ''
	for program in program_list:
		#measure in bell basis
		#constructing the change of basis matrix
		change = 1/np.sqrt(2) * np.array([[1,0,0,1],[1,0,0,-1],[0,1,1,0],[0,1,-1,0]])
		# Get the Quil definition for the new gate
		change_definition = DefGate("CHANGE", change)
		# Get the gate constructor
		CHANGE = change_definition.get_constructor()
		# Then we can use the new gate
		program += change_definition
		program += CHANGE(0,1)
		#Now will be changed to the computational basis
		result = qc.run_and_measure(program, trials = 1)
		bit_string += str(result[0])
		bit_string += str(result[1])
	return bit_string

# run the program on a QVM
qc = get_qc('9q-square-qvm')
# construct a Bell State program and choose a bit string to send
bit_string = '10010011110100001011000111001001100111'
entangled_start = Program(H(0), CNOT(0, 1))
#Get the program list to send to bob (takes half the bits)
#Here I am assuming that Alice only needs to send an even number of bits with this method,
#and in the case of an odd length bit string, can send the last bit classically in a separate message. 
programs_from_Alice = superdense_coding_arbitrary(bit_string, entangled_start)
info_Bob_got = measure_bob(programs_from_Alice)
print(info_Bob_got)
