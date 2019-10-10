#! /usr/bin/python3
from subprocess import call, Popen, PIPE, check_output
import os
import pandas as pd
import numpy as np
import shutil
import random
import csv
#os.mkdir('outputs')

k='5'
newpath = os.path.relpath('./htree_mesh_comp/mesh_config')
deviation_string = '/Standard_Deviation'
node_string ='/Individual_Node_Stats'
latency_string = '/latency'
output_string = './output_'

def replace_in_file(filename, key, newvalue):
	f = open(filename,"r")
	lines = f.readlines()
	f.close()
	for i,line in enumerate(lines):
		if line.split('=')[0].strip(' \n') == key:
			lines[i] = key + '=' + newvalue + ';\n'
	f = open(filename,"w")
	f.write("".join(lines))
	f.close()

def resize_inj_rate(filename, size, value):
	k = int(size)
	#inject_rates = []
	with open(filename,'r') as f:
		data = f.readlines()

	#for line in data:
	#	line = line.rstrip()
	#	inject_rates.append(line.split(' '))

	inject_rates = np.zeros((k*k, k*k))
	#inject_rates = np.resize(inject_rates, (k*k, k*k))
	rows, cols = inject_rates.shape
	for row in range(rows):
		for col in range(cols):
			if (value == 0):
				inject_rates[row][col] = float(random.randint(1,9))/1000
			else:
				inject_rates[row][col] = value
			#inject_rates[row][col] = np.rand(0,1)/1000
	#np.savetxt(filename, inject_rates,fmt='%s')
	np.savetxt(filename, inject_rates)
	f.close()


def program(value,count):
		
	replace_in_file(newpath,"k",k)
	resize_inj_rate('inj_rate.txt',k,value)


	#call(["./booksim", "htree_mesh_comp/mesh_config"])


	booksim_command = './booksim ' + 'htree_mesh_comp/mesh_config ' + '> output_log'
	os.system(booksim_command)
	#text = check_output(["./booksim", "htree_mesh_comp/mesh_config"])
	#text= pipe.communicate()[0]
	#f = open('output_log','w')
	#f.write(text.decode('utf-8'))
	#f.close()

	nodes = []
	with open('Individual_Node_Stats','r') as f:
		data = f.readlines()
	f.close()
	for line in data:
		line = line.rstrip()
		nodes.append(line.split("\t"))
	nodes = np.array(nodes)
	nodes = nodes.reshape(1,-1)

#	np.savetxt('dataset.csv', [nodes], delimiter=' ', fmt='%s')

	f=open("Standard_Deviation","r")
	lines=f.readlines()
	deviation=[]
	for x in lines:
		deviation.append(x.split('\t')[1])
	f.close()
	
	deviation = np.array(deviation)
	deviation = deviation.reshape(1,-1)
#	with open('dataset.csv','a') as f:
#		np.savetxt(f,deviation,delimiter=" ",fmt='%s')
	
	mean = []
	for x in lines:
		mean.append(x.split('\t')[0])
	
	mean = np.array(mean)
	mean = mean.reshape(1,-1)
#	with open('dataset.csv','a') as f:
#		np.savetxt(f,mean,delimiter=" ",fmt='%s')

	f = open('output_log','r')
	lines = f.readlines()
	f.close()

	find_latency = []

	for line in lines:
		if "Packet latency average" in line:
			find_latency.append(line)

	lat = find_latency[-1].split(" ")[4]
#	f = open(latency_string,"a")
#	f.write(lat)
#	f.close()

	result = np.concatenate((nodes,deviation,mean), axis = 1)
	result = np.append(result,lat)	
	result = result.reshape(1,-1)
	df = pd.DataFrame(result)
	#export_csv = df.to_csv (r'dataframe.csv', index = None, header=False)

	with open(r'dataframe'+k+'.csv','a') as fd:
		df.to_csv(fd, index = None, header= False)


	
	with open('dataset.csv','a') as f:
		np.savetxt(f,result,delimiter=" ",fmt='%s')


	os.mkdir('output_'+k+str(value)+str(count))
	shutil.copy2('Individual_Node_Stats', output_string+k+str(value)+str(count)+node_string)
	shutil.copy2('Standard_Deviation',output_string+k+str(value)+str(count)+deviation_string)
	shutil.copy2('inj_rate.txt',output_string+k+str(value)+str(count)+'/inj_rate.txt')
	f = open(output_string+k+str(value)+str(count)+latency_string,"w")
	f.write(lat)
	f.close()

for i in [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009]:
	program(i,0)

count = 0
for i in range(4990):
	count = count+1
	program(0,count)

k = int(k)
size = k*k*k*k + 1

header = []
for i in range(1,size):
	header.append("latency_"+str(i))

for i in range(1,size):
	header.append("std_arrival_"+str(i))

for i in range(1,size):
	header.append("mean_arrival_"+str(i))
header.append("average_latency")

df = pd.read_csv('dataframe'+str(k)+'.csv',header = None)

with open('dataframe'+str(k)+'.csv','wb') as f:
	w = csv.writer(f)
	w.writerow(header)
	for index, data in df.iterrows():
		w.writerow(data)
