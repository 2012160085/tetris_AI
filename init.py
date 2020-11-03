#WRECK IT RALPH PLAYER FOR BATTLE 2Pimport numpy as np
import numpy as np
import time
import random
import pickle
from multiprocessing import Process, Queue ,Lock , JoinableQueue,Manager
import pynput
from pynput.keyboard import Key, Controller
from ctypes import windll
import threading
import queue
def mouse_wait():
	while True:
		if mouse.position[1] >100:
			break
def getpixel(x,y):
    return windll.gdi32.GetPixel(dc,x,y)
def framePos():
	xx = 50
	yy = 500
	gap = 32
	find = True
	icolor = getpixel(xx,yy)
	
	while abs(gap) >= 1:
		
		while (icolor == 16777215) is find:
			xx += gap
			icolor = getpixel(int(xx),int(yy))
	
		gap = -gap/2
		find = not find

	gap = 8
	find = True
	xx += 1
	
	icolor = getpixel(int(xx),int(yy))
	
	while abs(gap) >= 1:
		
		while (icolor == 16777215) is False:
			yy += gap
			icolor = getpixel(int(xx),int(yy))
		yy -= gap
		icolor = getpixel(int(xx),int(yy))
	
		gap = gap/2
	print("GAME SCREEN DETECTED... /FRAME POS:",int(xx),int(yy))	
	return int(xx),int(yy)
def startingBlock(x,y):
	dx = x-anchor_x
	dy = y-anchor_y
	while True:
		val = 0
		for i in range(0,3):
			for j in range(0,2):
				if windll.gdi32.GetPixel(dc,dx+ghost_x+i*18,dy+ghost_y+j*18) == 6710886:
					val += 2**(i+j*3)
		if val == 56:
			return 0
		if val == 54:
			return 1
		if val == 60:
			return 5
		if val == 57:
			return 6
		if val == 51:
			return 2
		if val == 30:
			return 3
		if val == 58:
			return 4
def getInitmap(x,y):
	dx = x-anchor_x
	dy = y-anchor_y
	time.sleep(0.1)
	temp = []
	temp.append(startingBlock(x,y))
	time.sleep(0.5)
	for i in range(0,4):
		icolor = getpixel(dx+block_x[i],dy+block_y[i] )
		if icolor == colors[i][0]:
			temp.append(0)
		if icolor == colors[i][1]:
			temp.append(1)
		if icolor == colors[i][2]:
			temp.append(2)
		if icolor == colors[i][3]:
			temp.append(3)
		if icolor == colors[i][4]:
			temp.append(4)
		if icolor == colors[i][5]:
			temp.append(5)
		if icolor == colors[i][6]:
			temp.append(6)
	if len(temp) == 5:
		return temp

def newblock(x,y):
	dx = x-anchor_x
	dy = y-anchor_y
	while True:
		time.sleep(0.04)
		icolor = getpixel(dx+block_x[4],dy+block_y[4])
		if icolor == colors[4][0]:
			return 0
		if icolor == colors[4][1]:
			return 1
		if icolor == colors[4][2]:
			return 2
		if icolor == colors[4][3]:
			return 3
		if icolor == colors[4][4]:
			return 4
		if icolor == colors[4][5]:
			return 5
		if icolor == colors[4][6]:
			return 6
def opening_cmd(k):
	#[0,2,3,4,5,6]
	if k is 0:
		return 2,0
	if k is 2:
		return -1,0
	if k is 3:
		return 2,0
	if k is 4:
		return 2,0
	if k is 5:
		return -1,1
	if k is 6:
		return 2,3
def MoveByCmd(move,rot,wait,wall):
	
	
	mouse_wait()
	######################################################################################################### 4
	print("MOVEMENT INPUT",move,rot)
	if wall != 0:
		if wall > 0:
			keyboard.press(Key.right)
		else:
			keyboard.press(Key.left)
		if rot in [1,2]:
			keyboard.press(Key.up)
		if rot is 3:
			keyboard.press(Key.ctrl_l)
		starttime = time.time()
		while time.time()-starttime < 0.065:
			pass
		else:
			if rot in [1,2]:
				keyboard.release(Key.up)
			if rot is 3:
				keyboard.release(Key.ctrl_l)
		while time.time()-starttime < 0.13:
			pass
		else:
			if rot is 2:
				keyboard.press(Key.up)
		while time.time()-starttime < 0.195:
			pass
		else:
			if rot is 2:
				keyboard.release(Key.up)
		while time.time()-starttime < 0.21:
			pass
		else:
			if wall > 0:
				keyboard.release(Key.right)
			else:
				keyboard.release(Key.left)
		for i in range(0,move):
			starttime = time.time()
			if wall < 0:
				keyboard.press(Key.right)
			else:
				keyboard.press(Key.left)
			while time.time()-starttime < 0.065:
				pass
			starttime = time.time()
			if wall < 0:
				keyboard.release(Key.right)
			else:
				keyboard.release(Key.left)
			while time.time()-starttime < 0.065:
				pass
				
	######################################################################################################### 3 2 1 0
	
	else:
		rot_time = rot
		if rot_time == 3:
			rot_time = 1
		mov_time = abs(move)
		rotcount = 0
		movcount = 0
		for i in range(0,max(mov_time,rot_time)):
			if movcount < mov_time:
				if move > 0:
					keyboard.press(Key.right)
				else:
					keyboard.press(Key.left)
			if rotcount < rot_time:
				if rot in [1,2]:
					keyboard.press(Key.up)
				if rot is 3:
					keyboard.press(Key.ctrl_l)
			starttime = time.time()
			while time.time()-starttime < 0.06:
				pass
			else:
				if movcount < mov_time:
					movcount += 1
					if move > 0:
						keyboard.release(Key.right)
					else:
						keyboard.release(Key.left)
				if rotcount < rot_time:
					rotcount += 1
					if rot in [1,2]:
						keyboard.release(Key.up)
					if rot is 3:
						keyboard.release(Key.ctrl_l)
			time.sleep(0.065)
	keyboard.press(Key.space)
	time.sleep(0.06)
	keyboard.release(Key.space)
	if wait:
		time.sleep(0.3)
def ralph_move(dir,_id,t_id,x):
	if dir == 1:
		x += 6
	if _id == 0 and t_id == 0:
		if x in [0]:
			return 0,0,-1
		elif x in [6]:
			return 0,0,1
		else:
			return x-3,0,0
	if _id == 0 and t_id == 1:
		if x in [0,1]:
			return x,3,-1
		elif x in [8,9]:
			return 9-x,1,1
		elif x in [2,3,4]:
			return x-4,3,0
		else:
			return x-5,1,0
	if _id == 1:
		if x in [0,1]:
			return x,0,-1
		elif x in [7,8]:
			return 8-x,0,1
		else:
			return x-4,0,0
	if _id in [2,3] and t_id == 0:
		if x in [0]:
			return 0,0,-1
		elif x in [6,7]:
			return 7-x,0,1
		else:
			return x-3,0,0
	if _id in [2,3] and t_id == 1:
		if x in [0]:
			return 0,3,-1
		elif x in [7,8]:
			return 8-x,1,1
		elif x in [1,2,3]:
			return x-3,3,0
		else:
			return x-4,1,0
	if _id in [4,5,6] and t_id == 0:
		if x in [0]:
			return 0,0,-1
		elif x in [6,7]:
			return 7-x,0,1
		else:
			return x-3,0,0
	if _id in [4,5,6] and t_id == 1:
		if x in [0,1]:
			return x,1,-1
		elif x in [7,8]:
			return 8-x,1,1
		else:
			return x-4,1,0
	if _id in [4,5,6] and t_id == 2:
		if x in [0]:
			return 0,2,-1
		elif x in [6,7]:
			return 7-x,2,1
		else:
			return x-3,2,0
	if _id in [4,5,6] and t_id == 3:
		if x in [0]:
			return 0,3,-1
		elif x in [7,8]:
			return 8-x,3,1
		else:
			return x-3,3,0
	

	
def felix_move(dir,_id,t_id,x):

	if dir == 0:
		x += 4
	if _id == 0 and t_id == 0:
		if x in [0]:
			return 0,0,-1
		elif x in [6]:
			return 0,0,1
		else:
			return x-3,0,0
	if _id == 0 and t_id == 1:
		if x in [0,1]:
			return x,3,-1
		elif x in [8,9]:
			return 9-x,1,1
		elif x in [2,3,4]:
			return x-4,3,0
		else:
			return x-5,1,0
	if _id == 1:
		if x in [0,1]:
			return x,0,-1
		elif x in [7,8]:
			return 8-x,0,1
		else:
			return x-4,0,0
	if _id in [2,3] and t_id == 0:
		if x in [0]:
			return 0,0,-1
		elif x in [6,7]:
			return 7-x,0,1
		else:
			return x-3,0,0
	if _id in [2,3] and t_id == 1:
		if x in [0]:
			return 0,3,-1
		elif x in [7,8]:
			return 8-x,1,1
		elif x in [1,2,3]:
			return x-3,3,0
		else:
			return x-4,1,0
	if _id in [4,5,6] and t_id == 0:
		if x in [0]:
			return 0,0,-1
		elif x in [6,7]:
			return 7-x,0,1
		else:
			return x-3,0,0
	if _id in [4,5,6] and t_id == 1:
		if x in [0,1]:
			return x,1,-1
		elif x in [7,8]:
			return 8-x,1,1
		else:
			return x-4,1,0
	if _id in [4,5,6] and t_id == 2:
		if x in [0]:
			return 0,2,-1
		elif x in [6,7]:
			return 7-x,2,1
		else:
			return x-3,2,0
	if _id in [4,5,6] and t_id == 3:
		if x in [0]:
			return 0,3,-1
		elif x in [7,8]:
			return 8-x,3,1
		else:
			return x-3,3,0
class Fstate:
	def __init__(self,grid,blocks,stack,opening):
		self.grid = grid
		self.blocks = blocks
		self.stack = stack
		self.opening = opening
	def block_refill(self):
		if len(self.blocks[0]) < 7:
			self.blocks = (self.blocks[0] + newmap(1),self.blocks[1])
	def block_add(self,x,y):
		self.blocks = (self.blocks[0] + [newblock(x,y)],self.blocks[1])
	def block_add_by_id(self,id):
		self.blocks = (self.blocks[0] + [id],self.blocks[1])
	def block_delete(self):
		self.blocks = (self.blocks[0][:-1],self.blocks[1])
class Rstate:
	def __init__(self,state,blocks,stack):
		self.state = state
		self.blocks = blocks
		self.stack = stack
	def show(self,msg,stop=False):
		print(msg,"state",self.state,"blocks",self.blocks,"stack",self.stack)
		if stop:
			input()
	def block_refill(self):
		if len(self.blocks[0]) < 7:
			self.blocks = (self.blocks[0] + newmap(1),self.blocks[1])
	def block_add(self,x,y):
		self.blocks = (self.blocks[0] + [newblock(x,y)],self.blocks[1])
	def block_add_by_id(self,id):
		self.blocks = (self.blocks[0] + [id],self.blocks[1])
def OpeningBlock2State(k):
	if k == 0 :
		return 0
	if k == 2 :
		return 5
	if k == 3 :
		return 4
	if k == 4 :
		return 2
	if k == 5 :
		return 7
	if k == 6 :
		return 6
def map2opening(map):#CHECKED! 
	#I,O,Z,S,T,J,L
	idx = 0
	openable = [0,2,3,4,5,6]
	while map[idx] not in openable:
		idx += 1
	dir = 1
	if map[idx] in [2,5]:
		dir = 0
	return ((map[:idx] + map[idx+1:]),(dir,idx, map[idx]))
def permutation(lst):#CHECKED! 
 
    # If lst is empty then there are no permutations
    if len(lst) == 0:
        return []
 
    # If there is only one element in lst then, only
    # one permuatation is possible
    if len(lst) == 1:
        return [lst]
 
    # Find the permutations for lst if there are
    # more than 1 characters
 
    l = [] # empty list that will store current permutation
 
    # Iterate the input(lst) and calculate the permutation
    for i in range(len(lst)):
       m = lst[i]
 
       # Extract lst[i] or m from the list.  remLst is
       # remaining list
       remLst = lst[:i] + lst[i+1:]
 
       # Generating all permutations where m is first
       # element
       for p in permutation(remLst):
           l.append([m] + p)
    return l
def printer(a): #CHECKED! 
	for aa in a:
		print(aa)
		print()
def basic_grid(t):#max_h #CHECKED! 
	temp = np.array([[0]*6]*8)
	if t == 0:
		temp[8-1][0] = 1
	else:
		temp[8-1][5] = 1
	return temp
def init_WBCD(l_w,init_w,init_b):#CHECKED! 
	temp_w=[]
	temp_b=[]
	temp_w.append((np.random.rand(l_w[0],l_w[1])-0.5)*init_w[0])
	temp_w.append((np.random.rand(l_w[1],l_w[2])-0.5)*init_w[1])
	temp_w.append((np.random.rand(l_w[2],1)-0.5)*init_w[2])
	temp_b.append((np.random.rand(l_w[1])-0.5)*init_b[0])
	temp_b.append((np.random.rand(l_w[2])-0.5)*init_b[1])
	return (temp_w,temp_b)
def mut_WBCD(neural,l_w,mut_w,mut_b):#CHECKED! 
	temp_w=[]
	temp_b=[]
	temp_w.append(np.add(neural[0][0],(np.random.rand(l_w[0],l_w[1])-0.5)*mut_w[0]))
	temp_w.append(np.add(neural[0][1],(np.random.rand(l_w[1],l_w[2])-0.5)*mut_w[1]))
	temp_w.append(np.add(neural[0][2],(np.random.rand(l_w[2],1)-0.5)*mut_w[2]))
	temp_b.append(np.add((np.random.rand(l_w[1])-0.5)*mut_b[0],neural[1][0]))
	temp_b.append(np.add((np.random.rand(l_w[2])-0.5)*mut_b[1],neural[1][1]))
	return (temp_w,temp_b)
def convolution(_ori,_c,_d):#CHECKED! 
	temp = np.zeros((_c.shape[1],_ori.shape[1]-2))
	for i_id,i in enumerate(_c):
		for o_id,o in enumerate(i):
			temp[o_id] = np.add(temp[o_id],np.convolve(_ori[i_id],o, 'valid'))
	return np.add(temp,_d)
def tetromino(_id): #CHECKED! 
	temp = []
	if _id == 0:
		temp.append((np.array([[1,1,1,1]]),[0,0,0,0]))
		temp.append((np.array([[1],[1],[1],[1]]),[0]))
	if _id == 1:
		temp.append((np.array([[1,1],[1,1]]),[0,0]))
	if _id == 6:
		temp.append((np.array([[1,0,0],[1,1,1]]),[0,0,0]))
		temp.append((np.array([[1,1],[1,0],[1,0]]),[0,2]))
		temp.append((np.array([[1,1,1],[0,0,1]]),[1,1,0]))
		temp.append((np.array([[0,1],[0,1],[1,1]]),[0,0]))
	if _id == 5:
		temp.append((np.array([[0,0,1],[1,1,1]]),[0,0,0]))
		temp.append((np.array([[1,0],[1,0],[1,1]]),[0,0]))
		temp.append((np.array([[1,1,1],[1,0,0]]),[0,1,1]))
		temp.append((np.array([[1,1],[0,1],[0,1]]),[2,0]))
	if _id == 3:
		temp.append((np.array([[0,1,1],[1,1,0]]),[0,0,1]))
		temp.append((np.array([[1,0],[1,1],[0,1]]),[1,0]))
	if _id == 2:
		temp.append((np.array([[1,1,0],[0,1,1]]),[1,0,0]))
		temp.append((np.array([[0,1],[1,1],[1,0]]),[0,1]))
	if _id == 4:
		temp.append((np.array([[0,1,0],[1,1,1]]),[0,0,0]))
		temp.append((np.array([[1,0],[1,1],[1,0]]),[0,1]))
		temp.append((np.array([[1,1,1],[0,1,0]]),[1,0,1]))
		temp.append((np.array([[0,1],[1,1],[0,1]]),[1,0]))
	return temp
def statewav(state):#max_h #CHECKED! 
	temp = []
	for x in range(0,6):
		y = 0
		while state[y][x] == 0:
			y += 1
			if y == 8:
				break
		temp.append(y)
	return temp
def wavAdj(wav):#CHECKED! 
	maxval = np.max(wav)
	return [(maxval - w) for w in wav]
def possibleWays(blocks,depth):#CHECKED!
	temp = []
	shift = blocks[1]
	if shift == -1:
		nextBlocks = blocks[0][:depth+1]
		for i in range(0,2**(len(nextBlocks)-1)-1):
			binary = np.binary_repr(i,width=len(nextBlocks))
			temp_way=[]
			temp_shift = shift
			for j,block in enumerate(nextBlocks):
				if temp_shift == -1:
					temp_shift = block
				else:
					if binary[j]=='1':
						temp_shift,block = block,temp_shift
					temp_way.append(block)
			temp.append((temp_way,temp_shift))
		temp.append((nextBlocks[:-1],-1))
	else:
		nextBlocks = blocks[0][:depth]
		for i in range(0,2**len(nextBlocks)):
			binary = np.binary_repr(i,width=len(nextBlocks))
			temp_way=[]
			temp_shift = shift
			for j,block in enumerate(nextBlocks):
				if binary[j]=='1':
					temp_shift,block = block,temp_shift
				temp_way.append(block)
			temp.append((temp_way,temp_shift))
	return temp
def gridScore(state,neurals,curval):#CHECKED!
	weights,biases= neurals
	inpt = []
	st_wav = wavAdj(statewav(state))
	curve_value = 0
	curve_set = set()
	for i in range(0,4):	
		list_key = st_wav[i:i+3] - min(st_wav[i:i+3])
		list_key = np.minimum(list_key,4)
		cv = curval[str(list_key[0])+str(list_key[1])+str(list_key[2])]
		curve_value += len(cv)
		curve_set.update(cv)
	inpt.append((curve_value-12)/12)
	inpt.append((len(curve_set)-3)/3)
	inpt.append(np.std(st_wav)-1)
	inpt.append((st_wav[0]-4)/4)
	inpt.append((st_wav[-1]-4)/4)

	a1 = np.maximum(0,np.add(np.matmul(inpt,weights[0]),biases[0]))
	a2 = np.maximum(0,np.add(np.matmul(a1,weights[1]),biases[1]))
	a3 = np.matmul(a2,weights[2])

	return a3[0]
def newmap( k ):#CHECKED!
	temp=[]
	tetromino_set = [0,1,2,3,4,5,6]
	for i in range(0,k):
		np.random.shuffle(tetromino_set)
		temp += tetromino_set
	return temp
def candidates(f_state,neurals,curval): #[Grid,BlockState,Where] #CHECKED!
	temp = []
	blocks = f_state.blocks
	stwave = statewav(f_state.grid)
	empty_shift = False
	dir,open_count,open_id = f_state.opening
	stack = f_state.stack
	if blocks[1] == -1:
		ids=[blocks[0][0],blocks[0][1]]
		empty_shift = True
	else:
		ids=[blocks[0][0],blocks[1]]
	for s_id,_id in enumerate(ids):
		for t_id,tetro in enumerate(tetromino(_id)):
			b,bwav = tetro
			if open_count > 0:
				if dir == 0:
					x_from,x_to = 1,7
				else:
					x_from,x_to = 0,6
			else:
				x_from,x_to = 0,7
			for x in range(x_from,x_to-b.shape[1]):
				_state = np.copy(f_state.grid)
				dist = [bwav[i]+stwave[i+x] for i in range(0,len(bwav))]
				
				if all(xx == dist[0] for xx in dist):
					y = dist[0]
					
					if y >= b.shape[0] and stack + 8 - y + b.shape[0] <= maximum_height:#최대높이
						
						crossed = False
						for i in range(0,b.shape[1]):
							for j in range(0,b.shape[0]):
								if b[j][i] == 1:
									if f_state.grid[y-b.shape[0]+j][i+x] == 1:
										crossed = True
									_state[y-b.shape[0]+j][i+x] = 1
						if not crossed:
							temp_lineclear = 0
							remove_row_indx = []
							for i in range(y-b.shape[0],y):
								if all(_state[i]):
									temp_lineclear += 1
									remove_row_indx.append(i)
							_state = np.delete(_state,remove_row_indx,0)
							if temp_lineclear > 0:
								_state = np.concatenate(([[0]*6]*temp_lineclear,_state ), axis=0)
							if s_id == 1 and empty_shift:
								newmaps = blocks[0][2:]
							else:
								newmaps = blocks[0][1:]
							if s_id == 1:
								newshift = ids[0]
							else:
								newshift = blocks[1]
							f_state_1 = Fstate(_state,(newmaps,newshift),f_state.stack+temp_lineclear,(dir,open_count-1,open_id))
							temp.append((f_state_1,(f_state_1,_id,t_id,x,blocks[0][0]!=_id)))
	if len(temp) > 4:
		arg = np.argsort([gridScore(t[0].grid,neurals,curval) for t in temp])
		arg = arg[-4:] 
		return [temp[i] for i in arg]
	return temp#갯수제한
def nextGrids(gridNopenings,_id,neurals,curval):#CHECKED! 
	temp = []
	for gridNopening in gridNopenings:
		ttemp = []
		stack = gridNopening[2]
		dir,open_count,open_id = gridNopening[1]
		grid = gridNopening[0]
		stwave = statewav(grid)
		for tetro in tetromino(_id):
			b,bwav = tetro
			if open_count > 0:
				if dir == 0:
					x_from,x_to = 1,7
				else:
					x_from,x_to = 0,6
			else:
				x_from,x_to = 0,7
			for x in range(x_from,x_to-b.shape[1]):
				_state = np.copy((grid))
				dist = [bwav[i]+stwave[i+x] for i in range(0,len(bwav))]
				
				if all(x == dist[0] for x in dist):
					y = dist[0]
					
					
					if y >= b.shape[0] and stack + 8 - y + b.shape[0] <= maximum_height:#최대높이:
						
						crossed = False
						for i in range(0,b.shape[1]):
							for j in range(0,b.shape[0]):
								if b[j][i] == 1:
									if grid[y-b.shape[0]+j][i+x] == 1:
										crossed = True
									_state[y-b.shape[0]+j][i+x] = 1
						if not crossed:
							
							temp_lineclear = 0
							remove_row_indx = []
							for i in range(y-b.shape[0],y):
								if all(_state[i]):
									temp_lineclear += 1
									remove_row_indx.append(i)
							_state = np.delete(_state,remove_row_indx,0)
							if temp_lineclear > 0:
								_state = np.concatenate(([[0]*6]*temp_lineclear,_state ), axis=0)
							ttemp.append((_state,(dir,open_count-1,open_id),stack+temp_lineclear))
		if len(ttemp) > 4:
			arg = np.argsort([gridScore(t[0],neurals,curval) for t in ttemp])
			arg = arg[-4:]
			temp += [ttemp[i] for i in arg]
		else:
			temp += ttemp
	return temp#갯수제한
def terminalGrids(f_state,depth,neurals,curval):#CHECKED! #return [possible_grids],leng,doom

	pos_nextBlocks = possibleWays(f_state.blocks,depth)
	best_leng = -1
	temp=[]
	for pway in pos_nextBlocks:
		
		leng = 0
		gen_0 = [(f_state.grid,f_state.opening,f_state.stack)]
		while leng < len(pway[0]):
			block = pway[0][leng]
			gen_1 = nextGrids(gen_0,block,neurals,curval)
			if len(gen_1) == 0:
				break
			gen_0 = gen_1
			leng+=1
		if leng > best_leng:
			best_leng = leng
			temp = gen_0
		elif leng == best_leng:
			temp += gen_0
	return (temp,best_leng)
def Final_terminals(terminals,neurals,curval):#CHECKED!
	#terminals =[index,[grid,opening]]
	idScore = []
	for terminal in terminals:
		grids = terminal[1]
		bestscore = max([gridScore(g[0],neurals,curval) for g in grids])
		idScore.append(  (terminal[0],bestscore)  )
	who = -1
	bsc = -100
	for ele in idScore:
		if bsc < ele[1]:
			bsc = ele[1]
			who = ele[0]
	return who
def bestMove(qu,f_state,neurals,depth,curval):#CHECKED!
	possible_nextmoves = candidates(f_state,neurals,curval) #[ (한 단계 뒤의 f_state, 그때의 결과 및 커맨드) ] 리턴
	terminals = []
	best_leng = -1
	if len(possible_nextmoves) > 0:
		for move_idx,thismove in enumerate(possible_nextmoves):
			thisterm = terminalGrids(thismove[0],depth,neurals,curval) #terminalGrids 는 [가능한 그리드,오프닝],수명 리턴
			if best_leng < thisterm[1]:
				terminals = [(move_idx,thisterm[0])]
				best_leng = thisterm[1]
			elif best_leng == thisterm[1]:
				terminals += [(move_idx,thisterm[0])]
		who = Final_terminals(terminals,neurals,curval)
		qu.put( (possible_nextmoves[who][1],best_leng+1) )
	else:
		qu.put( (None,0) )

############################################################################################################################
def next_r_grids(adj_m,grids,block,stack):
	temp = []
	for grid in grids:
		for s in range(0,4):
			if s < stack:
				for i in adj_m[grid][block][s]:
					temp.append(i)
	return temp
def next_r_states(adj_m,r_state):
	state = r_state.state
	blocks = r_state.blocks
	stack = r_state.stack
	temp = []
	if blocks[1] == -1:
		newblocks = (blocks[0][1:],-1)
		temp+=[Rstate(ng,newblocks,stack-1) for ng in next_r_grids(adj_m,[state],blocks[0][0],stack)]
		newblocks = (blocks[0][2:],blocks[0][0])
		temp+=[Rstate(ng,newblocks,stack-1) for ng in next_r_grids(adj_m,[state],blocks[0][1],stack)]
	else:
		newblocks = (blocks[0][1:],blocks[1])
		temp+=[Rstate(ng,newblocks,stack-1) for ng in next_r_grids(adj_m,[state],blocks[0][0],stack)]
		newblocks = (blocks[0][1:],blocks[0][0])
		temp+=[Rstate(ng,newblocks,stack-1) for ng in next_r_grids(adj_m,[state],blocks[1],stack)]
	return temp
def bestMove_from_Rstate(adj_m,init_r_state,state_score):
	nrStates = next_r_states(adj_m,init_r_state)
	best_leng = -1
	temp=[]
	if len(nrStates) > 0:
		for next_Rstate in nrStates:
			blocks = next_Rstate.blocks
			
			pos_nextBlocks = possibleWays(blocks,4)

			for pway in pos_nextBlocks:
				states_0 = [next_Rstate.state]
				leng = 0
				stack = next_Rstate.stack
				while leng < len(pway[0]):
					block = pway[0][leng]
					states_1 = next_r_grids(adj_m,states_0,block,stack)
					if len(states_1) == 0:
						break
					states_0 = states_1
					leng += 1
					stack -= 1
				if leng >= best_leng:
					
					sh_id = pway[1]
					if sh_id == -1:
						sh_id = 7
					pwayscore = max([state_score[state*8+sh_id] for state in states_0 ])
					if leng > best_leng:
						temp = [(pwayscore,next_Rstate)]
					else:
						temp += [(pwayscore,next_Rstate)]
					best_leng = leng
		armax = np.argmax( [t[0] for t in temp])
		return temp[armax][1],best_leng+1
	else:
		return  None,0




def input_value(inpt,neurals):
	weights = neurals[0]
	biases = neurals[1]
	a1 = np.maximum(0,np.add(np.matmul(inpt,weights[0]),biases[0]))
	a2 = np.maximum(0,np.add(np.matmul(a1,weights[1]),biases[1]))
	a3 = np.add(np.matmul(a2,weights[2]),biases[2])

	return a3[0]


def Rstate_init_data(iter):
	curval = {'000':[0,1,4,5,6],'001':[0,1,2,3,4,5,6],'002':[0,1,5,6],'003':[0,1,5,6],'004':[0,1,5,6],
			  '010':[0,2,3,4],'011':[0,1,2,4,5,6],'012':[0,2,4],'013':[0,2,4,6],'014':[0,2,4],
			  '020':[0,5,6],'021':[0,3,4,6],'022':[0,1,6],'023':[0,6],'024':[0,6],
			  '030':[0],'031':[0,5],'032':[0],'033':[0],'034':[0],
			  '040':[0],'041':[0],'042':[0],'043':[0],'044':[0],
			  '100':[0,1,2,3,4,5,6],'101':[0,2,3,4],'102':[0,3,4,6],'103':[0,3,4],'104':[0,3,4],
			  '110':[0,1,3,4,5,6],'120':[0,2,4,6],'130':[0,6],'140':[0],
			  '200':[0,1,5,6],'201':[0,2,4,5],'202':[0,5,6],'203':[0,5],'204':[0,5],
			  '210':[0,3,4],'220':[0,1,5],'230':[0],'240':[0],
			  '300':[0,1,5,6],'301':[0,2,4],'302':[0,6],'303':[0],'304':[0],
			  '310':[0,3,4,5],'320':[0,5],'330':[0],'340':[0],
			  '400':[0,1,5,6],'401':[0,2,4],'402':[0,6],'403':[0],'404':[0],
			  '410':[0,3,4],'420':[0,5],'430':[0],'440':[0]
			  }
	adjMtrix = [[[0,7],[],[],[11],[3],[5],[1]],
			[[1,6],[],[10],[],[2],[0],[4]],
			[[2],[13],[0],[],[0],[20],[10,17,12]],
			[[3],[12],[],[1],[1],[11,16,13],[21]],
			[[4],[13],[26,24],[0,19],[0,17],[12],[]],
			[[5],[12],[1,18],[27,25],[1,16],[],[13]],
			[[6],[],[],[],[14],[16,2],[2,16]],
			[[7],[],[],[],[15],[3,17],[17,3]],
			[[8],[1],[22],[],[15],[],[]],
			[[9],[0],[],[23],[14],[],[]],
			[[10],[1],[],[],[],[],[15]],
			[[11],[0],[],[],[],[14],[]],
			[[12],[],[3],[],[1,11],[],[9,0]],
			[[13],[],[],[2],[0,10],[8,1],[]],
			[[14],[],[],[],[],[24,0],[13]],
			[[15],[],[],[],[],[12],[25,1]],
			[[16],[0],[],[13],[19,13],[17,10,12],[18]],
			[[17],[1],[12],[],[18,12],[19],[16,11,13]],
			[[18],[],[],[1],[],[],[]],
			[[19],[],[0],[],[],[],[]],
			[[20],[],[1],[],[10],[],[2,0]],
			[[21],[],[],[0],[11],[3,1],[]],
			[[22],[],[],[],[],[0],[23]],
			[[23],[],[],[],[],[22],[1]],
			[[24],[],[],[],[0],[],[14]],
			[[25],[],[],[],[1],[15],[]],
			[[26],[],[],[],[],[1],[]],
			[[27],[],[],[],[],[],[0]]
			]
	adjMtrix_hd =  [ [ [ [7    ],[0    ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [11   ],[     ],[     ],[     ] ],  [ [3    ],[     ],[     ],[     ] ],  [ [5    ],[     ],[     ],[     ] ],  [ [1    ],[     ],[     ],[     ] ] ],#0 
			  [ [ [6    ],[1    ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [10   ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [2    ],[     ],[     ],[     ] ],  [ [0    ],[     ],[     ],[     ] ],  [ [4    ],[     ],[     ],[     ] ] ],#1 
			  [ [ [     ],[     ],[2    ],[     ] ],  [ [13   ],[     ],[     ],[     ] ],  [ [0    ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[0    ],[     ],[     ] ],  [ [     ],[18   ],[     ],[     ] ],  [ [17   ],[10,12],[     ],[     ] ] ],#2 
			  [ [ [     ],[     ],[3    ],[     ] ],  [ [12   ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [1    ],[     ],[     ],[     ] ],  [ [     ],[1    ],[     ],[     ] ],  [ [16   ],[11,13],[     ],[     ] ],  [ [     ],[19   ],[     ],[     ] ] ],#3 
			  [ [ [     ],[     ],[4    ],[     ] ],  [ [     ],[13   ],[     ],[     ] ],  [ [     ],[22    ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[17   ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ] ],#4 
			  [ [ [     ],[     ],[5    ],[     ] ],  [ [     ],[12   ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[23   ],[     ],[     ] ],  [ [     ],[16   ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ] ],#5 
			  [ [ [     ],[     ],[     ],[6    ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [14   ],[     ],[     ],[     ] ],  [ [16   ],[2    ],[     ],[     ] ],  [ [2    ],[16   ],[     ],[     ] ] ],#6 
			  [ [ [     ],[     ],[     ],[7    ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [15   ],[     ],[     ],[     ] ],  [ [3    ],[17   ],[     ],[     ] ],  [ [17   ],[3    ],[     ],[     ] ] ],#7 
			  [ [ [     ],[     ],[8    ],[     ] ],  [ [     ],[1    ],[     ],[     ] ],  [ [     ],[20    ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[15   ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ] ],#8 
			  [ [ [     ],[     ],[9    ],[     ] ],  [ [     ],[0    ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[21   ],[     ],[     ] ],  [ [     ],[14   ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ] ],#9 
			  [ [ [     ],[     ],[10   ],[     ] ],  [ [1    ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [15   ],[     ],[     ],[     ] ] ],#10
			  [ [ [     ],[     ],[11   ],[     ] ],  [ [0    ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [14   ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ] ],#11
			  [ [ [     ],[12   ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [3    ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [1 ,11],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [9 ,0 ],[     ],[     ],[     ] ] ],#12
			  [ [ [     ],[13   ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [2    ],[     ],[     ],[     ] ],  [ [0 ,10],[     ],[     ],[     ] ],  [ [8 ,1 ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ] ],#13
			  [ [ [     ],[     ],[14   ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[22,0 ],[     ],[     ] ],  [ [     ],[13   ],[     ],[     ] ] ],#14
			  [ [ [     ],[     ],[15   ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[12   ],[     ],[     ] ],  [ [     ],[23,1 ],[     ],[     ] ] ],#15
			  [ [ [     ],[     ],[16   ],[     ] ],  [ [0    ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [13   ],[     ],[     ],[     ] ],  [ [     ],[13   ],[     ],[     ] ],  [ [10   ],[17,12],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ] ],#16
			  [ [ [     ],[     ],[17   ],[     ] ],  [ [1    ],[     ],[     ],[     ] ],  [ [12   ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[12   ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [11   ],[16,13],[     ],[     ] ] ],#17
			  [ [ [     ],[     ],[18   ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[10   ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[2 ,0 ],[     ],[     ] ] ],#18
			  [ [ [     ],[     ],[19   ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[11   ],[     ],[     ] ],  [ [     ],[3 ,1 ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ] ],#19
			  [ [ [     ],[     ],[20   ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[0    ],[     ],[     ] ],  [ [     ],[21   ],[     ],[     ] ] ],#20
			  [ [ [     ],[     ],[21   ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[20   ],[     ],[     ] ],  [ [     ],[1    ],[     ],[     ] ] ],#21
			  [ [ [     ],[     ],[22   ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[0    ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[14   ],[     ],[     ] ] ],#22
			  [ [ [     ],[     ],[23   ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[1    ],[     ],[     ] ],  [ [     ],[15   ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ] ]#23
		    ] 
	adj_m = adjMtrix_hd
	state_score = np.random.rand(len(adj_m)*8)
	rslt = []
	for i in range(0,24):
		success = 0
		for it in range(0,iter):
			block = newmap(1)
			shift = np.random.randint(7)
			testee = Rstate(i,(block,shift),9999)
			a,b = bestMove_from_Rstate(adj_m,testee,state_score)
			if b == 5:
				success+=1
		rslt.append(success/iter)
		print(i,success/iter)
	with open('tetris_ai/wir/rinit.pickle', 'wb') as f:
		pickle.dump(rslt, f)
def f2r_input(stack,max_h,now_grid_score,f_life,doom,secured_combo,secured,now_rinit):
	temp = []
	temp.append((stack-6)/6)
	heightlimit = maximum_height#최대높이
	heightlimit *= 0.5
	temp.append((max_h-heightlimit)/heightlimit) 
	temp.append((now_grid_score-0.09)/0.1)
	temp.append((f_life-3)/2)
	if doom:
		temp.append(1)
	else:
		temp.append(-1)
	temp.append((secured_combo-2.5)/2.5)
	if secured:
		temp.append(1)
	else:
		temp.append(-1)
	temp.append((now_rinit-0.5)/0.3)
	return temp
def r2f_input(stack,max_h,now_grid_score,f_life,doom,secured_combo,secured,combo):
	temp = []
	temp.append((stack-6)/6)
	heightlimit = maximum_height#최대높이
	heightlimit *= 0.5
	temp.append((max_h-heightlimit)/heightlimit) 
	temp.append((now_grid_score-0.09)/0.1)
	temp.append((f_life-3)/2)
	if doom:
		temp.append(1)
	else:
		temp.append(-1)
	temp.append((secured_combo-2.5)/2.5)
	if secured:
		temp.append(1)
	else:
		temp.append(-1)
	temp.append((combo-7)/7)
	return temp
def Init_Neural(l_w,init_w,init_b):
	temp_w=[]
	temp_b=[]
	temp_w.append((np.random.rand(l_w[0],l_w[1])-0.5)*init_w[0])
	temp_w.append((np.random.rand(l_w[1],l_w[2])-0.5)*init_w[1])
	temp_w.append((np.random.rand(l_w[2],1)-0.5)*init_w[2])
	temp_b.append((np.random.rand(l_w[1])-0.5)*init_b[0])
	temp_b.append((np.random.rand(l_w[2])-0.5)*init_b[1])
	temp_b.append((np.random.rand(1)-0.5)*init_b[2])
	return (temp_w,temp_b)
def Mut_Neural(neural,l_w,mut_w,mut_b):
	temp_w=[]
	temp_b=[]
	temp_w.append(np.add(neural[0][0],(np.random.rand(l_w[0],l_w[1])-0.5)*mut_w[0]))
	temp_w.append(np.add(neural[0][1],(np.random.rand(l_w[1],l_w[2])-0.5)*mut_w[1]))
	temp_w.append(np.add(neural[0][2],(np.random.rand(l_w[2],1)-0.5)*mut_w[2]))
	temp_b.append(np.add((np.random.rand(l_w[1])-0.5)*mut_b[0],neural[1][0]))
	temp_b.append(np.add((np.random.rand(l_w[2])-0.5)*mut_b[1],neural[1][1]))
	temp_b.append(np.add((np.random.rand(1)-0.5)*mut_b[2],neural[1][2]))
	return (temp_w,temp_b)

def ralph_move_key(r0,r1):
	block = r0.blocks[0][0]
	hold = False
	if r1.blocks[1] != r0.blocks[1]:
		block = r0.blocks[1]
		hold = True
	state0 = r0.state
	state1 = r1.state
	string = str(block)
	if state0 < 10:
		string+="0"
	string+=str(state0)
	if state1 < 10:
		string+="0"
	string+=str(state1)
	return string,hold,block
def shiftCmd():
	keyboard.press(Key.shift)
	time.sleep(0.07)
	keyboard.release(Key.shift)
	time.sleep(0.07)
def eval_coop(adj_m,curval,rinit_score,felix_neurals,ralph_neurals,f2r_neurals,r2f_neurals,r_cmd):#f2r_score,r2f_score):
	linesent=0
	lifelong=0
	hasHolden = False
	hasOpened = False
	q = queue.Queue()
	
	#INITIAL SETTINGS
	xxx,yyy = framePos()
	initial_map = getInitmap(xxx,yyy)
	print("------------------------------------------GAME START DETECTED")
	print("INITIAL MAP",initial_map)
	felixDown =False
	ralphDown =False
	opened_map,opening = map2opening(initial_map)
	opened_blockState = (opened_map,-1)
	opened_grid = basic_grid(opening[0])
	stack = 0
	print("OPENED MAP",opened_map)
	print("OPENING",opening)
	f_state = Fstate(opened_grid,opened_blockState,stack,opening)
	pre_bm(q,f_state,felix_neurals,3,curval)
	nbnb = newblock(xxx,yyy)
	f_state.block_add_by_id(nbnb)
	now_r = OpeningBlock2State(opening[2])
	now_rinit = rinit_score[now_r]
	#print("initial_map",initial_map)
	#print("opening",f_state.opening)
		
	while True:
		#FELIX
		felixCount = 0

		if felixDown and ralphDown:
			break
		while True:
			now_grid = f_state.grid
			now_blocks = f_state.blocks
			now_gridScore = gridScore(now_grid,felix_neurals,curval)
			now_maxHeight = max(wavAdj(statewav(now_grid)))+f_state.stack
			now_stack = f_state.stack
			#print(f_state.grid)
			#print("Now R",now_r)
			#print(now_blocks)
			#print("now_maxHeight",now_maxHeight,"now_stack",now_stack)
			
			now_r_state = Rstate(now_r,now_blocks,now_stack)
			next_r_state,secured_combo = bestMove_from_Rstate(adj_m,now_r_state,ralph_neurals)
			secured = (secured_combo == 4)
			nextMove,felix_life = q.get()	

			nbnb = f_state.blocks[0][-1]

			doom = (felix_life != 4)
			f2r_inpt = f2r_input(now_stack,now_maxHeight,now_gridScore,felix_life,doom,secured_combo,secured,now_rinit)
			f2r_score = input_value(f2r_inpt,f2r_neurals)
			#print("felix_life",felix_life)
			#print("ralph_life",secured_combo)
			#print("f2r_score",f2r_score)
			felixDown =False
			f2r_score += (2 * mouse.position[0]/1920 - 1)*handle_sensivity
			if felix_life is 0 or (f2r_score > 0 and now_stack > minimum_attack and felixCount > 0): #최소공격

				if felix_life is 0:
					felixDown =True
				q.put((nextMove,felix_life))
				print("################################# COMBO MODE")
				break
			else:
				lifelong+=1

			nextMove[0].block_add_by_id(nbnb)
			move,rot,walling = felix_move(nextMove[0].opening[0],nextMove[1],nextMove[2],nextMove[3])
			hold = nextMove[4]
			if f_state.opening[1] is 0 and not hasOpened:
				aa,bb =  opening_cmd(f_state.opening[2])
				print("PUT OPENING BLOCK") 
				hasOpened = True
				MoveByCmd(aa,bb,False,False)
				nextMove[0].block_add(xxx,yyy)
				nextMove[0].opening = (f_state.opening[0],-1,f_state.opening[2])
			if hold : 
				shiftCmd()

				if not hasHolden:
					hasHolden = True
					nextMove[0].block_add(xxx,yyy)
			if f_state.opening[1] is 1 and hold and not hasOpened:
				aa,bb =  opening_cmd(f_state.opening[2])
				print("PUT OPENING BLOCK")
				hasOpened = True
				MoveByCmd(aa,bb,False,False)
				nextMove[0].block_add(xxx,yyy)
				nextMove[0].opening = (f_state.opening[0],-1,f_state.opening[2])
				

			print("----------------------------------")
			
			
			next_f_state = nextMove[0]
			
			f_state = next_f_state
			felixCount += 1
			print("NEXT BLOCKS",f_state.blocks,"-")
			pre_bm(q,f_state,felix_neurals,3,curval)
			MoveByCmd(move,rot,False,walling)
			nbnb = newblock(xxx,yyy)
			f_state.block_add_by_id(nbnb)
		#RALPH
		combo = 0
		if felixDown and ralphDown:
			break

		while True:

			now_stack = now_r_state.stack
			now_maxHeight = max(wavAdj(statewav(now_grid)))+now_r_state.stack

			next_r_state,ralph_life = bestMove_from_Rstate(adj_m,now_r_state,ralph_neurals)
			secured = (ralph_life == 4)
			
			nextMove,felix_life = q.get()													
			doom = (felix_life != 4)
			
			r2f_inpt = r2f_input(now_stack,now_maxHeight,now_gridScore,felix_life,doom,ralph_life,secured,combo)
			r2f_score = input_value(r2f_inpt,r2f_neurals)

			ralphDown =False
			if ralph_life == 0 or (combo > 0 and r2f_score > 0):

				if ralph_life == 0:
					ralphDown =True	
				print("################################# BUILD MODE")
				break
			else:
				lifelong+=1

			
			f_state = Fstate(now_grid,next_r_state.blocks,next_r_state.stack,opening)
			cmd_key,hold,bb = ralph_move_key(now_r_state,next_r_state)	
			cmd = r_cmd[cmd_key] 
			print("HANDLING BLOCK",bb)
			print("STATE SHIFT: [",now_r_state.state,"->",next_r_state.state,"]")
			move,rot,walling = ralph_move(opening[0],bb,cmd[1],cmd[0])
			if hold : 
				shiftCmd()

				if not hasHolden:
					hasHolden = True
					next_r_state.block_add(xxx,yyy)
			
			print("--------------------------------")
			
			if combo > 0:
				next_r_state.block_add(xxx,yyy)
			now_r_state = next_r_state
			combo += 1
			linesent += int(combo/2)
			
			pre_bm(q,f_state,felix_neurals,3,curval)
			MoveByCmd(move,rot,True,walling)

		f_state = Fstate(now_grid,now_r_state.blocks,now_stack,opening)
		
		
		
		f_state.block_add(xxx,yyy)

		pre_bm(q,Fstate(f_state.grid,(f_state.blocks[0][:-1],f_state.blocks[1]),f_state.stack,f_state.opening),felix_neurals,3,curval)
		now_r = now_r_state.state
	print("GAMEOVER")
	print("LINESENT:",linesent)
	print("LIFELONG:",lifelong)
	

def pre_bm(qu,fstate,fneural,depth,cv):
	t = threading.Thread( target = bestMove, args = (qu,fstate,fneural,depth,cv))
	t.start()
############################################################################################################################

mouse = pynput.mouse.Controller()
dc= windll.user32.GetDC(0)
keyboard = Controller()

#####################################################
maximum_height = 18
minimum_attack= 5 
handle_sensivity = 0.75
#####################################################
anchor_x = 119
anchor_y = 735
ghost_x = 282
ghost_y = 625
block_x = (446,445,445,445,445)
block_y = (336,402,457,509,561)
colors = ( (16765485,4185599,7556604,3073166,12669921,3708671,15891275),
		   (16765485,4185599,7556604,3073166,12669921,3708671,15891275),
		   (16765485,4185599,7556604,3073166,12669921,3708671,15891275),
		   (16765485,4185599,7556604,3073166,12669921,3708671,15891275),
		   (16765485,4185599,7556604,3073166,12669921,3708671,15891275)
		  )
if True:
	curval = {'000':[0,1,4,5,6],'001':[0,1,2,3,4,5,6],'002':[0,1,5,6],'003':[0,1,5,6],'004':[0,1,5,6],
			  '010':[0,2,3,4],'011':[0,1,2,4,5,6],'012':[0,2,4],'013':[0,2,4,6],'014':[0,2,4],
			  '020':[0,5,6],'021':[0,3,4,6],'022':[0,1,6],'023':[0,6],'024':[0,6],
			  '030':[0],'031':[0,5],'032':[0],'033':[0],'034':[0],
			  '040':[0],'041':[0],'042':[0],'043':[0],'044':[0],
			  '100':[0,1,2,3,4,5,6],'101':[0,2,3,4],'102':[0,3,4,6],'103':[0,3,4],'104':[0,3,4],
			  '110':[0,1,3,4,5,6],'120':[0,2,4,6],'130':[0,6],'140':[0],
			  '200':[0,1,5,6],'201':[0,2,4,5],'202':[0,5,6],'203':[0,5],'204':[0,5],
			  '210':[0,3,4],'220':[0,1,5],'230':[0],'240':[0],
			  '300':[0,1,5,6],'301':[0,2,4],'302':[0,6],'303':[0],'304':[0],
			  '310':[0,3,4,5],'320':[0,5],'330':[0],'340':[0],
			  '400':[0,1,5,6],'401':[0,2,4],'402':[0,6],'403':[0],'404':[0],
			  '410':[0,3,4],'420':[0,5],'430':[0],'440':[0]
			  }
	adjMtrix_hd =  [ [ [ [7    ],[0    ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [11   ],[     ],[     ],[     ] ],  [ [3    ],[     ],[     ],[     ] ],  [ [5    ],[     ],[     ],[     ] ],  [ [1    ],[     ],[     ],[     ] ] ],#0 
			  [ [ [6    ],[1    ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [10   ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [2    ],[     ],[     ],[     ] ],  [ [0    ],[     ],[     ],[     ] ],  [ [4    ],[     ],[     ],[     ] ] ],#1 
			  [ [ [     ],[     ],[2    ],[     ] ],  [ [13   ],[     ],[     ],[     ] ],  [ [0    ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[0    ],[     ],[     ] ],  [ [     ],[18   ],[     ],[     ] ],  [ [17   ],[10,12],[     ],[     ] ] ],#2 
			  [ [ [     ],[     ],[3    ],[     ] ],  [ [12   ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [1    ],[     ],[     ],[     ] ],  [ [     ],[1    ],[     ],[     ] ],  [ [16   ],[11,13],[     ],[     ] ],  [ [     ],[19   ],[     ],[     ] ] ],#3 
			  [ [ [     ],[     ],[4    ],[     ] ],  [ [     ],[13   ],[     ],[     ] ],  [ [     ],[22    ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[17   ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ] ],#4 
			  [ [ [     ],[     ],[5    ],[     ] ],  [ [     ],[12   ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[23   ],[     ],[     ] ],  [ [     ],[16   ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ] ],#5 
			  [ [ [     ],[     ],[     ],[6    ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [14   ],[     ],[     ],[     ] ],  [ [16   ],[2    ],[     ],[     ] ],  [ [2    ],[16   ],[     ],[     ] ] ],#6 
			  [ [ [     ],[     ],[     ],[7    ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [15   ],[     ],[     ],[     ] ],  [ [3    ],[17   ],[     ],[     ] ],  [ [17   ],[3    ],[     ],[     ] ] ],#7 
			  [ [ [     ],[     ],[8    ],[     ] ],  [ [     ],[1    ],[     ],[     ] ],  [ [     ],[20    ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[15   ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ] ],#8 
			  [ [ [     ],[     ],[9    ],[     ] ],  [ [     ],[0    ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[21   ],[     ],[     ] ],  [ [     ],[14   ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ] ],#9 
			  [ [ [     ],[     ],[10   ],[     ] ],  [ [1    ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [15   ],[     ],[     ],[     ] ] ],#10
			  [ [ [     ],[     ],[11   ],[     ] ],  [ [0    ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [14   ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ] ],#11
			  [ [ [     ],[12   ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [3    ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [1 ,11],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [9 ,0 ],[     ],[     ],[     ] ] ],#12
			  [ [ [     ],[13   ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [2    ],[     ],[     ],[     ] ],  [ [0 ,10],[     ],[     ],[     ] ],  [ [8 ,1 ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ] ],#13
			  [ [ [     ],[     ],[14   ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[22,0 ],[     ],[     ] ],  [ [     ],[13   ],[     ],[     ] ] ],#14
			  [ [ [     ],[     ],[15   ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[12   ],[     ],[     ] ],  [ [     ],[23,1 ],[     ],[     ] ] ],#15
			  [ [ [     ],[     ],[16   ],[     ] ],  [ [0    ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [13   ],[     ],[     ],[     ] ],  [ [     ],[13   ],[     ],[     ] ],  [ [10   ],[17,12],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ] ],#16
			  [ [ [     ],[     ],[17   ],[     ] ],  [ [1    ],[     ],[     ],[     ] ],  [ [12   ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[12   ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [11   ],[16,13],[     ],[     ] ] ],#17
			  [ [ [     ],[     ],[18   ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[10   ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[2 ,0 ],[     ],[     ] ] ],#18
			  [ [ [     ],[     ],[19   ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[11   ],[     ],[     ] ],  [ [     ],[3 ,1 ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ] ],#19
			  [ [ [     ],[     ],[20   ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[0    ],[     ],[     ] ],  [ [     ],[21   ],[     ],[     ] ] ],#20
			  [ [ [     ],[     ],[21   ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[20   ],[     ],[     ] ],  [ [     ],[1    ],[     ],[     ] ] ],#21
			  [ [ [     ],[     ],[22   ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[0    ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[14   ],[     ],[     ] ] ],#22
			  [ [ [     ],[     ],[23   ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[      ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ],  [ [     ],[1    ],[     ],[     ] ],  [ [     ],[15   ],[     ],[     ] ],  [ [     ],[     ],[     ],[     ] ] ]#23
		    ] 
	ralph_cmd= {"00000":(0,0),"00007":(3,1),"30011":(2,1),"40003":(2,3),"50005":(2,3),"60001":(1,2),
		"00101":(0,0),"00106":(0,1),"20110":(0,1),"40102":(0,1),"50100":(0,2),"60104":(0,1),
		"00202":(0,0),"10213":(2,0),"20200":(1,0),"40200":(1,2),"50218":(1,0),"60210":(1,0),"60217":(2,3),"60212":(1,2),
		"00303":(0,0),"10312":(0,0),"30301":(0,0),"40301":(0,2),"50311":(0,0),"50316":(0,1),"50313":(0,2),"60319":(0,0),
		"00404":(0,0),"10413":(2,0),"20422":(2,1),"40417":(2,3),
		"00505":(0,0),"10512":(0,0),"30523":(0,1),"40516":(0,1),
		"00606":(0,0),"40614":(1,0),"50616":(1,0),"50602":(1,2),"60602":(1,0),"60616":(1,2),
		"00707":(0,0),"40715":(0,0),"50703":(0,0),"50717":(0,2),"60717":(0,0),"60703":(0,2),
		"00808":(0,0),"10801":(2,0),"20820":(2,1),"40815":(2,3),
		"00909":(0,0),"10900":(0,0),"30921":(0,1),"40914":(0,1),
		"01010":(0,0),"11001":(2,0),"61015":(2,3),
		"01111":(0,0),"11100":(0,0),"51114":(0,1),
		"01212":(0,0),"21203":(2,1),"41201":(1,2),"41211":(2,1),"61209":(2,1),"61200":(0,2),
		"01313":(0,0),"31302":(0,1),"41300":(0,2),"41310":(0,3),"51308":(0,3),"51301":(1,2),
		"01414":(0,0),"51422":(1,0),"51400":(1,2),"61413":(1,2),
		"01515":(0,0),"51512":(0,2),"61523":(0,0),"61501":(0,2),
		"01616":(0,0),"11600":(1,0),"31613":(1,0),"41613":(1,2),"51617":(1,0),"51610":(1,1),"51612":(1,2),
		"01717":(0,0),"11701":(1,0),"21712":(0,0),"41712":(0,2),"61716":(0,0),"61711":(1,3),"61713":(0,2),
		"01818":(0,0),"41810":(0,0),"61802":(0,0),"61800":(0,2),
		"01919":(0,0),"41911":(1,0),"51903":(1,0),"51901":(1,2),
		"02020":(0,0),"52000":(0,2),"62021":(0,0),
		"02121":(0,0),"52120":(1,0),"62101":(1,2),
		"02222":(0,0),"42200":(0,2),"62214":(0,0),
		"02323":(0,0),"42301":(1,2),"52315":(1,0)
		}
	adj_m = adjMtrix_hd
	with open('tetris_ai/wir/player/rinit.pickle', 'rb') as f:
		rinit = pickle.load(f)
	with open('tetris_ai/wir/player/ralph.pickle', 'rb') as f:
		ralph = pickle.load(f)
	with open('tetris_ai/wir/player/felix.pickle', 'rb') as f:
		felix = pickle.load(f)
	with open('tetris_ai/wir/player/coop.pickle', 'rb') as f:
		coop = pickle.load(f)
	eval_coop(adj_m,curval,rinit,felix[-1],ralph[-1],coop[0][0][-1],coop[0][1][-1],ralph_cmd)