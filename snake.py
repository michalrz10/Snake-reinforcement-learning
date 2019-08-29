import tensorflow as tf
import pygame
import sys
import random
import matplotlib.pyplot as plt
import numpy as np
import os
from time import sleep

class Component:
	def __init__(self,x,y):
		self.x=x
		self.y=y
	
	def getx(self): 
		return self.x
	def gety(self): 
		return self.y
	def setx(self,x):
		self.x=x
	def sety(self,y):
		self.y=y
		
class Model:
	def __init__(self):
		self.x=tf.placeholder(tf.float32,[None,5,5,4])
		self.cy=tf.placeholder(tf.float32)
		self.ay=tf.placeholder(tf.float32)
		self.akcja=tf.placeholder(tf.int32)
		self.c1=tf.layers.conv2d(self.x,32,4,1,'same',activation=tf.nn.relu)
		self.c2=tf.layers.conv2d(self.c1,256,2,1,'same',activation=tf.nn.relu)
		self.aflat=tf.layers.flatten(self.c2)
		self.cflat=tf.layers.flatten(self.c2)
		#-------------------------------
		self.cl1=tf.layers.dense(self.cflat, 256,tf.nn.relu)
		self.cl2=tf.layers.dense(self.cl1, 32,tf.nn.relu)
		self.cout=tf.layers.dense(self.cl2,4)
		self.cvalue=tf.squeeze(self.cout)
		self.closss=tf.losses.mean_squared_error(self.cy,self.cout)
		self.ctra=tf.train.AdamOptimizer(0.00002).minimize(self.closss)
		
		self.al1=tf.layers.dense(self.aflat, 256,tf.nn.relu)
		self.al2=tf.layers.dense(self.al1, 32,tf.nn.relu)
		self.aout=tf.layers.dense(self.al2,4)
		self.prob=tf.squeeze(tf.nn.softmax(self.aout))
		self.wyp=tf.gather(self.prob,self.akcja)
		self.alosss=-tf.log(self.wyp)*self.ay
		self.atra=tf.train.AdamOptimizer(0.00001).minimize(self.alosss)
		
		self.sess=tf.Session()
		self.saver = tf.train.Saver()
		#self.saver.restore(self.sess, './snake')
		self.sess.run(tf.global_variables_initializer())
	def	action(self,tab):
		return self.sess.run(self.prob,{self.x:[tab]})
	def predict(self,tab):
		return self.sess.run(self.cvalue,{self.x:[tab]})
	def alern(self,tab,outt,akcja):
		_, lo=self.sess.run([self.atra,self.alosss],{self.x:[tab], self.ay:outt, self.akcja:akcja})
		return lo
	def clern(self,tab,outt):
		_, lo=self.sess.run([self.ctra,self.closss],{self.x:[tab], self.cy:outt})
		return lo
class Snake:
	def __init__(self,points):
		self.start=False
		self.pp=points
		self.pos=[]
		self.side=0
		self.apple=[0,0]
		self.points=0
		self.reset()
		self.reward=0
		self.pp.clear()
	def reset(self):
		self.start=False
		self.pp.append(self.points)
		self.points=0
		self.pos.clear()
		self.pos.append(Component(random.randint(0,4),random.randint(0,4)))
		self.add()
		self.setapple()
		self.reward=-10
	def add(self):
		self.pos.append(Component(-1,-1))
	def point(self):
		self.points+=1
		self.reward=10
		self.add()
		if len(self.pos)==25:
			reset()
			self.reward=10
		else:
			self.setapple()
	def getpoints(self):
		return self.points
	def setapple(self):
		czy=True
		while czy:
			czy=False
			self.apple[0]=random.randint(0,4)
			self.apple[1]=random.randint(0,4)
			for i in self.pos:
				if i.getx()==self.apple[0] and i.gety()==self.apple[1]:
					czy=True
					break
		
	def getapple(self):
		return self.apple
	def setside(self,side):
		self.side=side
		self.start=True
	def move(self):
		if self.start:
			self.reward=-0.1
			if len(self.pos)==2:
				if self.side==0 and self.pos[0].getx()==self.pos[1].getx() and self.pos[0].gety()-1==self.pos[1].gety():
					self.reset()
					return True
				elif self.side==1 and self.pos[0].getx()+1==self.pos[1].getx() and self.pos[0].gety()==self.pos[1].gety():
					self.reset()
					return True
				elif self.side==2 and self.pos[0].getx()==self.pos[1].getx() and self.pos[0].gety()+1==self.pos[1].gety():
					self.reset()
					return True
				elif self.side==3 and self.pos[0].getx()-1==self.pos[1].getx() and self.pos[0].gety()==self.pos[1].gety():
					self.reset()
					return True
			
			for i in range(len(self.pos)-1,0,-1):
				self.pos[i].setx(self.pos[i-1].getx())
				self.pos[i].sety(self.pos[i-1].gety())
			if self.side==0:
				if self.pos[0].gety()==0:
					self.reset()
					return True
				else:
					self.pos[0].sety(self.pos[0].gety()-1)
			elif self.side==1:
				if self.pos[0].getx()==4:
					self.reset()
					return True
				else:
					self.pos[0].setx(self.pos[0].getx()+1)
			elif self.side==2:
				if self.pos[0].gety()==4:
					self.reset()
					return True
				else:
					self.pos[0].sety(self.pos[0].gety()+1)
			else:
				if self.pos[0].getx()==0:
					self.reset()
					return True
				else:
					self.pos[0].setx(self.pos[0].getx()-1)
			if self.pos[0].getx()==self.apple[0] and self.pos[0].gety()==self.apple[1]: self.point()
			for i in range(1,len(self.pos)):
				if self.pos[0].getx()==self.pos[i].getx() and self.pos[0].gety()==self.pos[i].gety():
					self.reset()
					return True
		return False
	def draw(self,window):
		for i in self.pos:
			pygame.draw.rect(window,(0,0,0),pygame.Rect(i.getx()*50,i.gety()*50,50,50))
		pygame.draw.rect(window,(255,0,0),pygame.Rect(self.apple[0]*50,self.apple[1]*50,50,50))

def game():
	pygame.init()
	window = pygame.display.set_mode((250, 250))
	pygame.display.set_caption(('Snake'))
	clock = pygame.time.Clock()
	points=[]
	snake=Snake(points)
	model=Model()
	end=True
	#pygame.time.set_timer(25,250)
	strata=[[],[]]
	gra=True
	wys=True
	tab=[]
	pom=[]
	
	for i in range(5):
		tab.append([])
		pom.append([])
		for _ in range(5):
			tab[-1].append([])
			tab[-1][-1].append(0)
			tab[-1][-1].append(0)
			tab[-1][-1].append(0)
			tab[-1][-1].append(0)
			pom[-1].append([])
			pom[-1][-1].append(0)
			pom[-1][-1].append(0)
			pom[-1][-1].append(0)
			pom[-1][-1].append(0)
	while True:
		for event in pygame.event.get():
			if event.type==pygame.QUIT:
				model.saver.save(model.sess, './snake', write_meta_graph=False)
				sys.exit(0)
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_q:
					plt.clf()
					plt.subplot(3,1,1)
					plt.plot(range(1,len(strata[0])+1),strata[0])
					plt.subplot(3,1,2)
					plt.plot(range(1,len(strata[1])+1),strata[1])
					plt.subplot(3,1,3)
					plt.plot(range(1,len(points)+1),points)
					plt.pause(0.0001)
				if event.key == pygame.K_w:	
					gra=not gra
					print('--------------------------\nShow game: '+str(gra)+'\nSlow: '+str(wys))
				if event.key == pygame.K_e:	
					wys=not wys
					print('--------------------------\nShow game: '+str(gra)+'\nSlow: '+str(wys))
				if event.key == pygame.K_s:
					model.saver.save(model.sess, './snake', write_meta_graph=False)
				
			'''if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_UP: snake.setside(0)
				elif event.key == pygame.K_DOWN: snake.setside(2)
				elif event.key == pygame.K_RIGHT: snake.setside(1)
				elif event.key == pygame.K_LEFT: snake.setside(3)
			if event.type==25:
				snake.move()'''

		#ai move
		for i in range(5): 
			for j in range(5):
				tab[i][j][3]=0
				tab[i][j][0]=0
				tab[i][j][2]=0
				tab[i][j][1]=0
		for i in snake.pos:
			if i.getx()!=-1 and i.gety()!=-1:
				tab[i.gety()][i.getx()][0]=1
		tab[snake.getapple()[1]][snake.getapple()[0]][3]=1
		tab[snake.pos[0].gety()][snake.pos[0].getx()][1]=1
		if len(snake.pos)==2:
			tab[snake.pos[0].gety()][snake.pos[0].getx()][2]=1
		else:
			tab[snake.pos[-1].gety()][snake.pos[-1].getx()][2]=1
		qwert=model.action(tab)
		akcja=np.random.choice(np.arange(len(qwert)),p=qwert)
		
		snake.setside(akcja)
		
		end=snake.move()
		
		
		if end:
			td_target=snake.reward
			td_error=td_target-model.predict(tab)
		else:
			for i in range(5): 
				for j in range(5):	
					pom[i][j][0]=0
					pom[i][j][2]=0
					pom[i][j][1]=0
					pom[i][j][3]=0
			for i in snake.pos:
				if i.getx()!=-1 and i.gety()!=-1:
					pom[i.gety()][i.getx()][0]=1
			pom[snake.getapple()[1]][snake.getapple()[0]][3]=1
			pom[snake.pos[0].gety()][snake.pos[0].getx()][1]=1
			if len(snake.pos)==2:
				pom[snake.pos[0].gety()][snake.pos[0].getx()][2]=1
			else:
				pom[snake.pos[-1].gety()][snake.pos[-1].getx()][2]=1
			
			td_target=snake.reward+0.95*model.predict(pom)
			td_error=td_target-model.predict(tab)
				
		strata[0].append(model.clern(tab,td_target))
		strata[1].append(model.alern(tab,td_error,akcja))
						
	
		if gra:	
			if wys: 
				sleep(0.2)
				print(qwert)
			window.fill((255,255,255))
			for i in range(4):
				pygame.draw.line(window,(0,0,0),(i*50+50,0),(i*50+50,250),1)
				pygame.draw.line(window,(0,0,0),(0,i*50+50),(250,i*50+50),1)
			snake.draw(window)
			pygame.display.flip()
			clock.tick(60)

game()	
