# -*- coding: utf-8 -*-



import tensorflow as tf
import numpy as np
import random
import math
import os



#------------------------------------------------------------
# 변수 설정
#------------------------------------------------------------
STONE_NONE = 0
STONE_PLAYER1 = 1
STONE_PLAYER2 = 2
STONE_MAX = 5

gridSize = 10
nbActions = gridSize * gridSize
nbStates = gridSize * gridSize
hiddenSize = 100
maxMemory = 500
batchSize = 50
epoch = 100
epsilonStart = 1
epsilonDiscount = 0.999
epsilonMinimumValue = 0.1
discount = 0.9
learningRate = 0.2
winReward = 1
#------------------------------------------------------------



#------------------------------------------------------------
# 가설 설정
#------------------------------------------------------------
X = tf.placeholder(tf.float32, [None, nbStates])
W1 = tf.Variable(tf.truncated_normal([nbStates, hiddenSize], stddev = 1.0 / math.sqrt(float(nbStates))))
b1 = tf.Variable(tf.truncated_normal([hiddenSize], stddev = 0.01))
input_layer = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.truncated_normal([hiddenSize, hiddenSize], stddev = 1.0 / math.sqrt(float(hiddenSize))))
b2 = tf.Variable(tf.truncated_normal([hiddenSize], stddev = 0.01))
hidden_layer = tf.nn.relu(tf.matmul(input_layer, W2) + b2)

W3 = tf.Variable(tf.truncated_normal([hiddenSize, nbActions], stddev = 1.0 / math.sqrt(float(hiddenSize))))
b3 = tf.Variable(tf.truncated_normal([nbActions], stddev = 0.01))
output_layer = tf.matmul(hidden_layer, W3) + b3

Y = tf.placeholder(tf.float32, [None, nbActions])
cost = tf.reduce_sum(tf.square(Y - output_layer)) / (2 * batchSize)
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)
#------------------------------------------------------------



#------------------------------------------------------------
# 랜덤값 구함
#------------------------------------------------------------
def randf(s, e):
	return (float(random.randrange(0, (e - s) * 9999)) / 10000) + s
#------------------------------------------------------------



#------------------------------------------------------------
# 오목 환경 클래스
#------------------------------------------------------------
class OmokEnvironment():

	#--------------------------------
	# 초기화
	#--------------------------------
	def __init__(self, gridSize):
		self.gridSize = gridSize
		self.nbStates = self.gridSize * self.gridSize
		self.state = np.zeros(self.nbStates, dtype = np.uint8)



	#--------------------------------
	# 리셋
	#--------------------------------
	def reset(self):
		self.state = np.zeros(self.nbStates, dtype = np.uint8)



	#--------------------------------
	# 현재 상태 구함
	#--------------------------------
	def getState(self):
		return np.reshape(self.state, (1, self.nbStates))



	#--------------------------------
	# 플레이어가 바뀐 현재 상태 구함
	#--------------------------------
	def getStateInverse(self):
		tempState = self.state.copy()
		
		for i in xrange(self.nbStates):
			if( tempState[i] == STONE_PLAYER1 ):
				tempState[i] = STONE_PLAYER2
			elif( tempState[i] == STONE_PLAYER2 ):
				tempState[i] = STONE_PLAYER1
		
		return np.reshape(tempState, (1, self.nbStates))



	#--------------------------------
	# 리워드 구함
	#--------------------------------
	def GetReward(self, player, action):
	
		# 왼쪽 검사
		if( action % self.gridSize > 0 ):
			if( self.state[action - 1] == player ):
				return 0.05

		# 오른쪽 검사
		if( action % self.gridSize < self.gridSize - 1 ):
			if( self.state[action + 1] == player ):
				return 0.05

		# 위 검사
		if( action - self.gridSize >= 0 ):
			if( self.state[action - self.gridSize] == player ):
				return 0.05

		# 아래 검사
		if( action + self.gridSize < self.nbStates ):
			if( self.state[action + self.gridSize] == player ):
				return 0.05

		# 왼쪽 위 검사
		if( (action % self.gridSize > 0) and (action - self.gridSize >= 0) ):
			if( self.state[action - 1 - self.gridSize] == player ):
				return 0.05

		# 오른쪽 위 검사
		if( (action % self.gridSize < self.gridSize - 1) and (action - self.gridSize >= 0) ):
			if( self.state[action + 1 - self.gridSize] == player ):
				return 0.05

		# 왼쪽 아래 검사
		if( (action % self.gridSize > 0) and (action + self.gridSize < self.nbStates) ):
			if( self.state[action - 1 + self.gridSize] == player ):
				return 0.05

		# 오른쪽 아래 검사
		if( (action % self.gridSize < self.gridSize - 1) and (action + self.gridSize < self.nbStates) ):
			if( self.state[action + 1 + self.gridSize] == player ):
				return 0.05
		
		return 0



	#--------------------------------
	# 매칭 검사
	#--------------------------------
	def CheckMatch(self, player):
		for y in xrange(self.gridSize):
			for x in xrange(self.gridSize):
			
				#--------------------------------
				# 오른쪽 검사
				#--------------------------------
				match = 0
				
				for i in xrange(STONE_MAX):
					if( x + i >= self.gridSize ):
						break
	
					if( self.state[y * self.gridSize + x + i] == player ):
						match += 1
					else:
						break;

					if( match >= STONE_MAX ):
						return True

				#--------------------------------
				# 아래쪽 검사
				#--------------------------------
				match = 0
				
				for i in xrange(STONE_MAX):
					if( y + i >= self.gridSize ):
						break
	
					if( self.state[(y + i) * self.gridSize + x] == player ):
						match += 1
					else:
						break;

					if( match >= STONE_MAX ):
						return True

				#--------------------------------
				# 오른쪽 대각선 검사
				#--------------------------------
				match = 0
				
				for i in xrange(STONE_MAX):
					if( (x + i >= self.gridSize) or (y + i >= self.gridSize) ):
						break
	
					if( self.state[(y + i) * self.gridSize + x + i] == player ):
						match += 1
					else:
						break;

					if( match >= STONE_MAX ):
						return True

				#--------------------------------
				# 왼쪽 대각선 검사
				#--------------------------------
				match = 0
				
				for i in xrange(STONE_MAX):
					if( (x - i < 0) or (y + i >= self.gridSize) ):
						break
	
					if( self.state[(y + i) * self.gridSize + x - i] == player ):
						match += 1
					else:
						break;

					if( match >= STONE_MAX ):
						return True
	
		return False



	#--------------------------------
	# 게임오버 검사
	#--------------------------------
	def isGameOver(self, player):
		if( self.CheckMatch(STONE_PLAYER1) == True ):
			if( player == STONE_PLAYER1 ):
				return True, winReward
			else:
				return True, 0
		elif( self.CheckMatch(STONE_PLAYER2) == True ):
			if( player == STONE_PLAYER1 ):
				return True, 0
			else:
				return True, winReward
		else:
			for i in xrange(self.nbStates):
				if( self.state[i] == STONE_NONE ):
					return False, 0
			return True, 0
				


	#--------------------------------
	# 상태 업데이트
	#--------------------------------
	def updateState(self, player, action):
		self.state[action] = player;



	#--------------------------------
	# 행동 수행
	#--------------------------------
	def act(self, player, action):
		self.updateState(player, action)
		gameOver, reward = self.isGameOver(player)
		
		if( reward == 0 ):
			reward = self.GetReward(player, action)
		
		if( player == STONE_PLAYER1 ):
			nextState = self.getState()
		else:
			nextState = self.getStateInverse()
		
		return nextState, reward, gameOver



	#--------------------------------
	# 행동 구함
	#--------------------------------
	def getAction(self, sess, currentState):
		q = sess.run(output_layer, feed_dict = {X: currentState})
		
		while( True ):
			action = q.argmax()

			if( self.state[action] == STONE_NONE ):
				return action
			else:
				q[0, action] = -99999
	


	#--------------------------------
	# 랜덤 행동 구함
	#--------------------------------
	def getActionRandom(self):
		while( True ):
			action = random.randrange(0, nbActions)

			if( self.state[action] == STONE_NONE ):
				return action
#------------------------------------------------------------



#------------------------------------------------------------
# 리플레이 메모리 클래스
#------------------------------------------------------------
class ReplayMemory:

	#--------------------------------
	# 초기화
	#--------------------------------
	def __init__(self, gridSize, maxMemory, discount):
		self.maxMemory = maxMemory
		self.gridSize = gridSize
		self.nbStates = self.gridSize * self.gridSize
		self.discount = discount
		
		self.inputState = np.empty((self.maxMemory, self.nbStates), dtype = np.uint8)
		self.actions = np.zeros(self.maxMemory, dtype = np.uint8)
		self.nextState = np.empty((self.maxMemory, self.nbStates), dtype = np.uint8)
		self.gameOver = np.empty(self.maxMemory, dtype = np.bool)
		self.rewards = np.empty(self.maxMemory, dtype = np.int8)
		self.count = 0
		self.current = 0



	#--------------------------------
	# 결과 기억
	#--------------------------------
	def remember(self, currentState, action, reward, nextState, gameOver):
		self.actions[self.current] = action
		self.rewards[self.current] = reward
		self.inputState[self.current, ...] = currentState
		self.nextState[self.current, ...] = nextState
		self.gameOver[self.current] = gameOver
		self.count = max(self.count, self.current + 1)
		self.current = (self.current + 1) % self.maxMemory



	#--------------------------------
	# 배치 구함
	#--------------------------------
	def getBatch(self, model, batchSize, nbActions, nbStates, sess, X):
		memoryLength = self.count
		chosenBatchSize = min(batchSize, memoryLength)
		
		inputs = np.zeros((chosenBatchSize, nbStates))
		targets = np.zeros((chosenBatchSize, nbActions))

		for i in xrange(chosenBatchSize):
			randomIndex = random.randrange(0, memoryLength)
			current_inputState = np.reshape(self.inputState[randomIndex], (1, nbStates))

			target = sess.run(model, feed_dict = {X: current_inputState})

			current_nextState = np.reshape(self.nextState[randomIndex], (1, nbStates))
			current_outputs = sess.run(model, feed_dict = {X: current_nextState})

			nextStateMaxQ = np.amax(current_outputs)

			if( nextStateMaxQ > winReward ):
				nextStateMaxQ = winReward
			
			if( self.gameOver[randomIndex] == True ):
				target[0, [self.actions[randomIndex]]] = self.rewards[randomIndex]
			else:
				target[0, [self.actions[randomIndex]]] = self.rewards[randomIndex] + self.discount * nextStateMaxQ

			inputs[i] = current_inputState
			targets[i] = target

		return inputs, targets
#------------------------------------------------------------



#------------------------------------------------------------
# 게임 플레이 함수
#------------------------------------------------------------
def playGame(env, memory, sess, saver, epsilon, iteration):

	#--------------------------------
	# 게임 반복
	#--------------------------------
	winCount = 0

	for i in xrange(epoch):
		env.reset()

		err = 0
		gameOver = False
		currentPlayer = STONE_PLAYER1
		
		while( gameOver != True ):
			#--------------------------------
			# 행동 수행
			#--------------------------------
			action = - 9999
			
			if( currentPlayer == STONE_PLAYER1 ):
				currentState = env.getState()
			else:
				currentState = env.getStateInverse()

			if( randf(0, 1) <= epsilon ):
				action = env.getActionRandom()
			else:
				action = env.getAction(sess, currentState)

			if( epsilon > epsilonMinimumValue ):
				epsilon = epsilon * epsilonDiscount
			
			nextState, reward, gameOver = env.act(currentPlayer, action)

			if( reward == 1 and currentPlayer == STONE_PLAYER1 ):
				winCount = winCount + 1

			#--------------------------------
			# 학습 수행
			#--------------------------------
			memory.remember(currentState, action, reward, nextState, gameOver)

			inputs, targets = memory.getBatch(output_layer, batchSize, nbActions, nbStates, sess, X)
			
			_, loss = sess.run([optimizer, cost], feed_dict = {X: inputs, Y: targets})
			err = err + loss
			
			if( currentPlayer == STONE_PLAYER1 ):
				currentPlayer = STONE_PLAYER2
			else:
				currentPlayer = STONE_PLAYER1

		print("Epoch " + str(iteration) + str(i) + ": err = " + str(err) + ": Win count = " + str(winCount) +
				" Win ratio = " + str(float(winCount) / float(i + 1) * 100))

		print(targets)

		if( (i % 10 == 0) and (i != 0) ):
			save_path = saver.save(sess, os.getcwd() + "/OmokModel.ckpt")
			print("Model saved in file: %s" % save_path)
#------------------------------------------------------------



#------------------------------------------------------------
# 메인 함수
#------------------------------------------------------------
def main(_):

	print("Training new model")

	# 환경 인스턴스 생성
	env = OmokEnvironment(gridSize)

	# 리플레이 메모리 인스턴스 생성
	memory = ReplayMemory(gridSize, maxMemory, discount)

	# 텐서플로우 초기화
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	# 세이브 설정
	saver = tf.train.Saver()

	# 모델 로드
	if( os.path.isfile(os.getcwd() + "/OmokModel.ckpt.index") == True ):
		saver.restore(sess, os.getcwd() + "/OmokModel.ckpt")
		print('Saved model is loaded!')
	
	# 게임 플레이
	iteration = 0
	while( True ):
		playGame(env, memory, sess, saver, epsilonStart, iteration);
		iteration += 1

	# 세션 종료
	sess.close()
#------------------------------------------------------------



#------------------------------------------------------------
# 메인 함수 실행
#------------------------------------------------------------
if __name__ == '__main__':
	tf.app.run()
#------------------------------------------------------------
















