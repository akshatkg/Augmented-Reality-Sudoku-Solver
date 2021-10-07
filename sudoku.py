import cv2
import numpy as np
import inspect, sys, re, operator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.python.ops.gen_array_ops import empty
#from model import Trainer
from solver import Solver
from skimage.segmentation import clear_border
import tensorflow as tf
import imutils
from imutils.perspective import four_point_transform


class Detector:
	def __init__(self):
		p = re.compile("stage_(?P<idx>[0-9]+)_(?P<name>[a-zA-Z0-9_]+)")

		self.stages = list(sorted(
		map(
			lambda x: (p.fullmatch(x[0]).groupdict()['idx'], p.fullmatch(x[0]).groupdict()['name'], x[1]),
			filter(
				lambda x: inspect.ismethod(x[1]) and p.fullmatch(x[0]),
				inspect.getmembers(self))),
		key=lambda x: x[0]))

		# For storing the recognized digits
		self.digits = [ [0 for i in range(9)] for j in range(9) ]
		
	# Takes as input 9x9 array of numpy images
	# Combines them into 1 image and returns
	# All 9x9 images need to be of same shape
	def makePreview(images):
		assert isinstance(images, list)
		assert len(images) > 0
		assert isinstance(images[0], list)
		assert len(images[0]) > 0
		assert isinstance(images[0], list)

		rows = len(images)
		cols = len(images[0])

		cellShape = images[0][0].shape

		padding = 10
		shape = (rows * cellShape[0] + (rows + 1) * padding, cols * cellShape[1] + (cols + 1) * padding)
		
		result = np.full(shape, 255, np.uint8)

		for row in range(rows):
			for col in range(cols):
				pos = (row * (padding + cellShape[0]) + padding, col * (padding + cellShape[1]) + padding)

				result[pos[0]:pos[0] + cellShape[0], pos[1]:pos[1] + cellShape[1]] = images[row][col]

		return result


	# Takes as input 9x9 array of digits
	# Prints it out on the console in the form of sudoku
	# None instead of number means that its an empty cell
	def showSudoku(array):
		cnt = 0
		for row in array:
			if cnt % 3 == 0:
				print('+-------+-------+-------+')

			colcnt = 0
			for cell in row:
				if colcnt % 3 == 0:
					print('| ', end='')
				print('. ' if cell is None else str(cell) + ' ', end='')
				colcnt += 1
			print('|')
			cnt += 1
		print('+-------+-------+-------+')

	# Runs the detector on the image at path, and returns the 9x9 solved digits
	# if show=True, then the stage results are shown on screen
	# Corrections is an array of the kind [(1,2,9), (3,3,4) ...] which implies
	# that the digit at (1,2) is corrected to 9
	# and the digit at (3,3) is corrected to 4
	# for sudoku2: corrections = [(7,5,9), (7,7,4)]
	def run(self, path='assets/sudokus/sudoku1.jpg', show = False, corrections = [(7,5,9), (7,7,4), (4,4,6), (6,4,4), (6,6,6)] ):
		self.path = path
		self.original = cv2.imread(path)

		self.run_stages(show)
		result = self.solve(corrections)


		if show:
			self.showSolved()
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		return result

	# Runs all the stages
	def run_stages(self, show):
		results = [('Original', self.original)]

		for idx, name, fun in self.stages:
			image = fun().copy()
			results.append((name, image))

		if show:
			for name, img in results:
				cv2.imshow(name, img)
		

	# Stages
	# Stage function name format: stage_[stage index]_[stage name]
	# Stages are executed increasing order of stage index
	# The function should return a numpy image, which is shown onto the screen
	# In case you have 81 images of 9x9 sudoku cells, you can use makePreview()
	# to create a single image out of those
	# You can pass data from one stage to another using class member variables
	def stage_1_preprocessing(self):
		self.gray = cv2.cvtColor(self.original.copy(), cv2.COLOR_BGR2GRAY)
		self.blur = cv2.GaussianBlur(self.gray, (9,9), 0)
		self.thresh = cv2.adaptiveThreshold(self.blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
		self.thresh = cv2.bitwise_not(self.thresh)

		self.image1 = self.thresh

		return self.thresh

	def stage_2_findcontours(self):
		# find contours in the thresholded image and sort them by size in
		# descending order
		cnts = cv2.findContours(self.image1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
		# initialize a contour that corresponds to the puzzle outline
		self.puzzleCnt = None
		# loop over the contours
		for c in cnts:
			# approximate the contour
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)
			# if our approximated contour has four points, then we can
			# assume we have found the outline of the puzzle
			if len(approx) == 4:
				self.puzzleCnt = approx
				break
		# if the puzzle contour is empty then our script could not find
		# the outline of the Sudoku puzzle so raise an error
		if self.puzzleCnt is None:
			raise Exception(("Could not find Sudoku puzzle outline. "
				"Try debugging your thresholding and contour steps."))
		
		# draw the contour of the puzzle on the image and then display
		# it to our screen for visualization/debugging purposes
		output = self.original.copy()
		cv2.drawContours(output, [self.puzzleCnt], -1, (0, 255, 0), 2)
		#cv2.imshow("Puzzle Outline", output)
		#cv2.waitKey(0)
		return output

	def stage_3_perspective(self):
		# apply a four point perspective transform to both the original
		# image and grayscale image to obtain a top-down bird's eye view
		# of the puzzle
		self.puzzle = four_point_transform(self.original, self.puzzleCnt.reshape(4, 2))
		self.warped = four_point_transform(self.gray, self.puzzleCnt.reshape(4, 2))
		
		# show the output warped image (again, for debugging purposes)
		#cv2.imshow("Puzzle Transform", puzzle)
		#cv2.waitKey(0)
		# return a 2-tuple of puzzle in both RGB and grayscale
		#return (puzzle, warped)
		return self.puzzle

	def stage_4_cellextraction(self):
		image = cv2.resize(self.image1, (28, 28))

		self.cells = [[image.copy() for i in range(9)] for j in range(9)]

		img = self.original
		img = imutils.resize(img, width=600)
		# find the puzzle in the image and then
		
		# a Sudoku puzzle is a 9x9 grid (81 individual cells), so we can
		# infer the location of each cell by dividing the warped image
		# into a 9x9 grid
		stepX = self.warped.shape[1] // 9
		stepY = self.warped.shape[0] // 9
		self.cellLocs = []
		# loop over the grid locations
		for y in range(0,9):
			r = []
			for x in range(0, 9):
				# compute the starting and ending (x, y)-coordinates of the
				# current cell
				startX = x * stepX
				startY = y * stepY
				endX = (x + 1) * stepX
				endY = (y + 1) * stepY
			    # add the (x, y)-coordinates to our cell locations list
				r.append((startX, startY, endX, endY))
				# crop the cell from the warped transform image and then
				cell = self.warped[startY:endY, startX:endX]
				# apply automatic thresholding to the cell and then clear any
				# connected borders that touch the border of the cell
				cell_thresh = cv2.threshold(cell, 0, 255,
					cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
				cell_thresh = clear_border(cell_thresh)
				cnts = cv2.findContours(cell_thresh.copy(), cv2.RETR_EXTERNAL,
				cv2.CHAIN_APPROX_SIMPLE)
				cnts = imutils.grab_contours(cnts)
				# if no contours were found than this is an empty cell
				if len(cnts) == 0:
					self.digits[y][x] = None
					self.cells[y][x] = cell_thresh
					
				# otherwise, find the largest contour in the cell and create a
				# mask for the contour
				else:
					c = max(cnts, key=cv2.contourArea)
					mask = np.zeros(cell_thresh.shape, dtype="uint8")
					cv2.drawContours(mask, [c], -1, 255, -1)

					# compute the percentage of masked pixels relative to the total
					# area of the image
					(h, w) = cell_thresh.shape
					percentFilled = cv2.countNonZero(mask) / float(w * h)
					# if less than 3% of the mask is filled then we are looking at
					# noise and can safely ignore the contour
					if percentFilled < 0.03:
						self.digits[y][x] = None
					# apply the mask to the thresholded cell
					self.cells[y][x] = cv2.bitwise_and(cell_thresh, cell_thresh, mask=mask)
			self.cellLocs.append(r)
		
		
		model = load_model(r'C:\Users\User\Desktop\sudoku-solver-2-akshatkg\assets\model')
		
		for y in range(0,9):
			for x in range(0,9): 
				# verify that the digit is not empty
				if self.digits[y][x] is not None:
					# resize the cell to 28x28 pixels and then prepare the
					# cell for classification
					self.cells[y][x] = np.array(self.cells[y][x])
					#cv2.imshow('digit', self.cells[y][x])
					#cv2.waitKey(0)
					roi = cv2.resize(self.cells[y][x], (28, 28))
					#print(type(roi))
					#print(roi.shape)
					roi = roi / 255.0
					#roi = roi.flatten()
					roi = img_to_array(roi)
					roi = np.expand_dims(roi, axis=0)

					# classify the digit and update the Sudoku board with the
					# prediction
					# pred = model.predict(roi).argmax(axis=1)[0]
					#pred = model.predict(roi)
					pred = model.predict(roi).argmax(axis=1)[0]
					#print(pred)
					self.digits[y][x] = pred	
			
		return Detector.makePreview(self.cells)


	# Solve function
	# Returns solution
	def solve(self, corrections):
		# Only upto 3 corrections allowed
		assert len(corrections) < 8
		self.solvable = False
		# Apply the corrections
		for i in range(len(corrections)):
			(y, x, n) = corrections[i]
			self.digits[y][x] = n
		#Detector.showSudoku(d.digits)
		# Solve the sudoku
		self.answers = [[ self.digits[j][i] for i in range(9) ] for j in range(9)]
		s = Solver(self.answers)
		if s.solve():
			self.solvable =True
			self.answers = s.digits
			return s.digits

		return [[None for i in range(9)] for j in range(9)]

	# Optional
	# Use this function to backproject the solved digits onto the original image
	# Save the image of "solved" sudoku into the 'assets/sudoku/' folder with
	# an appropriate name
	def showSolved(self):
		dig = np.array(self.digits).flatten()
		ans = np.array(self.answers).flatten()
		w = self.puzzle.shape[1] // 9
		h = self.puzzle.shape[0] // 9
		# for y in range(0,9):
		# 	for x in range(0,9):
		# 		if self.digits[y*9 + x] is not None:
		# 			cv2.putText(self.puzzle, str(self.digits[y*9 + x]), (x*w +int(w/2 - 10), int((y+0.8)*h)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255), 2, cv2.LINE_AA)
		black = np.zeros((self.puzzle.shape[0], self.puzzle.shape[1], 3), dtype = "uint8")
		for y in range(0,9):
			for x in range(0,9):
				if dig[y*9 + x] is None:
					cv2.putText(black, str(ans[y*9 + x]), (x*w +int(w/2 - 10), int((y+0.8)*h)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,180,0), 2, cv2.LINE_AA)

		points = np.float32(self.puzzleCnt).reshape(4,2) 
		pts2 = np.float32([points[1],points[0], points[2], points[3]]) 
		pts1 = np.float32([[0,0], [self.puzzle.shape[1], 0], [0,self.puzzle.shape[0]], [self.puzzle.shape[1],self.puzzle.shape[0]]])
		matrix = cv2.getPerspectiveTransform(pts1,pts2)
		img = cv2.warpPerspective(black,matrix,(self.original.shape[1],self.original.shape[0]))
		img = cv2.bitwise_not(img)
		img = cv2.bitwise_and(img,self.original)
		
		if self.solvable:
			cv2.imshow('a',img)
			cv2.waitKey(0)
			cv2.imwrite(r"C:\Users\User\Desktop\sudoku-solver-2-akshatkg\assets\sudokus\sudoku1_solved.jpg", img)
		else: 
			print('sudoku is not solvable')
		pass


if __name__ == '__main__':
	d = Detector()
	
	result = d.run('assets/sudokus/sudoku2.jpg', show=True)
	print('Recognized Sudoku:')
	Detector.showSudoku(d.digits)
	print('\n\nSolved Sudoku:')
	Detector.showSudoku(result)
	#print(d.digits)