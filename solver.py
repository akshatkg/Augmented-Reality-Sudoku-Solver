# Initialize this with the 9x9 array of digits
# None element in the 9x9 array represents empty cell
# solve() solves the sudoku and saves it in self.digits
# The solved sudoku can be accessed by the digits member of this class
class Solver:
	def __init__(self, digits):
		self.digits = digits

	# Solve function
	# Returns True if sudoku admits a solution
	# False otherwise
	# Solved sudoku can be found in self.digits

# A Backtracking program
# in Python to solve Sudoku problem


		
# Function to Find the entry in
# the Grid that is still not used
# Searches the grid to find an
# entry that is still unassigned. If
# found, the reference parameters
# row, col will be set the location
# that is unassigned, and true is
# returned. If no unassigned entries
# remains, false is returned.
# 'l' is a list variable that has
# been passed from the solve_sudoku function
# to keep track of incrementation
# of Rows and Columns
	def find_empty_location(self, arr, l):
		for row in range(9):
			for col in range(9):
				if(arr[row][col] is None):
					l[0]= row
					l[1]= col
					return True
		return False

	# Returns a boolean which indicates
	# whether any assigned entry
	# in the specified row matches
	# the given number.
	def used_in_row(self, arr, row, num):
		for i in range(9):
			if(arr[row][i] == num):
				return True
		return False

	# Returns a boolean which indicates
	# whether any assigned entry
	# in the specified column matches
	# the given number.
	def used_in_col(self, arr, col, num):
		for i in range(9):
			if(arr[i][col] == num):
				return True
		return False

	# Returns a boolean which indicates
	# whether any assigned entry
	# within the specified 3x3 box
	# matches the given number
	def used_in_box(self, arr, row, col, num):
		for i in range(3):
			for j in range(3):
				if(arr[i + row][j + col] == num):
					return True
		return False

	# Checks whether it will be legal
	# to assign num to the given row, col
	# Returns a boolean which indicates
	# whether it will be legal to assign
	# num to the given row, col location.
	def check_location_is_safe(self, arr, row, col, num):
		
		# Check if 'num' is not already
		# placed in current row,
		# current column and current 3x3 box
		return not self.used_in_row(self.digits, row, num) and not self.used_in_col(arr, col, num) and not self.used_in_box(arr, row - row % 3, col - col % 3, num)

	# Takes a partially filled-in grid
	# and attempts to assign values to
	# all unassigned locations in such a
	# way to meet the requirements
	# for Sudoku solution (non-duplication
	# across rows, columns, and boxes)
	def solve(self):
		
		# 'l' is a list variable that keeps the
		# record of row and col in
		# find_empty_location Function
		l =[0, 0]
		
		# If there is no unassigned
		# location, we are done
		if(not self.find_empty_location(self.digits, l)):
			return True
		
		# Assigning list values to row and col
		# that we got from the above Function
		row = l[0]
		col = l[1]
		
		# consider digits 1 to 9
		for num in range(1, 10):
			
			# if looks promising
			if(self.check_location_is_safe(self.digits,
							row, col, num)):
				
				# make tentative assignment
				self.digits[row][col]= num

				# return, if success,
				# ya !
				if(self.solve()):
					return True

				# failure, unmake & try again
				self.digits[row][col] = None
				
		# this triggers backtracking	
		return False


