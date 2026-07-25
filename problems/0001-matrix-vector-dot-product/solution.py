def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	if len(a[0])!=len(b):
		return -1
	result = []
	for row in a:
		val=0
		for val_a,val_b in zip(row,b):
			val += val_a*val_b
		result.append(val)

	return result