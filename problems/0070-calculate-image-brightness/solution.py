
def calculate_brightness(img):
	if img==[]:
		return -1
	init_len = len(img[0])
	if init_len==0:
		return -1
	for row in img:
		if len(row)!=init_len:
			return -1
	ans=0
	
	for row in img:
		for val in row:
			if val>255 or val<0:
				return -1
			ans+=val

	return ans/(len(img)*init_len)
