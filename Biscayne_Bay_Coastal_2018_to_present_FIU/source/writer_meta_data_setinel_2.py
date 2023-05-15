import json

def write_meta_data(image_file_name, date_downloaded, geojson_file_name, source):
	# a Python object (dict):

	x = {
		"Download Date": date_downloaded,
		"Geojson File Name": geojson_file_name,
		"Satellite": 'Sentinel 2',
		"Source": source
	}
	
	
	# convert into JSON:
	y = json.dumps(x,indent=4, sort_keys=True)

	jsonFile = open("out/"+image_file_name+".json", "w")
	jsonFile.write(y)
	jsonFile.close()


def test():
	print('testing meta data writer')
	write_meta_data('Test','Test','Test','Test')

if __name__ == '__main__':
	test()