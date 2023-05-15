import json
import os

# Get project name from current script name and remove extension
project_name = os.path.splitext(os.path.basename(__file__))[0]
# a Python object (dict):
project_obj = {
	"title": "Biscayne Bay Water Quality Monitoring Project from October 4th, 2018 to Present by Florida International University (FIU)",
	"directory_title": project_name,
	"abstract": "Data in this collection include water quality measurements gathered from a fleet of state-of-the-art and built-in-house robotic platforms deployed to Biscayne Bay of South Florida. Remote sensing images of the bay are also included. The data is organized in 4 levels. Level 1 contains data that is processed by it's originating source. Level 2 data contains data that was processed using third party software or custom computer programs. Level 3 data contains fused data. Level 4 data contains applications developed using data from previous levels.",
	"purpose": "This dataset was developed as part of a research project investigating machine learning applications in coastal waters such as Biscaybe Bay. Biscayne Bay has experienced a number of harmful water events in recent years and FIU, a nearby university, has expressed research interest for reasons such as marine life and economic conservation.",
	"point_of_contact": "",
	"principal_investigator": "",
	"originator":"Cesar A. Rojas,croja022@fiu.edu",
	"additional_contacts":[],
	"topic_category": "Inland Waters",
	"presentation_form": "Table Digital: spreadsheets, csv files, json, TIFF, netcdf",
	"keywords": ["MARINE", "ENVIRONMENT", "MONITORING", "Water-based Platforms", "ATLANTIC", "OCEAN"],
	"taxonomic_information": "",
	"spatial_bounds": "northbiscaynebay.json",
	"time_periods": ["20181004 to Present"],
	"data_table_attributes": "",
	"lineage_statement": "",
	"data_consistency_report": "",
	"process_steps": "",
	"completeness_report": "",
	"status_and_Maintenance": "",
	"constraints": "",
	"metadata_info": ""
}
print(os.path)
# convert into JSON:
project_json = json.dumps(project_obj,indent=4, sort_keys=True)

print("Creating or updating " + project_name +".json" )
json_file = open(project_name+".json", "w")
json_file.write(project_json)
json_file.close()

if os.path.exists(project_name):
	print(project_name+ " already exists.")
else:
	print("Creating " + project_name)
	os.mkdir(project_name)
	directories = ["media","data","documents", "literature", "articles", "vehicles", "instruments", "source", "locations"]
	for directory in directories:
		os.mkdir(os.path.join(project_name,directory))	
	
	sub_directories = [
	"data/level_1/remote_sensing/landsat_8",
	"data/level_1/remote_sensing/sentinel_2",
	"data/level_1/in-situ/time_series",
	"data/level_1/in-situ/isolated",
	"data/level_2/remote_sensing/landsat_8",
	"data/level_2/remote_sensing/sentinel_2",
	"data/level_3",
	"data/level_4"
	]
	for directory in sub_directories:
		os.makedirs(os.path.join(project_name,directory))