Dataset Title: Microdifuse reflectance spectra from Anopheles gambiae s.l
 

Principal Investigator: Mauro Pazmino, mauro.pazminobetancourth@glasgow.ac.uk
 

Description of the dataset:

FTIR measurements of mosquitoes samples using a nitrogen-purged Bruker Vertex70 with a Hyperion microscope (Bruker Corporation,USA) using a 15× reflective objective and liquid-nitrogen cooled mercury cadmium telluride (MCT) detector and Globar light source. Data extracted from Bruker OPUS 6.5 software in data point table format (tab seperated). Range: 600 to 4000 cm-1 and resolution of 4 cm-1. 

Attributes: Also see the Codes section
    Species: Mosquito species
    Age: Chronological age
    ID: Unique identifier
    Part: Part of the mosquito measured
    Sp Part: Specifict part of the mosquito measured

Codes: 
Species
		AK: Anopheles gambiae (kisumu strain)
		AC: Anopheles colluzi
		AR: Anopheles arabiensis

Age: 
		01D = 1 day old
        02D = 2 days old
		03D = 3 days old
        10D = 10 days old

Part:
		LG: Legs

Files: 
-Raw
- DRIFT_legs_2023: Raw data for species and age prediction
- DRIFT_Insecticide_resistnace.csv: Raw data for insecticide resistance and strain prediction

-Processed
Features and labels for predictions problems.
    Features file = X
    label file = y
    Prediction problem = age, species, status, strain 

Folder structure:
── Data                   
├── processed
 ├── X_age.csv
 ├── X_species.csv
 ├── X_status.csv
 ├── X_strains.csv
 ├── y_age.csv
 ├── y_species.csv
 ├── y_status.csv
 ├── y_strain.csv           
├── raw
 ├── DRIFT_Insecticide_resistance.csv
 ├── DRIFT_Legs_2023.csv                   
└── README.md

Species used: Anopheles gambiae (kisumu strain), Anopheles coluzzii (Ngousso strain) and Anopheles gambiae (Tiassale strain). Ages: 1 day old, 2 days old, 3 days old, 10 days old

Versioning: All changes to this dataset will be documented in a changelog in this