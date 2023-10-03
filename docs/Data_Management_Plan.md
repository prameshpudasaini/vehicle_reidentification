This Data Management Plan (DMP) outlines how data will be managed (collected, stored, shared, and preserved) in this research project.

## Data Collection and Acquisition

* Data Source: High-resolution events dataset (signal phase change and detector actuation events) collected from loop detectors on Indian School Rd & 19th Ave in Phoenix, AZ.
* Collection Method: Data was collected from City of Phoenix's TranSuite platform where the events dataset were archived.
* Data Types: Daily archives of high-resolution events in zipped csv format.
* Data Volume: 7 hours of data for testing the matching acccuracy; 2 weeks of data to train + validate the travel time prediction model.

## Data Storage

* Storage Location: Raw data are stored in CATS cloud-based server; processed data are stored in this GitHub repository.
* Repository Structure: Three folders for script, data, and output.
* File Naming Convention: yyyy_mm_dd_HH

## Data Preprocessing

* Data Cleaning: refer to the script folder

## Data Analysis

* Algorithms and Models
  * algorithm to process events 
  * algoritm to generate candidate match pairs
  * ML models to predict travel time
  * optimization framework to reidentify vehicles

* Parameter Tuning: Tuning model hyperparameters are discussed in the ML_models script.

## Metadata and Documentation

* Documentation: Code documentation is available as comments in the scripts.

## Data Sharing and Access

* Access Permissions: As per the LICENSE for this repo.
* Licensing: As per the LICENSE for this repo.
* Data Sharing Policy: As per the LICENSE for this repo.

## Ethical Considerations

* Privacy: 
* Consent: 

## Data Dissemination

* Publication: Research still ongoing.

This DMP will be reviewed and updated as the project progresses, data evolves, and new ethical considerations emerge.
