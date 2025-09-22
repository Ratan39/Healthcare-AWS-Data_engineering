CREATE DATABASE healthcare_curated;

CREATE EXTERNAL TABLE healthcare_curated.patient (
  patient_id    string,
  gender        string,
  birth_date    string,
  race          string,
  ethnicity     string,
  address_line  string,
  address_city  string,
  address_state string,
  address_postal string,
  country       string,
  geo_lat       double,
  geo_lon       double
)
STORED AS PARQUET
LOCATION 's3://healthcare-json-curated/patient/';


CREATE EXTERNAL TABLE healthcare_curated.observation (
  observation_id     string,
  status             string,
  category           string,
  code_system        string,
  code               string,
  code_display       string,
  value_quantity     double,
  value_unit         string,
  value_string       string,
  effective_datetime string,
  patient_id         string,
  encounter_id       string
)
STORED AS PARQUET
LOCATION 's3://healthcare-json-curated/observation/';

CREATE EXTERNAL TABLE healthcare_curated.condition (
  condition_id        string,
  code_system         string,
  code                string,
  code_display        string,
  clinical_status     string,
  verification_status string,
  onset_datetime      string,
  recorded_datetime   string,
  patient_id          string,
  encounter_id        string
)
STORED AS PARQUET
LOCATION 's3://healthcare-json-curated/condition/';

CREATE EXTERNAL TABLE healthcare_curated.encounter (
  encounter_id      string,
  status            string,
  class             string,
  type_text         string,
  period_start      string,
  period_end        string,
  location_name     string,
  service_provider  string,
  practitioner_name string,
  practitioner_role string,
  patient_id        string
)
STORED AS PARQUET
LOCATION 's3://healthcare-json-curated/encounter/';


select * from healthcare_curated.patient;
select * from healthcare_curated.condition;
select * from healthcare_curated.observation;
select * from healthcare_curated.encounter;

select * from healthcare_curated.observation
order by effective_datetime;
where effective_datetime between '2023-01-01T00:00:00-00:00' and '2023-12-31T00:00:00-00:00';

select count(value_quantity) as num_of_values from healthcare_curated.observation
where code_display = 'Hematocrit [Volume Fraction] of Blood by Automated count' and value_quantity < 70;

select value_quantity from healthcare_curated.observation
where code_display = 'Hematocrit [Volume Fraction] of Blood by Automated count' and value_quantity < 70;


select value_quantity from healthcare_curated.observation
where code_display = 'Hemoglobin [Presence] in Urine by Test strip' and value_quantity < 180;

select value_quantity from healthcare_curated.observation
where code_display = 'Cholesterol in HDL [Mass/volume] in Serum or Plasma' and value_quantity < 70;

select value_quantity from healthcare_curated.observation
where code_display = 'Glucose [Mass/volume] in Blood' and value_quantity < 2000