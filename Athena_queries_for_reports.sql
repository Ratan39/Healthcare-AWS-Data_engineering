##Reports for cardio vascular disease

with observations as (
SELECT
    patient_id,
    code_display,
    value_quantity,
    ROW_NUMBER() OVER (PARTITION BY patient_id, code_display ORDER BY effective_datetime DESC) AS rn
  FROM healthcare_curated.observation
  WHERE code_display IN (
    'Cholesterol in HDL [Mass/volume] in Serum or Plasma',
    'Low Density Lipoprotein Cholesterol',
    'Triglycerides',
    'Cholesterol [Mass/volume] in Serum or Plasma'
  )
  ),
  Pivot as (
  SELECT
    patient_id,
    MAX(CASE WHEN code_display = 'Cholesterol in HDL [Mass/volume] in Serum or Plasma'
             THEN TRY_CAST(value_quantity AS DOUBLE) END) AS hdl,
    MAX(CASE WHEN code_display = 'Low Density Lipoprotein Cholesterol'
             THEN TRY_CAST(value_quantity AS DOUBLE) END) AS ldl,
    MAX(CASE WHEN code_display = 'Triglycerides'
             THEN TRY_CAST(value_quantity AS DOUBLE) END) AS trig,
    MAX(CASE WHEN code_display = 'Cholesterol [Mass/volume] in Serum or Plasma'
             THEN TRY_CAST(value_quantity AS DOUBLE) END) AS total_chol
  FROM observations
  WHERE rn = 1
  GROUP BY patient_id
 )
 SELECT
  patient_id AS patient,

  hdl,
  CASE
    WHEN hdl IS NULL              THEN 'n/a'
    WHEN hdl >= 60                THEN 'Protective'
    WHEN hdl BETWEEN 40 AND 59    THEN 'Normal'
    WHEN hdl < 40                 THEN 'Low'
  END AS hdl_status,

  ldl,
  CASE
    WHEN ldl IS NULL              THEN 'n/a'
    WHEN ldl >= 160               THEN 'High'
    WHEN ldl BETWEEN 130 AND 159  THEN 'Borderline'
    WHEN ldl BETWEEN 100 AND 129  THEN 'Near optimal'
    WHEN ldl < 100                THEN 'Optimal'
  END AS ldl_status,

  trig,
  CASE
    WHEN trig IS NULL             THEN 'n/a'
    WHEN trig >= 200              THEN 'High'
    WHEN trig BETWEEN 150 AND 199 THEN 'Borderline'
    WHEN trig < 150               THEN 'Normal'
  END AS triglycerides_status,

  total_chol,
  CASE
    WHEN total_chol IS NULL       THEN 'n/a'
    WHEN total_chol >= 240        THEN 'High'
    WHEN total_chol BETWEEN 200 AND 239 THEN 'Borderline'
    WHEN total_chol < 200         THEN 'Desirable'
  END AS total_chol_status,

  CASE
    WHEN ldl >= 130 OR trig >= 150 OR hdl < 40 OR total_chol >= 240
      THEN 'At risk'
    WHEN hdl IS NULL AND ldl IS NULL AND trig IS NULL AND total_chol IS NULL
      THEN 'Insufficient data'
    ELSE 'Likely normal'
  END AS overall_cvd_risk

FROM pivot;

#Report for pre diabetes

WITH observations AS (
  SELECT
    patient_id,
    code_display,
    value_quantity,          -- numeric lab values
    value_string,            -- qualitative text like 'Positive', 'Negative', 'Trace'
    ROW_NUMBER() OVER (
      PARTITION BY patient_id, code_display
      ORDER BY effective_datetime DESC
    ) AS rn
  FROM healthcare_curated.observation
  WHERE code_display IN (
    'Hemoglobin A1c/Hemoglobin.total in Blood',
    'Glucose [Mass/volume] in Blood',
    'Glucose [Mass/volume] in Urine by Test strip',
    'Glucose [Presence] in Urine by Test strip'
  )
),
pivot AS (
  SELECT
    patient_id,
    MAX(CASE WHEN code_display = 'Hemoglobin A1c/Hemoglobin.total in Blood'
             THEN TRY_CAST(value_quantity AS DOUBLE) END) AS a1c,
    MAX(CASE WHEN code_display = 'Glucose [Mass/volume] in Blood'
             THEN TRY_CAST(value_quantity AS DOUBLE) END) AS glucose_blood,
    -- Last qualitative urine glucose result (prefer a non-null textual value)
    MAX(CASE WHEN code_display IN ('Glucose [Mass/volume] in Urine by Test strip',
                                   'Glucose [Presence] in Urine by Test strip')
             THEN LOWER(TRIM(value_string)) END) AS glucose_urine_txt
  FROM observations
  WHERE rn = 1
  GROUP BY patient_id
)
SELECT
  patient_id AS patient,

  a1c,
  CASE
    WHEN a1c IS NULL THEN 'n/a'
    WHEN a1c >= 6.5 THEN 'Diabetes'
    WHEN a1c >= 5.7 THEN 'Prediabetes'
    ELSE 'Normal'
  END AS a1c_status,

  glucose_blood,
  CASE
    WHEN glucose_blood IS NULL THEN 'n/a'
    WHEN glucose_blood >= 126 THEN 'Diabetes'
    WHEN glucose_blood BETWEEN 100 AND 125 THEN 'Prediabetes'
    WHEN glucose_blood BETWEEN 70 AND 99 THEN 'Normal'
    WHEN glucose_blood < 70 THEN 'Low'
  END AS glucose_blood_status,

  glucose_urine_txt,
  CASE
    WHEN glucose_urine_txt IS NULL                  THEN 'n/a'
    WHEN glucose_urine_txt IN ('positive','pos')    THEN 'Abnormal'
    WHEN glucose_urine_txt IN ('trace')             THEN 'Borderline'
    WHEN glucose_urine_txt IN ('negative','neg')    THEN 'Normal'
    ELSE 'n/a'
  END AS glucose_urine_status,

  -- Overall T2D interpretation (prioritize diagnostic signals)
  CASE
    WHEN a1c >= 6.5 OR glucose_blood >= 126 OR glucose_urine_txt IN ('positive','pos')
      THEN 'Diabetes likely (lab criteria met)'
    WHEN (a1c BETWEEN 5.7 AND 6.4) OR (glucose_blood BETWEEN 100 AND 125) OR glucose_urine_txt = 'trace'
      THEN 'Prediabetes / Elevated risk'
    WHEN a1c IS NULL AND glucose_blood IS NULL AND glucose_urine_txt IS NULL
      THEN 'Insufficient data'
    ELSE 'Normal'
  END AS overall_t2d_risk
FROM pivot