import sys, re
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import functions as F

# --------- Parameters ----------
args = getResolvedOptions(sys.argv, ["JOB_NAME", "raw_bucket", "curated_bucket", "temp_path"])
RAW = args["raw_bucket"].rstrip("/")        # e.g., s3://healthcare-json-raw/fhir
CUR = args["curated_bucket"].rstrip("/")    # e.g., s3://healthcare-json-curated
TMP = args["temp_path"].rstrip("/")         # e.g., s3://healthcare-json-temp/glue/tmp

# --------- Glue / Spark setup ----------
conf = SparkConf()  # no static conf changes after session starts
sc = SparkContext.getOrCreate(conf=conf)
glue = GlueContext(sc)
spark = glue.spark_session
job = Job(glue)
job.init(args["JOB_NAME"], args)

# --------- Helpers ----------
def ref_to_uuid(colname):
    # Extract the UUID from FHIR reference "urn:uuid:<id>"
    return F.regexp_extract(F.col(colname), r"urn:uuid:([A-Za-z0-9-]+)", 1)

def jget(colname, *json_paths):
    # Safely extract nested fields via JSON paths from a struct that might or might not have them
    exprs = [F.get_json_object(F.to_json(F.col(colname)), p) for p in json_paths]
    return F.coalesce(*exprs)

# CodeableConcept helpers that tolerate array-or-single shapes at both CC and coding level
def cc_attr(colname, attr):  # attr in {"system","code","display"}
    return F.coalesce(
        jget(colname, f"$.coding[0].{attr}"),
        jget(colname, f"$.coding.{attr}"),
        jget(colname, f"$[0].coding[0].{attr}"),
        jget(colname, f"$[0].coding.{attr}")
    )

def cc_text(colname):
    return F.coalesce(
        jget(colname, "$.text"),
        jget(colname, "$[0].text")
    )

def cc_display(colname):
    return cc_attr(colname, "display")

# --------- Read raw JSON (bookmarked) ----------
src_dyn = glue.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={"paths": [RAW], "recurse": True},
    format="json"
)
src = src_dyn.toDF()

# Flatten bundle entries safely
entries = (
    src.select(F.explode_outer("entry").alias("e"))
       .select(F.col("e.resource").alias("resource"))
)

# Also make a flattened view for convenience where it’s safe
flat = entries.select("resource.*")

# =============== PATIENT ===============
addr0 = F.col("address")[0]
pat = (
    flat.filter(F.col("resourceType") == "Patient")
        .select(
            F.col("id").alias("patient_id"),
            F.col("gender").alias("gender"),
            F.col("birthDate").alias("birth_date"),
            addr0["line"].alias("address_line_array"),
            addr0["city"].alias("address_city"),
            addr0["state"].alias("address_state"),
            addr0["postalCode"].alias("address_postal"),
            addr0["country"].alias("country"),
            addr0["extension"][0]["extension"][0]["valueDecimal"].alias("geo_lat"),
            addr0["extension"][0]["extension"][1]["valueDecimal"].alias("geo_lon"),
            F.col("extension")[0]["extension"][1]["valueString"].alias("race"),
            F.col("extension")[1]["extension"][1]["valueString"].alias("ethnicity"),
        )
        .withColumn(
            "address_line",
            F.when(F.col("address_line_array").isNotNull(),
                   F.array_join(F.col("address_line_array"), "|"))
        )
        .drop("address_line_array")
        .dropDuplicates(["patient_id"])
)

# =============== ENCOUNTER ===============
enc = (
    flat.filter(F.col("resourceType") == "Encounter")
        .select(
            F.col("id").alias("encounter_id"),
            F.col("subject.reference").alias("patient_ref"),
            F.col("status").alias("status"),
            F.col("class.code").alias("class_code"),
            cc_text("type").alias("type_text"),
            F.col("period.start").alias("period_start"),
            F.col("period.end").alias("period_end"),
            jget("location", "$[0].location.display").alias("location_name"),
            F.col("serviceProvider.display").alias("service_provider"),
            jget("participant", "$[0].individual.display").alias("practitioner_name"),
            F.coalesce(
                jget("participant", "$[0].type[0].text"),
                jget("participant", "$[0].type.text")
            ).alias("practitioner_role"),
        )
        .withColumn("patient_id", ref_to_uuid("patient_ref"))
        .drop("patient_ref")
        .dropDuplicates(["encounter_id"])
)

# =============== CONDITION ===============
con = (
    flat.filter(F.col("resourceType") == "Condition")
        .select(
            F.col("id").alias("condition_id"),
            F.col("subject.reference").alias("patient_ref"),
            F.col("encounter.reference").alias("encounter_ref"),
            cc_attr("code", "system").alias("code_system"),
            cc_attr("code", "code").alias("code"),
            cc_attr("code", "display").alias("code_display"),
            cc_attr("clinicalStatus", "code").alias("clinical_status"),
            cc_attr("verificationStatus", "code").alias("verification_status"),
            F.col("onsetDateTime").alias("onset_datetime"),
            F.col("recordedDate").alias("recorded_datetime"),
        )
        .withColumn("patient_id", ref_to_uuid("patient_ref"))
        .withColumn("encounter_id", ref_to_uuid("encounter_ref"))
        .drop("patient_ref", "encounter_ref")
        .dropDuplicates(["condition_id"])
)

# =============== OBSERVATION ===============
# Strong fix: handle Glue "choice type" by coalescing struct members (double/int) into one DOUBLE
value_qty = F.coalesce(
    F.col("resource.valueQuantity.value.double"),
    F.col("resource.valueQuantity.value.int").cast("double"),
    # Fallback via JSON path in case some rows didn't materialize as a choice struct
    jget("resource.valueQuantity", "$.value").cast("double")
)

obs = (
    entries.filter(F.col("resource.resourceType") == "Observation")
        .select(
            F.col("resource.id").alias("observation_id"),
            F.col("resource.subject.reference").alias("patient_ref"),
            F.col("resource.encounter.reference").alias("encounter_ref"),
            F.col("resource.status").alias("status"),
            cc_display("resource.category").alias("category"),
            cc_attr("resource.code", "system").alias("code_system"),
            cc_attr("resource.code", "code").alias("code"),
            cc_attr("resource.code", "display").alias("code_display"),

            value_qty.alias("value_quantity"),
            F.col("resource.valueQuantity.unit").alias("value_unit"),

            # Textual/other value[x] coalesced to a string (kept as-is)
            F.coalesce(
                jget("resource", "$.valueString"),
                cc_attr("resource.valueCodeableConcept", "display"),
                cc_text("resource.valueCodeableConcept"),
                jget("resource", "$.valueInteger"),
                jget("resource", "$.valueBoolean"),
                jget("resource", "$.valueDateTime"),
                jget("resource", "$.valueTime"),
                jget("resource", "$.valuePeriod.start"),
                jget("resource", "$.valueRange.low.value"),
                jget("resource", "$.valueSampledData.data")
            ).alias("value_string"),

            jget("resource", "$.effectiveDateTime").alias("effective_datetime"),
        )
        .withColumn("patient_id", ref_to_uuid("patient_ref"))
        .withColumn("encounter_id", ref_to_uuid("encounter_ref"))
        .drop("patient_ref", "encounter_ref")
        .dropDuplicates(["observation_id"])
)

# --------- Write Parquet (append; bookmarks ensure incrementality) ----------
def write_parquet(df, subpath, partitions=None):
    path = f"{CUR}/{subpath}"
    writer = df.write.mode("append")
    if partitions:
        writer = writer.partitionBy(*partitions)
    writer.parquet(path)

write_parquet(pat, "patient")
write_parquet(enc, "encounter")
write_parquet(con, "condition")
write_parquet(obs, "observation")

job.commit()
