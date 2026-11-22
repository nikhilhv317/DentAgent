"""FHIR export/import, HL7 ADT, CSV import and DICOM conversion."""
import io

import numpy as np
import pytest

import db


def test_fhir_export_is_valid_and_complete(client, admin):
    b = client.get("/api/patients/1/fhir", headers=admin).json()   # Ramesh: Penicillin; 2 conditions; 3 visits
    kinds = [e["resource"]["resourceType"] for e in b["entry"]]
    assert b["resourceType"] == "Bundle" and kinds.count("Patient") == 1
    assert kinds.count("AllergyIntolerance") == 1 and kinds.count("Condition") == 2
    assert kinds.count("Encounter") >= 3
    allergy = next(e["resource"] for e in b["entry"] if e["resource"]["resourceType"] == "AllergyIntolerance")
    assert allergy["code"]["text"] == "Penicillin"


def test_fhir_import_creates_then_reuses_chart(client, admin):
    pat = {"resourceType": "Patient", "name": [{"family": "Fernandes", "given": ["Maria"]}],
           "gender": "female", "birthDate": "1990-04-02", "telecom": [{"system": "phone", "value": "9811122233"}]}
    bundle = {"resourceType": "Bundle", "type": "collection", "entry": [
        {"resource": pat},
        {"resource": {"resourceType": "AllergyIntolerance", "patient": {"reference": "Patient/x"},
                      "code": {"text": "Lidocaine"}}}]}
    first = client.post("/api/interop/fhir", json=bundle, headers=admin).json()
    assert first["created"] is True and db.get_patient(first["patient_id"])["allergies"] == "Lidocaine"
    again = client.post("/api/interop/fhir", json=pat, headers=admin).json()
    assert again == {"patient_id": first["patient_id"], "created": False}
    assert client.post("/api/interop/fhir", json={"resourceType": "Patient", "gender": "nonsense"},
                       headers=admin).status_code == 400


def test_hl7_adt_a04_registers_patient_and_acks(client, admin):
    msg = ("MSH|^~\\&|HIS|HOSP|DENTAGENT|CLINIC|20261121093000||ADT^A04|MSG0001|P|2.5\r"
           "PID|1||H12345^^^HOSP||Shah^Rohan||19850312|M|||||9876501234\r"
           "AL1|1|DA|^Penicillin\r"
           "DG1|1||E11^Type 2 diabetes\r")
    r = client.post("/api/interop/hl7", content=msg, headers=admin).json()
    p = db.get_patient(r["patient_id"])
    assert p["name"] == "Rohan Shah" and p["sex"] == "M" and p["allergies"] == "Penicillin"
    assert p["conditions"] == "Type 2 diabetes" and "MSA|AA|MSG0001" in r["ack"]
    bad = client.post("/api/interop/hl7", content="MSH|^~\\&|X|Y|||1||ORU^R01|9|P|2.5\r", headers=admin)
    assert bad.status_code == 400 and "MSA|AE" in bad.json()["detail"]["ack"]


def test_hl7_and_csv_need_admin(client, staff):
    assert client.post("/api/interop/hl7", content="MSH|", headers=staff).status_code == 403


def test_csv_patients_and_appointments(client, admin):
    pts = "Patient Name,Mobile,Age,Gender,Allergies\nAsha Rao,9000011111,40,female,\n,9000022222,30,M,\nBad Age,9000033333,forty,M,\n"
    r = client.post("/api/interop/csv/patients", files={"file": ("p.csv", pts)}, headers=admin).json()
    assert r["created"] == 1 and [e["row"] for e in r["errors"]] == [3, 4]
    appts = ("name,phone,date,time,type\nAsha Rao,9000011111,2030-01-15,10:30,cleaning\n"
             "Asha Rao,9000011111,2030-01-15,10:30,cleaning\nAsha Rao,9000011111,15/01/2030,11:00,whitening\n")
    r = client.post("/api/interop/csv/appointments", files={"file": ("a.csv", appts)}, headers=admin).json()
    assert r["booked"] == 1 and r["duplicates_skipped"] == 1 and r["errors"][0]["row"] == 4


def test_dicom_conversion(tmp_path):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    from interop import dicom
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.3"   # digital intra-oral X-ray
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(tmp_path / "x.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
    arr = (np.linspace(0, 4095, 64 * 48).reshape(48, 64)).astype(np.uint16)
    ds.Rows, ds.Columns, ds.SamplesPerPixel = 48, 64, 1
    ds.PhotometricInterpretation, ds.BitsAllocated, ds.BitsStored = "MONOCHROME2", 16, 12
    ds.HighBit, ds.PixelRepresentation, ds.Modality = 11, 0, "IO"
    ds.PixelData = arr.tobytes()
    ds.save_as(str(tmp_path / "x.dcm"), enforce_file_format=True)
    img, info = dicom.to_image(str(tmp_path / "x.dcm"))
    px = np.asarray(img)
    assert img.size == (64, 48) and info["modality"] == "IO"
    assert px[0, 0, 0] == 0 and px[-1, -1, 0] == 255      # full grey range after windowing
