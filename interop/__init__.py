"""
Interoperability (roadmap item 8): getting data in and out of DentAgent in the
formats other clinic systems speak.

  fhir.py   FHIR R4 (R4B) — export a patient as a Bundle; import a Patient/Bundle
  hl7.py    HL7 v2 ADT (A04 register / A08 update) — patient registration feeds
  dicom.py  DICOM images — convert and screen with the imaging agent
  csv_io.py CSV — the generic practice-management connector (patients, appointments)

Every import goes through the same duplicate check the app uses
(db.find_duplicate_candidates), so feeding the same patient twice reuses the chart.
"""
