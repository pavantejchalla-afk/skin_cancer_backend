from fastapi import APIRouter
from app.database.db import reports_collection

router = APIRouter()


@router.post("/save")
async def save_report(report: dict):

    reports_collection.insert_one(report)

    return {
        "message": "Report saved"
    }


@router.get("/")
async def get_reports():

    reports = list(
        reports_collection.find({}, {"_id": 0})
    )

    return reports