from fastapi import APIRouter
from app.database.db import appointments_collection

router = APIRouter()


@router.post("/book")
async def book_appointment(data: dict):

    appointments_collection.insert_one(data)

    return {
        "message": "Appointment booked"
    }