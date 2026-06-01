from fastapi import APIRouter

from app.database.db import users_collection


router = APIRouter()


@router.post("/signup")
async def signup(user: dict):

    users_collection.insert_one(user)

    return {
        "message": "User created successfully"
    }


@router.post("/login")
async def login(user: dict):

    existing = users_collection.find_one({
        "email": user["email"]
    })

    if not existing:

        return {
            "error": "User not found"
        }

    return {
        "message": "Login successful"
    }