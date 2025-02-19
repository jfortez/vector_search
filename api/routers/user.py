from fastapi import APIRouter

user_router = APIRouter(prefix="/users", tags=["users"])


@user_router.get("/")
def read_users():
    return {"message": "Hello World"}